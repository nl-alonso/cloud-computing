#!/usr/bin/env python3
"""
LLaMA LoRA Fine-tuning on Vertex AI with GCS Dataset Support
===========================================================

This script demonstrates how to fine-tune LLaMA models (including LLaMA 4 Scout) 
using Low Rank Adaptation (LoRA) on Google Cloud Vertex AI platform 
with datasets stored in Google Cloud Storage.
"""

import os
import json
import argparse
import logging
import tempfile
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset, load_dataset
import evaluate
from google.cloud import aiplatform
from google.cloud import storage
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoRATrainingConfig:
    """Configuration class for LoRA training parameters"""
    
    # Model parameters
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Default to LLaMA 3.1
    tokenizer_name: Optional[str] = None
    cache_dir: str = "./cache"
    local_model_path: Optional[str] = None  # For LLaMA 4 Scout local models
    use_local_model: bool = False  # Flag to use local LLaMA 4 Scout model
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"  # Updated for LLaMA 3.1+ architecture
    ])
    lora_bias: str = "none"
    
    # Training parameters
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    max_length: int = 512
    
    # Vertex AI parameters
    project_id: str = ""
    location: str = "us-central1"
    staging_bucket: str = ""
    
    # Dataset parameters - Updated for GCS support
    dataset_name: str = "alpaca"
    dataset_path: Optional[str] = None
    train_file: Optional[str] = None  # Can be GCS path: gs://bucket/train.json
    validation_file: Optional[str] = None  # Can be GCS path: gs://bucket/val.json
    gcs_dataset_bucket: Optional[str] = None  # GCS bucket for datasets
    gcs_dataset_prefix: Optional[str] = None  # Prefix for dataset files in GCS
    
    # Quantization parameters
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "llama-lora-finetuning"


class LLaMA4ScoutModelDetector:
    """Helper class to detect and validate LLaMA 4 Scout models"""
    
    @staticmethod
    def find_local_llama4_models() -> list:
        """Find locally downloaded LLaMA 4 Scout models"""
        possible_paths = [
            Path.home() / ".llama",
            Path.home() / ".cache" / "llama",
            Path("./models"),
            Path("./cache"),
            Path("/tmp/llama_models")
        ]
        
        found_models = []
        for base_path in possible_paths:
            if base_path.exists():
                for model_dir in base_path.iterdir():
                    if model_dir.is_dir():
                        # Check for common LLaMA model files
                        has_model_files = any([
                            (model_dir / "pytorch_model.bin").exists(),
                            (model_dir / "model.safetensors").exists(),
                            (model_dir / "consolidated.00.pth").exists(),
                            any(model_dir.glob("*.safetensors")),
                            any(model_dir.glob("*.bin"))
                        ])
                        
                        has_config = (model_dir / "config.json").exists()
                        has_tokenizer = any([
                            (model_dir / "tokenizer.model").exists(),
                            (model_dir / "tokenizer.json").exists()
                        ])
                        
                        if has_model_files and (has_config or has_tokenizer):
                            found_models.append(str(model_dir))
        
        return found_models
    
    @staticmethod
    def validate_model_path(model_path: str) -> bool:
        """Validate that a local model path contains necessary files"""
        path = Path(model_path)
        if not path.exists():
            return False
        
        # Check for essential files
        required_patterns = [
            "*.safetensors", "*.bin", "*.pth",  # Model weights
            "config.json",                      # Model config
            "tokenizer*"                        # Tokenizer files
        ]
        
        found_files = []
        for pattern in required_patterns:
            if list(path.glob(pattern)):
                found_files.append(pattern)
        
        return len(found_files) >= 2  # At least model weights and one other file


class GCSDatasetLoader:
    """Helper class for loading datasets from Google Cloud Storage"""
    
    def __init__(self, project_id: str):
        self.client = storage.Client(project=project_id)
    
    def download_file_from_gcs(self, gcs_path: str, local_path: Optional[str] = None) -> str:
        """Download a file from GCS to local storage"""
        if not gcs_path.startswith('gs://'):
            return gcs_path  # Already a local path
        
        # Parse GCS path
        gcs_path_clean = gcs_path.replace('gs://', '')
        bucket_name, blob_name = gcs_path_clean.split('/', 1)
        
        # Create local path if not provided
        if local_path is None:
            local_path = os.path.join(tempfile.gettempdir(), os.path.basename(blob_name))
        
        # Download the file
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        logger.info(f"Downloading {gcs_path} to {local_path}")
        blob.download_to_filename(local_path)
        
        return local_path
    
    def list_files_in_gcs(self, bucket_name: str, prefix: str = "") -> list:
        """List files in a GCS bucket with optional prefix"""
        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return [f"gs://{bucket_name}/{blob.name}" for blob in blobs if not blob.name.endswith('/')]


class LLaMALoRATrainer:
    """LLaMA LoRA Fine-tuning Trainer for Vertex AI with GCS support and LLaMA 4 Scout"""
    
    def __init__(self, config: LoRATrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.gcs_loader = None
        self.model_detector = LLaMA4ScoutModelDetector()
        
        # Initialize Vertex AI
        if config.project_id:
            aiplatform.init(project=config.project_id, location=config.location)
            self.gcs_loader = GCSDatasetLoader(config.project_id)
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with LoRA configuration"""
        
        # Handle local LLaMA 4 Scout models
        if self.config.use_local_model or self.config.local_model_path:
            if self.config.local_model_path:
                model_path = self.config.local_model_path
            else:
                # Auto-detect local models
                local_models = self.model_detector.find_local_llama4_models()
                if not local_models:
                    raise ValueError("No local LLaMA models found. Run download_llama4_scout.py first.")
                
                logger.info(f"Found {len(local_models)} local model(s):")
                for i, model in enumerate(local_models):
                    logger.info(f"  {i+1}. {model}")
                
                # Use the first found model or let user choose
                model_path = local_models[0]
                logger.info(f"Using model: {model_path}")
            
            if not self.model_detector.validate_model_path(model_path):
                raise ValueError(f"Invalid model path: {model_path}")
            
            logger.info(f"Loading local LLaMA model from: {model_path}")
            model_name_or_path = model_path
        else:
            logger.info(f"Loading model from Hugging Face: {self.config.model_name}")
            model_name_or_path = self.config.model_name
        
        # Configure quantization
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        tokenizer_name = self.config.tokenizer_name or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.use_4bit else torch.float32,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",  # Use Flash Attention 2 if available
        )
        
        # Prepare model for k-bit training if using quantization
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete")
    
    def prepare_dataset(self):
        """Prepare training and evaluation datasets with GCS support"""
        logger.info("Preparing datasets...")
        
        # Handle GCS dataset loading
        if self.config.gcs_dataset_bucket and self.config.gcs_dataset_prefix:
            # Load dataset from GCS bucket
            logger.info(f"Loading dataset from GCS bucket: {self.config.gcs_dataset_bucket}")
            files = self.gcs_loader.list_files_in_gcs(
                self.config.gcs_dataset_bucket, 
                self.config.gcs_dataset_prefix
            )
            
            train_files = [f for f in files if 'train' in os.path.basename(f)]
            val_files = [f for f in files if 'val' in os.path.basename(f) or 'test' in os.path.basename(f)]
            
            if train_files:
                local_train_file = self.gcs_loader.download_file_from_gcs(train_files[0])
                self.config.train_file = local_train_file
                
            if val_files:
                local_val_file = self.gcs_loader.download_file_from_gcs(val_files[0])
                self.config.validation_file = local_val_file
        
        # Download individual files from GCS if specified
        if self.config.train_file and self.config.train_file.startswith('gs://'):
            self.config.train_file = self.gcs_loader.download_file_from_gcs(self.config.train_file)
            
        if self.config.validation_file and self.config.validation_file.startswith('gs://'):
            self.config.validation_file = self.gcs_loader.download_file_from_gcs(self.config.validation_file)
        
        # Load dataset
        if self.config.train_file and self.config.validation_file:
            # Load custom dataset files
            dataset = load_dataset(
                "json",
                data_files={
                    "train": self.config.train_file,
                    "validation": self.config.validation_file
                }
            )
        elif self.config.dataset_path:
            # Load from Hugging Face or other path
            dataset = load_dataset(self.config.dataset_path)
        else:
            # Load default dataset (Alpaca-style)
            dataset = load_dataset("tatsu-lab/alpaca")
        
        def format_instruction_llama(example):
            """Format instruction data for LLaMA training with proper tokens"""
            if "instruction" in example:
                # Check if this looks like LLaMA 4 (might need different formatting)
                is_llama4 = self.config.use_local_model or "llama-4" in self.config.model_name.lower()
                
                if is_llama4:
                    # Use LLaMA 4 Scout format (similar to LLaMA 3 but may evolve)
                    if example.get("input", "").strip():
                        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}\n\nInput: {example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    else:
                        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    
                    response = example.get("output", "") + "<|eot_id|>"
                else:
                    # Use LLaMA 3.1 format
                    if example.get("input", "").strip():
                        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}\n\nInput: {example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    else:
                        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    
                    response = example.get("output", "") + "<|eot_id|>"
                
                return {"text": prompt + response}
            else:
                # Handle other dataset formats
                return {"text": example.get("text", "")}
        
        def tokenize_function(examples):
            """Tokenize the examples"""
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Format and tokenize datasets
        if "train" in dataset:
            train_data = dataset["train"].map(format_instruction_llama)
            self.train_dataset = train_data.map(
                tokenize_function,
                batched=True,
                remove_columns=train_data.column_names
            )
        
        if "validation" in dataset or "test" in dataset:
            eval_key = "validation" if "validation" in dataset else "test"
            eval_data = dataset[eval_key].map(format_instruction_llama)
            self.eval_dataset = eval_data.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_data.column_names
            )
        else:
            # Split train dataset for evaluation
            split_dataset = self.train_dataset.train_test_split(test_size=0.1)
            self.train_dataset = split_dataset["train"]
            self.eval_dataset = split_dataset["test"]
        
        logger.info(f"Training dataset size: {len(self.train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(self.eval_dataset) if self.eval_dataset else 0}")
    
    def setup_training_arguments(self):
        """Setup training arguments"""
        model_type = "llama4" if (self.config.use_local_model or "llama-4" in self.config.model_name.lower()) else "llama3"
        
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model="eval_loss" if self.eval_dataset else None,
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=f"{model_type}-lora-{self.config.model_name.split('/')[-1]}",
            gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        )
    
    def train(self):
        """Execute the training process"""
        logger.info("Starting training...")
        
        # Initialize wandb if enabled
        if self.config.use_wandb:
            model_type = "llama4" if (self.config.use_local_model or "llama-4" in self.config.model_name.lower()) else "llama3"
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__,
                name=f"{model_type}-lora-{self.config.model_name.split('/')[-1]}"
            )
        
        # Setup trainer
        training_args = self.setup_training_arguments()
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if self.eval_dataset else None,
        )
        
        # Start training
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Upload to GCS if staging bucket is provided
        if self.config.staging_bucket:
            self.upload_to_gcs()
        
        logger.info("Training completed!")
    
    def upload_to_gcs(self):
        """Upload trained model to Google Cloud Storage"""
        logger.info(f"Uploading model to GCS bucket: {self.config.staging_bucket}")
        
        client = storage.Client(project=self.config.project_id)
        bucket = client.bucket(self.config.staging_bucket.replace("gs://", ""))
        
        model_type = "llama4" if (self.config.use_local_model or "llama-4" in self.config.model_name.lower()) else "llama3"
        
        for root, dirs, files in os.walk(self.config.output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                blob_path = os.path.relpath(local_path, self.config.output_dir)
                blob = bucket.blob(f"{model_type}-lora-model/{blob_path}")
                blob.upload_from_filename(local_path)
        
        logger.info("Model uploaded to GCS successfully!")
    
    def inference_example(self, prompt: str, max_length: int = 200):
        """Example inference with the fine-tuned model using appropriate format"""
        if self.model is None:
            logger.error("Model not loaded. Please run training first.")
            return None
        
        # Determine model type for formatting
        is_llama4 = self.config.use_local_model or "llama-4" in self.config.model_name.lower()
        
        # Format prompt appropriately
        if is_llama4:
            # LLaMA 4 Scout format (may evolve)
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # LLaMA 3.1 format
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(formatted_prompt):]


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="LLaMA 3.1/4 Scout LoRA Fine-tuning on Vertex AI with GCS Support")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--local_model_path", type=str, help="Path to local LLaMA 4 Scout model")
    parser.add_argument("--use_local_model", action="store_true", help="Use locally downloaded LLaMA 4 Scout model")
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--staging_bucket", type=str, help="GCS bucket for model storage")
    parser.add_argument("--gcs_dataset_bucket", type=str, help="GCS bucket containing datasets")
    parser.add_argument("--gcs_dataset_prefix", type=str, help="Prefix for dataset files in GCS")
    parser.add_argument("--train_file", type=str, help="Training file path (local or GCS)")
    parser.add_argument("--validation_file", type=str, help="Validation file path (local or GCS)")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = LoRATrainingConfig(**config_dict)
    else:
        config = LoRATrainingConfig()
    
    # Override with command line arguments
    config.model_name = args.model_name
    config.project_id = args.project_id
    if args.local_model_path:
        config.local_model_path = args.local_model_path
        config.use_local_model = True
    if args.use_local_model:
        config.use_local_model = True
    if args.staging_bucket:
        config.staging_bucket = args.staging_bucket
    if args.gcs_dataset_bucket:
        config.gcs_dataset_bucket = args.gcs_dataset_bucket
    if args.gcs_dataset_prefix:
        config.gcs_dataset_prefix = args.gcs_dataset_prefix
    if args.train_file:
        config.train_file = args.train_file
    if args.validation_file:
        config.validation_file = args.validation_file
    config.num_train_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.lora_r = args.lora_r
    config.lora_alpha = args.lora_alpha
    config.output_dir = args.output_dir
    
    # Initialize trainer
    trainer = LLaMALoRATrainer(config)
    
    # Setup and train
    trainer.setup_model_and_tokenizer()
    trainer.prepare_dataset()
    trainer.train()
    
    # Example inference
    example_prompt = "Explain the concept of machine learning in simple terms."
    response = trainer.inference_example(example_prompt)
    logger.info(f"Example response: {response}")


if __name__ == "__main__":
    main() 