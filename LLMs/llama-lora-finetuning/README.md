# LLaMA 3.1 & 4 Scout LoRA Fine-tuning on Vertex AI with GCS Dataset Support

This project provides a complete solution for fine-tuning **LLaMA 3.1** and **LLaMA 4 Scout** models using Low Rank Adaptation (LoRA) on Google Cloud Vertex AI platform with datasets stored in Google Cloud Storage. LoRA is an efficient fine-tuning technique that significantly reduces the number of trainable parameters while maintaining performance.

## New Features

- **LLaMA 4 Scout Support**: Full integration with Meta's latest LLaMA 4 Scout models
- **Local Model Support**: Use locally downloaded LLaMA 4 Scout models
- **Automatic Model Detection**: Smart detection of local LLaMA models
- **LLaMA 3.1 Support**: Updated for LLaMA 3.1 models with improved architecture
- **GCS Dataset Integration**: Load datasets directly from Google Cloud Storage
- **Flash Attention 2**: Optimized attention mechanism for faster training
- **Enhanced Chat Format**: Proper formatting for both LLaMA 3.1 and 4 Scout
- **Improved Data Pipeline**: Better data preprocessing and validation

## Features

- **Multi-Model Support**: Works with LLaMA 3.1, LLaMA 4 Scout, and other compatible models
- **Efficient LoRA Fine-tuning**: Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **4-bit Quantization**: Memory-efficient training with bitsandbytes
- **Vertex AI Integration**: Seamless deployment on Google Cloud
- **GCS Dataset Support**: Load datasets directly from Google Cloud Storage buckets
- **Multiple Dataset Formats**: Support for Alpaca, conversation, Q&A, and completion formats
- **Optimized Training**: Proper tokenization and formatting for different LLaMA versions
- **Monitoring**: Integration with Weights & Biases for experiment tracking
- **Flexible Configuration**: JSON-based configuration management
- **Automatic Data Preparation**: Scripts to convert various data formats and upload to GCS

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Prep     │    │   GCS Dataset   │    │  LoRA Config    │
│   data_prep     │───▶│   Storage       │───▶│   Training      │
│   .py           │    │   (gs://...)    │    │   Script        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
│   LLaMA 4 Scout    │    │   GCS Loader    │    │  Vertex AI      │
│   Download Helper  │    │   Downloads     │    │  Deployment     │
│   download_llama4  │    │   Datasets      │    │  Script         │
│   _scout.py        │    └─────────────────┘    └─────────────────┘
└─────────────────┘                                     │
         │                                              ▼
         ▼                                       ┌─────────────────┐
┌─────────────────┐                             │   Fine-tuned    │
│   Local LLaMA   │                             │   LLaMA 3.1/4   │
│   Models        │─────────────────────────────▶│   Scout (LoRA)  │
│   Auto-Detect   │                             └─────────────────┘
└─────────────────┘
```

## Prerequisites

1. **Google Cloud Setup**:
   - GCP Project with Vertex AI API enabled
   - Service account with appropriate permissions
   - GCS bucket for model storage and datasets
   - Compute Engine API enabled

2. **Local Environment**:
   - Python 3.9+
   - CUDA-capable GPU (for local testing)
   - Access to LLaMA model weights

3. **Model Access**:
   - **LLaMA 3.1**: Hugging Face account with model access
   - **LLaMA 4 Scout**: Custom download URL (provided by Meta)
   - Accepted Meta license agreement

## Installation

1. **Clone or download this project**:
   ```bash
   # Files are in /tmp/llama-lora-finetuning/
   cd /tmp/llama-lora-finetuning
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

   Or install manually:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Google Cloud authentication**:
   ```bash
   gcloud auth application-default login
   # OR set service account key
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
   ```

4. **Setup authentication for your chosen model**:
   
   **For LLaMA 3.1:**
   ```bash
   huggingface-cli login
   ```
   
   **For LLaMA 4 Scout:**
   ```bash
   pip install llama-stack  # If not already installed
   # You'll need your custom download URL when downloading
   ```

5. **Setup Weights & Biases (optional)**:
   ```bash
   wandb login
   ```

## Quick Start with LLaMA 4 Scout

### 1. Download LLaMA 4 Scout Model

First, list available models:
```bash
python download_llama4_scout.py --list
```

Download your desired model (you'll be prompted for your custom URL):
```bash
python download_llama4_scout.py --download llama-4-scout
```

**Note**: When prompted, paste your custom download URL that starts with `https://llama4.llamameta.net/...`

### 2. Prepare and Upload Dataset to GCS

```bash
python data_prep.py \
  --create_sample \
  --upload_to_gcs \
  --gcs_bucket your-dataset-bucket \
  --gcs_prefix datasets/llama4-scout/ \
  --project_id your-gcp-project-id
```

### 3. Configure Training for LLaMA 4 Scout

Edit `config_llama4_scout.json`:
```json
{
  "model_name": "llama-4-scout-local",
  "use_local_model": true,
  "local_model_path": null,
  "project_id": "your-gcp-project-id",
  "staging_bucket": "gs://your-model-bucket",
  "gcs_dataset_bucket": "your-dataset-bucket",
  "gcs_dataset_prefix": "datasets/llama4-scout/"
}
```

### 4. Train with LLaMA 4 Scout

```bash
python llama_lora_finetuning.py \
  --config config_llama4_scout.json \
  --project_id your-gcp-project-id
```

Or specify the model path directly:
```bash
python llama_lora_finetuning.py \
  --local_model_path /path/to/llama4-scout \
  --project_id your-gcp-project-id \
  --gcs_dataset_bucket your-dataset-bucket
```

## Quick Start with LLaMA 3.1

### 1. Prepare and Upload Dataset to GCS

```bash
python data_prep.py \
  --create_sample \
  --upload_to_gcs \
  --gcs_bucket your-dataset-bucket \
  --gcs_prefix datasets/llama3/ \
  --project_id your-gcp-project-id
```

### 2. Configure Training for LLaMA 3.1

Edit `config.json`:
```json
{
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "project_id": "your-gcp-project-id",
  "staging_bucket": "gs://your-model-bucket",
  "gcs_dataset_bucket": "your-dataset-bucket",
  "gcs_dataset_prefix": "datasets/llama3/"
}
```

### 3. Train with LLaMA 3.1

```bash
python llama_lora_finetuning.py \
  --project_id your-gcp-project-id \
  --gcs_dataset_bucket your-dataset-bucket \
  --gcs_dataset_prefix datasets/llama3/ \
  --config config.json
```

## Deploy to Vertex AI

```bash
python vertex_ai_deployment.py \
  --project_id your-gcp-project-id \
  --staging_bucket gs://your-model-bucket \
  --location us-central1 \
  --machine_type n1-standard-8 \
  --accelerator_type NVIDIA_TESLA_V100
```

## Supported Models

### LLaMA 3.1 Models (Hugging Face)

| Model | Size | Context Length | Recommended GPU |
|-------|------|----------------|-----------------|
| `Meta-Llama-3.1-8B` | 8B | 128K | V100, A100 |
| `Meta-Llama-3.1-8B-Instruct` | 8B | 128K | V100, A100 |
| `Meta-Llama-3.1-70B` | 70B | 128K | A100 (multiple) |
| `Meta-Llama-3.1-70B-Instruct` | 70B | 128K | A100 (multiple) |

### LLaMA 4 Scout Models (Local Download)

| Model | Size | Context Length | Status | Recommended GPU |
|-------|------|----------------|--------|-----------------|
| `llama-4-scout` | TBD | TBD | Early Access | A100+ |

**Note**: LLaMA 4 Scout models require a custom download URL and are currently in early access.

## LLaMA 4 Scout Setup Guide

### Step 1: Install LLaMA Stack CLI

```bash
pip install llama-stack
# or upgrade existing installation
pip install llama-stack -U
```

### Step 2: List Available Models

```bash
llama model list
# or for all versions
llama model list --show-all
```

### Step 3: Download LLaMA 4 Scout

Using our helper script:
```bash
# Interactive mode
python download_llama4_scout.py

# Direct download
python download_llama4_scout.py --download MODEL_ID
```

Using llama CLI directly:
```bash
llama model download --source meta --model-id MODEL_ID
```

When prompted, paste your custom download URL.

### Step 4: Verify Download

```bash
python download_llama4_scout.py --info
```

### Step 5: Configure for Training

The training script will auto-detect your downloaded models, or you can specify the path:

```bash
# Auto-detect (recommended)
python llama_lora_finetuning.py --use_local_model --project_id YOUR_PROJECT

# Specify path
python llama_lora_finetuning.py --local_model_path /path/to/model --project_id YOUR_PROJECT
```

## Configuration Options

### Model Selection

| Configuration | Description | Example |
|---------------|-------------|---------|
| **LLaMA 3.1** | Use Hugging Face models | `"model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct"` |
| **LLaMA 4 Scout** | Use local models | `"use_local_model": true, "local_model_path": null` |
| **Custom Path** | Specify exact path | `"local_model_path": "/home/user/.llama/model"` |

### GCS Dataset Configuration

**Option 1: Bucket + Prefix (Recommended)**
```json
{
  "gcs_dataset_bucket": "your-dataset-bucket",
  "gcs_dataset_prefix": "datasets/llama4-scout/"
}
```

**Option 2: Direct File Paths**
```json
{
  "train_file": "gs://bucket/path/train.json",
  "validation_file": "gs://bucket/path/validation.json"
}
```

## LoRA Parameters

### Optimized Settings by Model

| Model Type | lora_r | lora_alpha | Target Modules | Notes |
|------------|--------|------------|----------------|-------|
| **LLaMA 3.1** | 16 | 32 | All attention + MLP | Balanced performance |
| **LLaMA 4 Scout** | 16 | 32 | All attention + MLP | May evolve with updates |
| **Large Models (70B+)** | 32 | 64 | All modules | Higher rank for complexity |

### Target Modules
```json
"lora_target_modules": [
  "q_proj", "k_proj", "v_proj", "o_proj",
  "gate_proj", "up_proj", "down_proj"
]
```

## Training Examples

### Example 1: LLaMA 4 Scout with Auto-Detection
```bash
python llama_lora_finetuning.py \
  --use_local_model \
  --project_id your-project \
  --gcs_dataset_bucket your-datasets \
  --gcs_dataset_prefix experiments/llama4/ \
  --num_epochs 3 \
  --learning_rate 2e-4
```

### Example 2: LLaMA 3.1 with Direct Files
```bash
python llama_lora_finetuning.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_file gs://bucket/train.json \
  --validation_file gs://bucket/val.json \
  --project_id your-project
```

### Example 3: Custom LLaMA 4 Scout Path
```bash
python llama_lora_finetuning.py \
  --local_model_path ~/.llama/llama-4-scout \
  --config config_llama4_scout.json \
  --project_id your-project
```

## Data Format Handling

The system automatically detects and formats data appropriately for each model:

### LLaMA 3.1 Format
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Your instruction here<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Assistant response<|eot_id|>
```

### LLaMA 4 Scout Format
The system uses the same format as LLaMA 3.1 but can be updated as the format evolves.

## Hardware Recommendations

### LLaMA 4 Scout
- **Development**: RTX 4090 (24GB), A100 (40GB)
- **Production**: A100 (80GB) or multiple A100s
- **Memory**: 64GB+ system RAM recommended

### LLaMA 3.1-8B
- **Development**: RTX 3090/4090, V100
- **Production**: A100 (40GB)
- **Memory**: 32GB+ system RAM

### Vertex AI Recommendations
- **Machine Type**: `n1-standard-16` for larger models
- **GPU**: `NVIDIA_TESLA_A100` for best performance
- **Storage**: SSD persistent disks for faster I/O

## Troubleshooting

### LLaMA 4 Scout Issues

1. **Model Download Failed**:
   ```bash
   # Check llama-stack installation
   pip install llama-stack -U
   
   # Verify your custom URL is correct
   # URL should start with: https://llama4.llamameta.net/
   ```

2. **Model Not Detected**:
   ```bash
   # Check for local models
   python download_llama4_scout.py --info
   
   # Specify path manually
   python llama_lora_finetuning.py --local_model_path /exact/path/to/model
   ```

3. **Authentication Issues**:
   ```bash
   # Re-download with correct URL
   python download_llama4_scout.py --download MODEL_ID
   ```

### Common Training Issues

1. **CUDA Out of Memory**:
   - Reduce `per_device_train_batch_size` to 1
   - Increase `gradient_accumulation_steps` to 16+
   - Enable gradient checkpointing
   - Use smaller `max_length`

2. **Model Loading Error**:
   ```bash
   # For LLaMA 3.1
   huggingface-cli login
   
   # For LLaMA 4 Scout
   python download_llama4_scout.py --info
   ```

### Debug Commands

```bash
# Test model detection
python -c "
from llama_lora_finetuning import LLaMA4ScoutModelDetector
detector = LLaMA4ScoutModelDetector()
models = detector.find_local_llama4_models()
print('Found models:', models)
"

# Test GCS access
gsutil ls gs://your-dataset-bucket/

# Validate dataset
python data_prep.py --input gs://bucket/data.json --format alpaca --output_dir ./test
```

## Advanced Configuration

### Multi-Model Training Pipeline
```bash
# Train LLaMA 3.1 baseline
python llama_lora_finetuning.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct

# Train LLaMA 4 Scout with same data
python llama_lora_finetuning.py --use_local_model --output_dir ./results_llama4
```

### Custom Data Preprocessing
```python
# In data_prep.py, customize the format_instruction functions
# for your specific use case and model requirements
```

## Example Configurations

### For Code Generation (LLaMA 4 Scout)
```json
{
  "use_local_model": true,
  "lora_r": 32,
  "lora_alpha": 64,
  "learning_rate": 1e-4,
  "max_length": 1024,
  "wandb_project": "llama4-scout-code-generation"
}
```

### For Conversation (LLaMA 3.1)
```json
{
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "lora_r": 16,
  "lora_alpha": 32,
  "learning_rate": 2e-4,
  "max_length": 512
}
```

## Project Structure

```
llama-lora-finetuning/
├── llama_lora_finetuning.py    # Main training script
├── download_llama4_scout.py    # LLaMA 4 Scout download helper
├── data_prep.py                # Data preparation with GCS upload
├── vertex_ai_deployment.py     # Vertex AI deployment
├── config.json                 # LLaMA 3.1 configuration
├── config_llama4_scout.json    # LLaMA 4 Scout configuration
├── config_gcs_example.json     # Comprehensive GCS examples
├── requirements.txt            # All dependencies
├── setup.sh                    # Automated setup script
└── README.md                   # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/llama4-improvements`
3. Make your changes with tests
4. Update documentation
5. Submit a pull request

## License

This project is licensed under the MIT License. Note that LLaMA models have their own license requirements from Meta.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs)
3. Check [PEFT library documentation](https://huggingface.co/docs/peft)
4. Review [LLaMA model documentation](https://huggingface.co/meta-llama)
5. For LLaMA 4 Scout: Refer to Meta's official documentation

## Changelog

### v3.0 (Current)
- Added LLaMA 4 Scout support
- Local model detection and validation
- Enhanced download helper script
- Multi-model training pipeline
- Improved configuration management

### v2.0
- Added LLaMA 3.1 support
- GCS dataset integration
- Flash Attention 2 optimization
- Enhanced data preprocessing
- Improved error handling

### v1.0
- Initial LLaMA 2 support
- Basic LoRA fine-tuning
- Vertex AI deployment

## Acknowledgments

- **Meta AI** for LLaMA 3.1 and LLaMA 4 Scout models
- **Hugging Face** for transformers and PEFT libraries
- **Google Cloud** for Vertex AI platform
- **Microsoft** for LoRA technique
- **Tri Dao** for Flash Attention 2 