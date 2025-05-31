#!/bin/bash

# LLaMA 3.1/4 Scout LoRA Fine-tuning Setup Script with GCS Dataset Support
# ========================================================================

set -e  # Exit on any error

echo "LLaMA 3.1/4 Scout LoRA Fine-tuning Setup with GCS Dataset Support"
echo "===================================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Python $required_version or higher is required for LLaMA 3.1/4 Scout. Found: $python_version"
    exit 1
fi

echo "Python $python_version detected"

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies for LLaMA 3.1/4 Scout..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Dependencies installed"
else
    echo "requirements.txt not found"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p cache
mkdir -p results
echo "Directories created"

# Check for LLaMA Stack CLI
echo "Checking LLaMA Stack CLI..."
if command -v llama &> /dev/null; then
    echo "LLaMA Stack CLI detected"
    llama_version=$(llama --version 2>/dev/null || echo "unknown")
    echo "LLaMA CLI Version: $llama_version"
else
    echo "LLaMA Stack CLI not found"
    echo "This is needed for LLaMA 4 Scout models"
    echo "Install with: pip install llama-stack"
fi

# Check for Google Cloud CLI
if command -v gcloud &> /dev/null; then
    echo "Google Cloud CLI detected"
    
    # Check if authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "."; then
        echo "Google Cloud authentication found"
        
        # Get current project
        current_project=$(gcloud config get-value project 2>/dev/null || echo "not-set")
        if [ "$current_project" != "not-set" ]; then
            echo "Current GCP project: $current_project"
        else
            echo "No default GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
        fi
    else
        echo "Google Cloud authentication not found"
        echo "Run: gcloud auth application-default login"
    fi
else
    echo "Google Cloud CLI not found"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
fi

# Check for Hugging Face CLI
if command -v huggingface-cli &> /dev/null; then
    echo "Hugging Face CLI detected"
    
    # Check if authenticated
    if [ -f ~/.cache/huggingface/token ] || [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
        echo "Hugging Face authentication found"
    else
        echo "Hugging Face authentication not found"
        echo "Run: huggingface-cli login"
        echo "Note: LLaMA 3.1 models require authentication and license acceptance"
    fi
else
    echo "Hugging Face CLI not found (installed with transformers)"
    echo "You can authenticate later with: huggingface-cli login"
fi

# Check for CUDA (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    
    # Check CUDA version
    cuda_version=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | cut -d' ' -f3)
    if [ -n "$cuda_version" ]; then
        echo "CUDA Version: $cuda_version"
    fi
else
    echo "No NVIDIA GPU detected (OK for Vertex AI deployment)"
fi

# Create sample configuration files
if [ ! -f "config_local.json" ]; then
    echo "Creating sample configuration with GCS dataset support..."
    cat > config_local.json << EOF
{
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "project_id": "your-gcp-project-id",
  "staging_bucket": "gs://your-model-storage-bucket",
  "location": "us-central1",
  
  "gcs_dataset_bucket": "your-dataset-bucket",
  "gcs_dataset_prefix": "datasets/llama3/",
  
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 2e-4,
  "max_length": 512,
  
  "use_4bit": true,
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_quant_type": "nf4",
  
  "use_wandb": false,
  "wandb_project": "llama3-lora-finetuning"
}
EOF
    echo "Sample LLaMA 3.1 configuration created: config_local.json"
else
    echo "Configuration file already exists"
fi

# Create LLaMA 4 Scout configuration template
if [ ! -f "config_llama4_local.json" ]; then
    echo "Creating LLaMA 4 Scout configuration template..."
    cat > config_llama4_local.json << EOF
{
  "model_name": "llama-4-scout-local",
  "use_local_model": true,
  "local_model_path": null,
  "project_id": "your-gcp-project-id",
  "staging_bucket": "gs://your-model-storage-bucket",
  
  "gcs_dataset_bucket": "your-dataset-bucket",
  "gcs_dataset_prefix": "datasets/llama4-scout/",
  
  "lora_r": 16,
  "lora_alpha": 32,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 2e-4,
  
  "use_4bit": true,
  "use_wandb": true,
  "wandb_project": "llama4-scout-lora-finetuning"
}
EOF
    echo "LLaMA 4 Scout configuration template created: config_llama4_local.json"
fi

# Generate sample data for LLaMA models
echo "Generating sample data optimized for LLaMA models..."
python data_prep.py --create_sample --output_dir ./data
echo "Sample data created in ./data/"

# Test Flash Attention 2 availability
echo "Testing Flash Attention 2 availability..."
python -c "
try:
    import flash_attn
    print('Flash Attention 2 is available')
except ImportError:
    print('Flash Attention 2 not available - will use standard attention')
    print('Install with: pip install flash-attn --no-build-isolation')
" 2>/dev/null || echo "Could not test Flash Attention 2"

# Check for required packages
echo "Checking key dependencies..."
python -c "
import sys
required_packages = {
    'transformers': '4.40.0',
    'peft': '0.10.0', 
    'torch': '2.1.0',
    'datasets': '2.18.0',
    'bitsandbytes': '0.43.0'
}

optional_packages = {
    'llama_stack': 'unknown'  # For LLaMA 4 Scout
}

for package, min_version in required_packages.items():
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f'{package}: {version}')
    except ImportError:
        print(f'{package}: not installed')

for package, min_version in optional_packages.items():
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f'{package}: {version} (for LLaMA 4 Scout)')
    except ImportError:
        print(f'{package}: not installed (needed for LLaMA 4 Scout)')
"

# Check for local LLaMA models
echo "Checking for local LLaMA models..."
python -c "
import sys
from pathlib import Path

possible_paths = [
    Path.home() / '.llama',
    Path.home() / '.cache' / 'llama',
    Path('./models'),
    Path('./cache')
]

found_any = False
for path in possible_paths:
    if path.exists() and any(path.iterdir()):
        print(f'Found models directory: {path}')
        for item in path.iterdir():
            if item.is_dir():
                print(f'   {item.name}')
        found_any = True

if not found_any:
    print('No local LLaMA models found')
    print('  For LLaMA 4 Scout: run python download_llama4_scout.py')
    print('  For LLaMA 3.1: models will be downloaded from Hugging Face')
"

# Print next steps
echo ""
echo "Setup Complete!"
echo "=================="
echo ""
echo "Next steps:"
echo ""
echo "1. Authentication Setup:"
echo "   For LLaMA 3.1: huggingface-cli login"
echo "   For Google Cloud: gcloud auth application-default login"
echo ""
echo "2. Configure your training:"
echo "   • Edit config_local.json for LLaMA 3.1 models"
echo "   • Edit config_llama4_local.json for LLaMA 4 Scout"
echo "   • Set your GCP project_id and bucket names"
echo ""
echo "3a. For LLaMA 4 Scout (requires custom download URL):"
echo "   • List available models: python download_llama4_scout.py --list"
echo "   • Download model: python download_llama4_scout.py --download MODEL_ID"
echo "   • Train: python llama_lora_finetuning.py --config config_llama4_local.json --project_id YOUR_PROJECT"
echo ""
echo "3b. For LLaMA 3.1 (from Hugging Face):"
echo "   • Train: python llama_lora_finetuning.py --config config_local.json --project_id YOUR_PROJECT"
echo ""
echo "4. Upload sample data to GCS (optional):"
echo "   python data_prep.py --create_sample --upload_to_gcs \\"
echo "     --gcs_bucket YOUR_DATASET_BUCKET --gcs_prefix datasets/llama/ \\"
echo "     --project_id YOUR_PROJECT_ID"
echo ""
echo "5. Deploy to Vertex AI:"
echo "   python vertex_ai_deployment.py --project_id YOUR_PROJECT \\"
echo "     --staging_bucket gs://YOUR_BUCKET --location us-central1"
echo ""
echo "See README.md for detailed instructions and examples"
echo ""
echo "Useful links:"
echo "   - LLaMA 3.1 models: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "   - LLaMA 4 Scout: Requires custom download URL"
echo "   - Vertex AI docs: https://cloud.google.com/vertex-ai/docs"
echo "   - GCS docs: https://cloud.google.com/storage/docs"
echo ""

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment is active. To deactivate: deactivate"
else
    echo "To activate virtual environment: source venv/bin/activate"
fi

echo ""
echo "Ready to fine-tune LLaMA 3.1 and LLaMA 4 Scout with LoRA on Vertex AI!" 