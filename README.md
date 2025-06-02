# Cloud Computing Code Examples

A collection of practical cloud computing examples, tutorials, and implementations for various platforms and use cases.

## Overview

This repository contains real-world examples and implementations for cloud computing technologies, focusing on machine learning, AI model deployment, and cloud-native solutions.

## Projects

### 1. [Large Language Models (LLMs)](./LLMs/)

Collection of LLM-related projects including fine-tuning, deployment, and optimization techniques.

#### [LLaMA LoRA Fine-tuning on Vertex AI](./LLMs/llama-lora-finetuning/)

A comprehensive solution for fine-tuning LLaMA 3.1 and LLaMA 4 Scout models using Low Rank Adaptation (LoRA) on Google Cloud Vertex AI platform with GCS dataset support.

**Features:**
- Multi-model support (LLaMA 3.1 from Hugging Face, LLaMA 4 Scout locally)
- GCS dataset integration with automatic loading
- Memory-efficient 4-bit quantization with LoRA
- Flash Attention 2 optimization
- Vertex AI deployment automation
- Comprehensive data preparation pipeline

**Technologies:** Google Cloud Vertex AI, Hugging Face Transformers, PEFT, PyTorch, Google Cloud Storage

[View Project →](./LLMs/llama-lora-finetuning/)
## Repository Structure

```
cloud-computing/
├── README.md                    # This file
├── LICENSE                      # MIT License
└── LLMs/                       # Large Language Model projects
    └── llama-lora-finetuning/  # LLaMA fine-tuning with LoRA on Vertex AI
        ├── README.md           # Project-specific documentation
        ├── llama_lora_finetuning.py # Main training script
        ├── data_prep.py        # Data preparation with GCS support
        ├── download_llama4_scout.py # LLaMA 4 Scout download helper
        ├── vertex_ai_deployment.py # Vertex AI deployment script
        ├── requirements.txt    # Dependencies
        ├── setup.sh           # Automated setup script
        ├── config.json        # LLaMA 3.1 configuration
        ├── config_llama4_scout.json # LLaMA 4 Scout configuration
        └── config_gcs_example.json # Comprehensive GCS examples
```

## Getting Started

Each project contains its own README with detailed setup and usage instructions. Navigate to the project directory and follow the specific documentation.

### Prerequisites

- Python 3.9+
- Google Cloud SDK (for GCP projects)
- Git

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nl-alonso/cloud-computing.git
   cd cloud-computing
   ```

2. **Choose a project and follow its README:**
   ```bash
   cd LLMs/llama-lora-finetuning
   ./setup.sh
   ```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- New cloud computing examples
- Improvements to existing projects
- Bug fixes and optimizations
- Documentation enhancements

### Adding New Projects

When adding new projects to this repository:

1. Create a new directory with a descriptive name
2. Include a comprehensive README.md
3. Add setup instructions and requirements
4. Update this main README to include your project
5. Follow the established code style and documentation patterns

## Technologies Covered

Current and planned technologies include:

- **Google Cloud Platform**
  - Vertex AI
  - Cloud Storage
  - Compute Engine
  - Cloud Functions

- **Machine Learning & AI**
  - Large Language Models (LLaMA, GPT, Claude)
  - Fine-tuning techniques (LoRA, QLoRA, Full Fine-tuning)
  - Model deployment and serving
  - Data preprocessing pipelines
  - Model optimization and quantization

- **Cloud-Native Technologies**
  - Docker containers
  - Kubernetes deployments
  - Serverless computing
  - CI/CD pipelines

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Nick Alonso - [@nl-alonso](https://github.com/nl-alonso)

Project Link: [https://github.com/nl-alonso/cloud-computing](https://github.com/nl-alonso/cloud-computing)

## Acknowledgments

- Google Cloud for Vertex AI platform
- Meta AI for LLaMA models
- Hugging Face for transformers and PEFT libraries
- The open-source community for various tools and libraries used in these projects 