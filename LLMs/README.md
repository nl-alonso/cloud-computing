# Large Language Models (LLMs)

This directory contains projects related to Large Language Models, including fine-tuning, deployment, optimization, and various implementation techniques on cloud platforms.

## Projects

### [LLaMA LoRA Fine-tuning on Vertex AI](./llama-lora-finetuning/)

A comprehensive solution for fine-tuning LLaMA 3.1 and LLaMA 4 Scout models using Low Rank Adaptation (LoRA) on Google Cloud Vertex AI platform.

**Key Features:**
- Multi-model support (LLaMA 3.1 from Hugging Face, LLaMA 4 Scout locally)
- GCS dataset integration with automatic loading and preprocessing
- Memory-efficient 4-bit quantization with LoRA fine-tuning
- Flash Attention 2 optimization for faster training
- Vertex AI deployment automation and scaling
- Comprehensive data preparation pipeline with multiple format support

**Technologies:** Google Cloud Vertex AI, Hugging Face Transformers, PEFT, PyTorch, Google Cloud Storage, Weights & Biases

**Use Cases:**
- Custom domain fine-tuning (code generation, conversation, Q&A)
- Research and experimentation with latest LLaMA models
- Production deployment on Google Cloud infrastructure
- Educational purposes for understanding LoRA fine-tuning

### [Claude-Salesforce MCP Voice Assistant](./claude-salesforce-mcp/)

A powerful voice-powered web application that integrates Claude AI with Salesforce CRM using natural language processing to query Salesforce data through voice commands.

**Key Features:**
- Voice-to-SOQL conversion using Claude AI for natural language understanding
- Real-time Salesforce OAuth 2.0 authentication and session management
- Multi-object support (Accounts, Opportunities, Contacts, Leads)
- Browser-based voice recognition with modern web interface
- Automated test data generation and CSV import templates
- Enterprise-grade security with environment-based credential management

**Technologies:** Claude 3.5 Sonnet (Anthropic), Salesforce REST API, Flask, Web Speech API, OAuth 2.0

**Use Cases:**
- Sales team productivity enhancement through voice-driven CRM queries
- Executive dashboards with natural language data access
- CRM training and demonstration with realistic sample data
- Integration prototype for voice-enabled business applications

[→ Go to Project](./claude-salesforce-mcp/)
[→ Go to Project](./llama-lora-finetuning/)

## Coming Soon

Future projects planned for this directory:

- **GPT Fine-tuning Examples** - OpenAI API fine-tuning workflows
- **Model Serving Solutions** - Scalable inference deployments
- **LLM Evaluation Frameworks** - Automated testing and benchmarking
- **Multi-Cloud Deployments** - AWS, Azure, and GCP comparisons
- **LLM Security & Safety** - Content filtering and safety implementations

## Quick Start

Each project contains its own detailed README with setup instructions. General prerequisites:

1. **Python 3.9+**
2. **Cloud Platform Account** (GCP for current projects)
3. **Model Access** (Hugging Face account for gated models)

Navigate to any project directory and run:
```bash
./setup.sh  # Automated setup
```

## Contributing

When adding new LLM projects to this directory:

1. Create a descriptive project folder name
2. Include a comprehensive README with:
   - Clear project description
   - Setup and installation instructions
   - Usage examples
   - Hardware requirements
   - Troubleshooting guide
3. Add your project to this README
4. Follow established coding standards
5. Include configuration examples

## Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning) Library](https://huggingface.co/docs/peft)
- [Google Cloud Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [LoRA: Low-Rank Adaptation Paper](https://arxiv.org/abs/2106.09685) 