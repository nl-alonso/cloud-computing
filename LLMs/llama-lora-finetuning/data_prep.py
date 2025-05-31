#!/usr/bin/env python3
"""
Data Preparation Script for LLaMA 3.1 LoRA Fine-tuning with GCS Support
=======================================================================

This script helps prepare custom datasets for LoRA fine-tuning.
Supports various input formats, converts them to the required format,
and can upload datasets to Google Cloud Storage.
"""

import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import os


def prepare_alpaca_format(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert data to Alpaca format with instruction, input, and output fields.
    
    Expected format:
    {
        "instruction": "Task description",
        "input": "Additional context (optional)",
        "output": "Expected response"
    }
    """
    formatted_data = []
    
    for item in data:
        formatted_item = {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", "")
        }
        formatted_data.append(formatted_item)
    
    return formatted_data


def prepare_conversation_format(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert conversation data to instruction format.
    
    Expected input format:
    {
        "messages": [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"}
        ]
    }
    """
    formatted_data = []
    
    for item in data:
        messages = item.get("messages", [])
        if len(messages) >= 2:
            user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            assistant_msg = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
            
            if user_msg and assistant_msg:
                formatted_item = {
                    "instruction": user_msg,
                    "input": "",
                    "output": assistant_msg
                }
                formatted_data.append(formatted_item)
    
    return formatted_data


def prepare_qa_format(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert Q&A data to instruction format.
    
    Expected input format:
    {
        "question": "Question text",
        "answer": "Answer text"
    }
    """
    formatted_data = []
    
    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        if question and answer:
            formatted_item = {
                "instruction": question,
                "input": "",
                "output": answer
            }
            formatted_data.append(formatted_item)
    
    return formatted_data


def prepare_text_completion_format(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert text completion data to instruction format.
    
    Expected input format:
    {
        "prompt": "Prompt text",
        "completion": "Completion text"
    }
    """
    formatted_data = []
    
    for item in data:
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")
        
        if prompt and completion:
            formatted_item = {
                "instruction": "Complete the following text:",
                "input": prompt,
                "output": completion
            }
            formatted_data.append(formatted_item)
    
    return formatted_data


def create_sample_dataset() -> List[Dict[str, str]]:
    """Create a sample dataset for demonstration purposes optimized for LLaMA 3.1."""
    sample_data = [
        {
            "instruction": "Explain the concept of machine learning in simple terms.",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It works by finding patterns in data and using those patterns to make predictions or decisions about new, unseen data. Think of it like teaching a computer to recognize patterns the same way humans do, but much faster and with larger amounts of information."
        },
        {
            "instruction": "Write a Python function to calculate the factorial of a number.",
            "input": "",
            "output": "Here's a Python function to calculate factorial:\n\n```python\ndef factorial(n):\n    \"\"\"Calculate factorial of n recursively.\"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\n# Alternative iterative approach:\ndef factorial_iterative(n):\n    \"\"\"Calculate factorial of n iteratively.\"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n```\n\nBoth functions calculate the factorial by either using recursion or iteration."
        },
        {
            "instruction": "Translate the following English sentence to French.",
            "input": "The weather is beautiful today.",
            "output": "Le temps est magnifique aujourd'hui."
        },
        {
            "instruction": "Summarize the key benefits of renewable energy.",
            "input": "",
            "output": "Key benefits of renewable energy include:\n\n1. **Environmental sustainability** - Significantly reduces greenhouse gas emissions and air pollution\n2. **Energy independence** - Reduces reliance on fossil fuel imports and volatile energy markets\n3. **Economic advantages** - Creates jobs, provides stable long-term energy costs, and stimulates local economies\n4. **Inexhaustible supply** - Sources like solar, wind, and hydroelectric are naturally replenished\n5. **Improved public health** - Cleaner air and water quality lead to reduced healthcare costs\n6. **Technological innovation** - Drives advancement in energy storage, grid management, and efficiency technologies"
        },
        {
            "instruction": "Explain the difference between supervised and unsupervised learning.",
            "input": "",
            "output": "**Supervised Learning:**\n- Uses labeled training data where input-output pairs are provided\n- Algorithm learns to map inputs to correct outputs\n- Goal is to make accurate predictions on new, unseen data\n- Examples: classification (spam detection), regression (price prediction)\n- Performance can be measured against known correct answers\n\n**Unsupervised Learning:**\n- Works with unlabeled data to discover hidden patterns or structures\n- No predetermined target variable or correct answers provided\n- Goal is to find meaningful patterns, groups, or representations in data\n- Examples: clustering (customer segmentation), dimensionality reduction (data visualization)\n- Success is measured by interpretability and usefulness of discovered patterns\n\nThe key difference is that supervised learning predicts specific outcomes based on examples, while unsupervised learning explores data to find underlying structures without guidance."
        },
        {
            "instruction": "What are the main components of a neural network?",
            "input": "",
            "output": "The main components of a neural network include:\n\n1. **Neurons (Nodes)** - Basic processing units that receive inputs, apply weights, and produce outputs\n2. **Layers** - Collections of neurons organized in:\n   - Input layer: receives raw data\n   - Hidden layers: process information\n   - Output layer: produces final results\n3. **Weights** - Numerical parameters that determine the strength of connections between neurons\n4. **Biases** - Additional parameters that help the model fit data better\n5. **Activation Functions** - Mathematical functions (like ReLU, sigmoid) that introduce non-linearity\n6. **Loss Function** - Measures how far the network's predictions are from actual values\n7. **Optimizer** - Algorithm (like Adam, SGD) that adjusts weights to minimize loss\n\nThese components work together to learn patterns from data through backpropagation and gradient descent."
        },
        {
            "instruction": "How do you optimize code for better performance?",
            "input": "",
            "output": "Here are key strategies for code optimization:\n\n**Algorithmic Optimization:**\n- Choose efficient algorithms and data structures\n- Reduce time complexity (O(n²) → O(n log n))\n- Use appropriate data structures (hash tables, trees)\n\n**Memory Management:**\n- Minimize memory allocations\n- Reuse objects when possible\n- Use memory pools for frequent allocations\n\n**Language-Specific Techniques:**\n- Use built-in functions and libraries\n- Avoid premature optimization\n- Profile code to identify bottlenecks\n\n**Caching and Memoization:**\n- Cache expensive computations\n- Use database query optimization\n- Implement proper indexing\n\n**Parallel Processing:**\n- Use multithreading for I/O-bound tasks\n- Implement multiprocessing for CPU-bound tasks\n- Consider vectorization for numerical computations\n\n**General Principles:**\n- Measure before optimizing\n- Focus on the most impactful changes\n- Balance readability with performance\n- Use profiling tools to identify real bottlenecks"
        },
        {
            "instruction": "Explain quantum computing concepts for beginners.",
            "input": "",
            "output": "Quantum computing is a revolutionary computing paradigm that leverages quantum mechanics principles:\n\n**Key Concepts:**\n\n1. **Qubits** - Unlike classical bits (0 or 1), qubits can exist in 'superposition' - simultaneously 0 and 1 until measured\n\n2. **Superposition** - Allows quantum computers to process multiple possibilities simultaneously, providing exponential computational advantages\n\n3. **Entanglement** - Qubits can be 'entangled,' meaning the state of one immediately affects another, regardless of distance\n\n4. **Quantum Interference** - Quantum algorithms manipulate probability amplitudes to increase correct answer probabilities\n\n**Advantages:**\n- Exponentially faster for specific problems (cryptography, optimization, simulation)\n- Can solve certain problems classical computers cannot\n\n**Current Limitations:**\n- Extremely sensitive to environmental interference\n- Requires near absolute zero temperatures\n- High error rates and limited qubit counts\n- Only advantageous for specific problem types\n\n**Applications:**\n- Drug discovery and molecular simulation\n- Cryptography and security\n- Financial modeling and optimization\n- Artificial intelligence and machine learning\n\nQuantum computing won't replace classical computers but will solve specific complex problems much faster."
        }
    ]
    return sample_data


def split_dataset(data: List[Dict[str, str]], train_ratio: float = 0.8) -> tuple:
    """Split dataset into training and validation sets."""
    import random
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split index
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data


def save_dataset(data: List[Dict[str, str]], output_path: str):
    """Save dataset to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_path} with {len(data)} examples")


def upload_to_gcs(local_file_path: str, gcs_bucket: str, gcs_path: str, project_id: Optional[str] = None):
    """Upload a file to Google Cloud Storage."""
    try:
        from google.cloud import storage
        
        client = storage.Client(project=project_id) if project_id else storage.Client()
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)
        
        print(f"Uploading {local_file_path} to gs://{gcs_bucket}/{gcs_path}")
        blob.upload_from_filename(local_file_path)
        
        print(f"File uploaded successfully to gs://{gcs_bucket}/{gcs_path}")
        return f"gs://{gcs_bucket}/{gcs_path}"
        
    except ImportError:
        print("Google Cloud Storage library not installed. Install with: pip install google-cloud-storage")
        return None
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None


def load_and_convert_dataset(input_path: str, format_type: str) -> List[Dict[str, str]]:
    """Load and convert dataset based on the specified format."""
    # Load data
    if input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
        data = df.to_dict('records')
    else:
        raise ValueError("Unsupported file format. Please use JSON or CSV.")
    
    # Convert based on format type
    if format_type == "alpaca":
        return prepare_alpaca_format(data)
    elif format_type == "conversation":
        return prepare_conversation_format(data)
    elif format_type == "qa":
        return prepare_qa_format(data)
    elif format_type == "completion":
        return prepare_text_completion_format(data)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LLaMA 3.1 LoRA fine-tuning with GCS support")
    parser.add_argument("--input", type=str, help="Input data file (JSON or CSV)")
    parser.add_argument("--format", type=str, choices=["alpaca", "conversation", "qa", "completion"], 
                       default="alpaca", help="Input data format")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset")
    
    # GCS upload options
    parser.add_argument("--upload_to_gcs", action="store_true", help="Upload datasets to Google Cloud Storage")
    parser.add_argument("--gcs_bucket", type=str, help="GCS bucket name for upload")
    parser.add_argument("--gcs_prefix", type=str, default="datasets/", help="GCS prefix for dataset files")
    parser.add_argument("--project_id", type=str, help="Google Cloud project ID")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.create_sample:
        print("Creating sample dataset for LLaMA 3.1...")
        data = create_sample_dataset()
    elif args.input:
        print(f"Loading and converting dataset from {args.input}...")
        data = load_and_convert_dataset(args.input, args.format)
    else:
        raise ValueError("Please provide either --input file or use --create_sample")
    
    # Split dataset
    train_data, val_data = split_dataset(data, args.train_ratio)
    
    # Save datasets locally
    train_path = output_dir / "train.json"
    val_path = output_dir / "validation.json"
    
    save_dataset(train_data, str(train_path))
    save_dataset(val_data, str(val_path))
    
    print(f"\nDataset preparation complete!")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Files saved in: {output_dir}")
    
    # Upload to GCS if requested
    if args.upload_to_gcs:
        if not args.gcs_bucket:
            print("GCS bucket name is required for upload. Use --gcs_bucket")
        else:
            print(f"\nUploading datasets to GCS bucket: {args.gcs_bucket}")
            
            # Upload training data
            train_gcs_path = f"{args.gcs_prefix}train.json"
            train_gcs_url = upload_to_gcs(str(train_path), args.gcs_bucket, train_gcs_path, args.project_id)
            
            # Upload validation data
            val_gcs_path = f"{args.gcs_prefix}validation.json"
            val_gcs_url = upload_to_gcs(str(val_path), args.gcs_bucket, val_gcs_path, args.project_id)
            
            if train_gcs_url and val_gcs_url:
                print(f"\nDatasets uploaded successfully!")
                print(f"Training data: {train_gcs_url}")
                print(f"Validation data: {val_gcs_url}")
                print(f"\nYou can now use these GCS paths in your training config:")
                print(f"   train_file: \"{train_gcs_url}\"")
                print(f"   validation_file: \"{val_gcs_url}\"")
                print(f"   OR")
                print(f"   gcs_dataset_bucket: \"{args.gcs_bucket}\"")
                print(f"   gcs_dataset_prefix: \"{args.gcs_prefix}\"")
    
    # Show sample
    if train_data:
        print(f"\nSample training example:")
        print(json.dumps(train_data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main() 