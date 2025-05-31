#!/usr/bin/env python3
"""
Vertex AI Deployment Script for LLaMA LoRA Fine-tuning
======================================================

This script submits the LoRA fine-tuning job to Google Cloud Vertex AI.
"""

import argparse
import json
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip
from datetime import datetime


def create_custom_job(
    project_id: str,
    location: str,
    staging_bucket: str,
    job_display_name: str,
    python_package_gcs_uri: str,
    python_module: str,
    args: list,
    machine_type: str = "n1-standard-8",
    accelerator_type: str = "NVIDIA_TESLA_V100",
    accelerator_count: int = 1,
    replica_count: int = 1
):
    """
    Create and submit a custom training job to Vertex AI
    
    Args:
        project_id: GCP project ID
        location: GCP location (e.g., 'us-central1')
        staging_bucket: GCS bucket for staging
        job_display_name: Display name for the job
        python_package_gcs_uri: GCS URI of the training package
        python_module: Python module to execute
        args: Arguments to pass to the training script
        machine_type: Machine type for training
        accelerator_type: GPU type
        accelerator_count: Number of GPUs
        replica_count: Number of replicas
    """
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)
    
    # Define the worker pool spec
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": replica_count,
            "python_package_spec": {
                "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest",
                "package_uris": [python_package_gcs_uri],
                "python_module": python_module,
                "args": args,
            },
        }
    ]
    
    # Create the custom job
    job = aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=staging_bucket,
    )
    
    # Submit the job
    print(f"Submitting job: {job_display_name}")
    job.run(sync=False)  # Set to True if you want to wait for completion
    
    print(f"Job submitted successfully!")
    print(f"Job resource name: {job.resource_name}")
    print(f"View job in console: https://console.cloud.google.com/ai/platform/locations/{location}/training/{job.name}")
    
    return job


def package_training_code(staging_bucket: str, source_dir: str = "."):
    """
    Package the training code and upload to GCS
    
    Args:
        staging_bucket: GCS bucket for staging
        source_dir: Local directory containing the training code
    
    Returns:
        GCS URI of the uploaded package
    """
    import os
    import tarfile
    import tempfile
    from google.cloud import storage
    
    # Create a temporary tar file
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        tar_path = tmp_file.name
    
    # Create tar archive
    with tarfile.open(tar_path, 'w:gz') as tar:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.py', '.json', '.txt')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    tar.add(file_path, arcname=arcname)
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(staging_bucket.replace('gs://', ''))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    blob_name = f"training_packages/llama_lora_{timestamp}.tar.gz"
    blob = bucket.blob(blob_name)
    
    print(f"Uploading training package to gs://{staging_bucket}/{blob_name}")
    blob.upload_from_filename(tar_path)
    
    # Clean up
    os.unlink(tar_path)
    
    return f"gs://{staging_bucket}/{blob_name}"


def main():
    parser = argparse.ArgumentParser(description="Deploy LLaMA LoRA training to Vertex AI")
    parser.add_argument("--project_id", type=str, required=True, help="GCP project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP location")
    parser.add_argument("--staging_bucket", type=str, required=True, help="GCS staging bucket")
    parser.add_argument("--job_name", type=str, help="Custom job name")
    parser.add_argument("--config", type=str, default="config.json", help="Training config file")
    parser.add_argument("--machine_type", type=str, default="n1-standard-8", help="Machine type")
    parser.add_argument("--accelerator_type", type=str, default="NVIDIA_TESLA_V100", help="GPU type")
    parser.add_argument("--accelerator_count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--replica_count", type=int, default=1, help="Number of replicas")
    
    args = parser.parse_args()
    
    # Generate job name if not provided
    if not args.job_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.job_name = f"llama-lora-training-{timestamp}"
    
    # Package and upload training code
    print("Packaging training code...")
    package_uri = package_training_code(args.staging_bucket)
    
    # Prepare training arguments
    training_args = [
        "--config", "config.json",
        "--project_id", args.project_id,
        "--staging_bucket", args.staging_bucket
    ]
    
    # Create and submit the job
    job = create_custom_job(
        project_id=args.project_id,
        location=args.location,
        staging_bucket=args.staging_bucket,
        job_display_name=args.job_name,
        python_package_gcs_uri=package_uri,
        python_module="llama_lora_finetuning",
        args=training_args,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        replica_count=args.replica_count
    )
    
    return job


if __name__ == "__main__":
    main() 