#!/usr/bin/env python3
"""
LLaMA 4 Scout Model Download Helper
==================================

This script helps you download LLaMA 4 Scout models using the llama-stack CLI.
You'll need to provide your own custom download URL when prompted.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def check_llama_stack():
    """Check if llama-stack is installed"""
    try:
        result = subprocess.run(['llama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("llama-stack is installed")
            return True
        else:
            print("llama-stack command not working properly")
            return False
    except FileNotFoundError:
        print("llama-stack is not installed")
        print("Install with: pip install llama-stack")
        return False

def list_available_models(show_all=False):
    """List available LLaMA models"""
    print("\nFetching available LLaMA models...")
    
    cmd = ['llama', 'model', 'list']
    if show_all:
        cmd.append('--show-all')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Available models:")
            print("=" * 50)
            print(result.stdout)
            return True
        else:
            print(f"Error listing models: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def download_model(model_id, custom_url=None):
    """Download a LLaMA model"""
    print(f"\nStarting download for model: {model_id}")
    print("Note: You will be prompted for your custom download URL")
    print("Paste your URL when the llama CLI asks for it.")
    
    cmd = ['llama', 'model', 'download', '--source', 'meta', '--model-id', model_id]
    
    try:
        # Run interactively so user can input the URL when prompted
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"Successfully downloaded {model_id}")
            return True
        else:
            print(f"Download failed for {model_id}")
            return False
    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def get_model_info():
    """Get information about downloaded models"""
    print("\nChecking downloaded models...")
    
    # Try to find where models are stored
    home_dir = Path.home()
    possible_paths = [
        home_dir / ".llama",
        home_dir / ".cache" / "llama",
        Path("./models"),
        Path("./cache")
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found model directory: {path}")
            if list(path.glob("*")):
                print("Contents:")
                for item in path.iterdir():
                    if item.is_dir():
                        print(f"  {item.name}")
                    else:
                        print(f"  {item.name}")
            else:
                print("  (empty)")
        
def main():
    parser = argparse.ArgumentParser(description="LLaMA 4 Scout Model Download Helper")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--list-all", action="store_true", help="List all available models (including older versions)")
    parser.add_argument("--download", type=str, help="Download specific model by ID")
    parser.add_argument("--info", action="store_true", help="Show information about downloaded models")
    
    args = parser.parse_args()
    
    print("LLaMA 4 Scout Model Download Helper")
    print("=" * 40)
    
    # Check if llama-stack is installed
    if not check_llama_stack():
        print("\nTo install llama-stack:")
        print("pip install llama-stack")
        sys.exit(1)
    
    # Handle different operations
    if args.list or args.list_all:
        list_available_models(show_all=args.list_all)
        
    elif args.download:
        print(f"\nPreparing to download: {args.download}")
        print("\nIMPORTANT: You will need your custom download URL")
        print("When prompted by the llama CLI, paste your custom URL that was provided to you.")
        print("The URL should start with: https://llama4.llamameta.net/...")
        
        confirm = input("\nDo you have your custom download URL ready? (y/N): ")
        if confirm.lower() in ['y', 'yes']:
            download_model(args.download)
        else:
            print("Please obtain your custom download URL first, then run this script again.")
            
    elif args.info:
        get_model_info()
        
    else:
        # Interactive mode
        print("\nInteractive Mode")
        print("1. List available models")
        print("2. Download a model")
        print("3. Show downloaded models info")
        print("4. Exit")
        
        while True:
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == "1":
                show_all = input("Show all versions? (y/N): ").lower() in ['y', 'yes']
                list_available_models(show_all=show_all)
                
            elif choice == "2":
                model_id = input("Enter model ID to download: ").strip()
                if model_id:
                    print("\nYou will need your custom download URL")
                    print("Make sure you have it ready before proceeding.")
                    confirm = input("Continue? (y/N): ")
                    if confirm.lower() in ['y', 'yes']:
                        download_model(model_id)
                else:
                    print("Please enter a valid model ID")
                    
            elif choice == "3":
                get_model_info()
                
            elif choice == "4":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main() 