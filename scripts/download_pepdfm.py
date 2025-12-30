#!/usr/bin/env python3
"""
Helper script to download PepDFM model from HuggingFace.

Usage:
    python scripts/download_pepdfm.py
    python scripts/download_pepdfm.py --output pretrained/generators/pepdfm.ckpt
"""

import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Download PepDFM model from HuggingFace")
    parser.add_argument(
        '--output', 
        type=str, 
        default='pretrained/generators/pepdfm.ckpt',
        help='Output path for the checkpoint file'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='ChatterjeeLab/MOG-DFM',
        help='HuggingFace model identifier'
    )
    parser.add_argument(
        '--list-files',
        action='store_true',
        help='List all files in the repository without downloading'
    )
    parser.add_argument(
        '--download-repo',
        action='store_true',
        help='Download entire repository (not just checkpoint)'
    )
    parser.add_argument(
        '--repo-output',
        type=str,
        default='external/mog-dfm',
        help='Output directory for repository download'
    )
    
    args = parser.parse_args()
    
    try:
        from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)
    
    print(f"Repository: {args.model_name}")
    print("=" * 60)
    
    # List files first
    print("Listing files in repository...")
    try:
        files = list_repo_files(args.model_name)
        print(f"\nFound {len(files)} files:")
        for f in files:
            print(f"  - {f}")
        
        # Find checkpoint files
        checkpoint_files = [f for f in files if f.endswith(('.pt', '.pth', '.ckpt'))]
        if checkpoint_files:
            print(f"\n✓ Found {len(checkpoint_files)} checkpoint file(s):")
            for f in checkpoint_files:
                print(f"  - {f}")
        else:
            print("\n⚠ No checkpoint files found (looking for .pt, .pth, .ckpt)")
            print("  The repository might contain the model in a different format.")
        
        if args.list_files:
            return
        
        # Download entire repository if requested
        if args.download_repo:
            print(f"\nDownloading entire repository to: {args.repo_output}")
            repo_path = snapshot_download(
                repo_id=args.model_name,
                local_dir=args.repo_output,
                ignore_patterns=["*.md", "*.txt", "dataset/**"]  # Skip docs and large datasets
            )
            print(f"✓ Repository downloaded to: {repo_path}")
            print(f"\nTo use it, add to your Python path:")
            print(f"  import sys")
            print(f"  sys.path.insert(0, '{Path(repo_path).absolute()}')")
            return
        
        # Download the checkpoint file
        if checkpoint_files:
            # Use the first checkpoint file found
            checkpoint_filename = checkpoint_files[0]
            print(f"\nDownloading: {checkpoint_filename}")
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the file
            downloaded_path = hf_hub_download(
                repo_id=args.model_name,
                filename=checkpoint_filename,
                cache_dir=None  # Use default cache
            )
            
            print(f"✓ Downloaded to cache: {downloaded_path}")
            
            # Copy to output location
            import shutil
            shutil.copy(downloaded_path, output_path)
            print(f"✓ Copied to: {output_path}")
            print(f"\nFile size: {output_path.stat().st_size / 1e6:.2f} MB")
            
        else:
            # Try downloading the entire repository
            print("\nNo checkpoint files found. Downloading entire repository...")
            repo_path = snapshot_download(
                repo_id=args.model_name,
                cache_dir=None,
                ignore_patterns=["*.md", "*.txt"]
            )
            print(f"✓ Repository downloaded to: {repo_path}")
            print("\nPlease check the repository for model files and copy them manually.")
            print(f"Repository location: {repo_path}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

