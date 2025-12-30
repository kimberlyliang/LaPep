#!/usr/bin/env python3
"""
Check where the PepDFM model is loaded from.

This script shows:
1. Where local checkpoint files are located
2. Where HuggingFace cache is located
3. Where the model would be loaded from
"""

import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 70)
    print("PepDFM Model Location Checker")
    print("=" * 70)
    
    # Check local checkpoint locations
    print("\n1. LOCAL CHECKPOINT LOCATIONS:")
    print("-" * 70)
    
    possible_paths = [
        "pepdfm.ckpt",
        "pretrained/generators/pepdfm.ckpt",
        Path(__file__).parent.parent / "pepdfm.ckpt",
        Path(__file__).parent.parent / "pretrained" / "generators" / "pepdfm.ckpt",
    ]
    
    found_local = False
    for path in possible_paths:
        path_obj = Path(path)
        if path_obj.exists():
            size_mb = path_obj.stat().st_size / 1e6
            print(f"  ✓ Found: {path_obj.absolute()}")
            print(f"    Size: {size_mb:.2f} MB")
            found_local = True
        else:
            print(f"  ✗ Not found: {path_obj}")
    
    if not found_local:
        print("  ⚠ No local checkpoint found")
    
    # Check HuggingFace cache
    print("\n2. HUGGINGFACE CACHE LOCATIONS:")
    print("-" * 70)
    
    # Check environment variables
    cache_dirs = []
    if os.environ.get('HF_HOME'):
        cache_dirs.append(('HF_HOME', os.environ.get('HF_HOME')))
    if os.environ.get('TRANSFORMERS_CACHE'):
        cache_dirs.append(('TRANSFORMERS_CACHE', os.environ.get('TRANSFORMERS_CACHE')))
    if os.environ.get('HF_HUB_CACHE'):
        cache_dirs.append(('HF_HUB_CACHE', os.environ.get('HF_HUB_CACHE')))
    
    if cache_dirs:
        print("  Environment variables:")
        for var_name, var_path in cache_dirs:
            path_obj = Path(var_path)
            if path_obj.exists():
                print(f"    ✓ {var_name}: {path_obj.absolute()}")
            else:
                print(f"    ✗ {var_name}: {var_path} (does not exist)")
    
    # Check default locations
    default_cache = Path.home() / ".cache" / "huggingface"
    if default_cache.exists():
        print(f"\n  Default cache: {default_cache.absolute()}")
        
        # Check for MOG-DFM in cache
        hub_dir = default_cache / "hub"
        if hub_dir.exists():
            mog_dfm_dirs = list(hub_dir.glob("models--ChatterjeeLab--MOG-DFM*"))
            if mog_dfm_dirs:
                print(f"    ✓ Found MOG-DFM cache:")
                for mog_dir in mog_dfm_dirs:
                    print(f"      - {mog_dir}")
                    # Look for snapshot directories
                    snapshots = list(mog_dir.glob("snapshots/*"))
                    if snapshots:
                        for snap in snapshots:
                            print(f"        Snapshot: {snap}")
                            # Check for checkpoint files
                            ckpt_files = list(snap.glob("**/*.ckpt")) + \
                                        list(snap.glob("**/*.pt")) + \
                                        list(snap.glob("**/*.pth"))
                            if ckpt_files:
                                for ckpt in ckpt_files:
                                    size_mb = ckpt.stat().st_size / 1e6
                                    print(f"          ✓ {ckpt.name} ({size_mb:.2f} MB)")
            else:
                print(f"    ✗ MOG-DFM not found in cache")
    
    # Check scratch directories (if on cluster)
    print("\n3. SCRATCH DIRECTORY CHECK:")
    print("-" * 70)
    if os.path.exists('/scratch'):
        import getpass
        username = getpass.getuser()
        scratch_dirs = [
            f'/scratch/pranamlab/{username}/.cache/huggingface',
            f'/scratch/{username}/.cache/huggingface',
        ]
        for scratch_dir in scratch_dirs:
            path_obj = Path(scratch_dir)
            if path_obj.exists():
                print(f"  ✓ Found: {path_obj}")
                # Check for MOG-DFM
                hub_dir = path_obj / "hub"
                if hub_dir.exists():
                    mog_dfm_dirs = list(hub_dir.glob("models--ChatterjeeLab--MOG-DFM*"))
                    if mog_dfm_dirs:
                        print(f"    ✓ MOG-DFM cache found here")
            else:
                print(f"  ✗ Not found: {scratch_dir}")
    else:
        print("  (Not on a system with /scratch)")
    
    # Show what load_dfm_model would use
    print("\n4. WHERE MODEL WOULD BE LOADED FROM:")
    print("-" * 70)
    
    try:
        from generators.dfm_wrapper import load_dfm_model
        
        # Check if local file exists
        local_paths = [
            Path("pretrained/generators/pepdfm.ckpt"),
            Path(__file__).parent.parent / "pretrained" / "generators" / "pepdfm.ckpt",
        ]
        
        local_exists = False
        for local_path in local_paths:
            if local_path.exists():
                print(f"  ✓ Would load from LOCAL: {local_path.absolute()}")
                local_exists = True
                break
        
        if not local_exists:
            print("  ⚠ No local checkpoint found")
            print("  → Would download from HuggingFace: ChatterjeeLab/MOG-DFM")
            
            # Determine cache location
            cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.environ.get('HF_HUB_CACHE')
            if cache_dir is None:
                if os.path.exists('/scratch'):
                    import getpass
                    username = getpass.getuser()
                    cache_dir = f'/scratch/pranamlab/{username}/.cache/huggingface'
                else:
                    cache_dir = str(Path.home() / ".cache" / "huggingface")
            
            print(f"  → Would cache to: {cache_dir}")
            
    except Exception as e:
        print(f"  ✗ Error checking: {e}")
    
    print("\n" + "=" * 70)
    print("To check HuggingFace cache contents, run:")
    print("  python -c \"from huggingface_hub import list_repo_files; print(list_repo_files('ChatterjeeLab/MOG-DFM'))\"")
    print("=" * 70)

if __name__ == '__main__':
    main()

