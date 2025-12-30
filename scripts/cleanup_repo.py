#!/usr/bin/env python3
"""
Cleanup script for LaPep repository.

Removes temporary files, organizes example files, and prepares the repo for use.
"""

import os
import shutil
from pathlib import Path

def main():
    repo_root = Path(__file__).parent.parent
    
    print("=" * 70)
    print("LaPep Repository Cleanup")
    print("=" * 70)
    
    # 1. Move example PepDFM files to docs/examples (for reference)
    print("\n1. Organizing example files...")
    examples_dir = repo_root / "docs" / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    example_files = [
        "PepDFM_unconditioned_sample.py",
        "PepDFM_test.py",
        "PepDFM_multiobjectivegeneration.py",
    ]
    
    for filename in example_files:
        src = repo_root / filename
        if src.exists():
            dst = examples_dir / filename
            if not dst.exists():
                shutil.move(str(src), str(dst))
                print(f"  ✓ Moved {filename} → docs/examples/")
            else:
                print(f"  - {filename} already in docs/examples/")
                if src.exists():
                    src.unlink()
                    print(f"    Removed duplicate from root")
    
    # 2. Clean up __pycache__ directories (they'll be regenerated)
    print("\n2. Cleaning Python cache...")
    pycache_dirs = list(repo_root.rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"  ✓ Removed {pycache_dir.relative_to(repo_root)}")
        except Exception as e:
            print(f"  ⚠ Could not remove {pycache_dir}: {e}")
    
    # 3. Check for .pyc files
    pyc_files = list(repo_root.rglob("*.pyc"))
    for pyc_file in pyc_files:
        try:
            pyc_file.unlink()
            print(f"  ✓ Removed {pyc_file.relative_to(repo_root)}")
        except Exception as e:
            print(f"  ⚠ Could not remove {pyc_file}: {e}")
    
    # 4. Verify important directories exist
    print("\n3. Verifying directory structure...")
    required_dirs = [
        "pretrained/generators",
        "pretrained/language",
        "pretrained/predictors",
        "data",
        "docs",
        "scripts",
    ]
    
    for dir_path in required_dirs:
        full_path = repo_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/ exists")
        else:
            print(f"  ⚠ {dir_path}/ missing - creating...")
            full_path.mkdir(parents=True, exist_ok=True)
    
    # 5. Check for required checkpoint files
    print("\n4. Checking required files...")
    required_files = {
        "pretrained/generators/peptune-pretrained.ckpt": "PepMDLM model",
        "pretrained/generators/pepdfm.ckpt": "PepDFM model",
        "pretrained/language/preference_net.ckpt": "Preference network",
        "config_template.json": "Config template",
    }
    
    for file_path, description in required_files.items():
        full_path = repo_root / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / 1e6
            print(f"  ✓ {file_path} ({size_mb:.2f} MB) - {description}")
        else:
            print(f"  ⚠ {file_path} MISSING - {description}")
    
    # 6. Create .gitkeep files in empty directories if needed
    print("\n5. Ensuring git tracking...")
    gitkeep_dirs = [
        "pretrained/predictors/wt",
        "pretrained/predictors/smiles",
    ]
    
    for dir_path in gitkeep_dirs:
        full_path = repo_root / dir_path
        gitkeep = full_path / ".gitkeep"
        if full_path.exists() and not any(full_path.iterdir()):
            gitkeep.touch()
            print(f"  ✓ Created {dir_path}/.gitkeep")
    
    print("\n" + "=" * 70)
    print("Cleanup complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review SETUP_CHECKLIST.md for setup instructions")
    print("  2. Install dependencies: pip install -r requirements.txt")
    print("  3. Download missing models if needed")
    print("  4. Copy config_template.json to config.json and configure")

if __name__ == '__main__':
    main()

