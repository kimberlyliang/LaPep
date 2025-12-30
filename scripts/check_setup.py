#!/usr/bin/env python3
"""
Comprehensive setup checker for LaPep.

Checks all dependencies, models, and configuration.
"""

import sys
from pathlib import Path
import importlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """Check if all required packages can be imported."""
    print("\n1. CHECKING DEPENDENCIES:")
    print("-" * 70)
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'transformers': 'Transformers',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
    }
    
    optional_packages = {
        'huggingface_hub': 'HuggingFace Hub',
        'openpyxl': 'OpenPyXL (for Excel files)',
        'lightning': 'PyTorch Lightning',
        'fair_esm': 'fair-esm',
        'SmilesPE': 'SmilesPE',
    }
    
    all_ok = True
    for module_name, display_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} MISSING - install with: pip install {module_name}")
            all_ok = False
    
    print("\n  Optional packages:")
    for module_name, display_name in optional_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"    ✓ {display_name}")
        except ImportError:
            print(f"    - {display_name} (optional)")
    
    return all_ok

def check_models():
    """Check if required model files exist."""
    print("\n2. CHECKING MODEL CHECKPOINTS:")
    print("-" * 70)
    
    repo_root = Path(__file__).parent.parent
    
    models = {
        'pretrained/generators/peptune-pretrained.ckpt': 'PepMDLM (SMILES)',
        'pretrained/generators/pepdfm.ckpt': 'PepDFM (WT)',
        'pretrained/language/preference_net.ckpt': 'Preference Network',
    }
    
    all_present = True
    for model_path, description in models.items():
        full_path = repo_root / model_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / 1e6
            print(f"  ✓ {model_path} ({size_mb:.2f} MB) - {description}")
        else:
            print(f"  ✗ {model_path} MISSING - {description}")
            all_present = False
    
    return all_present

def check_config():
    """Check if config.json exists and is valid."""
    print("\n3. CHECKING CONFIGURATION:")
    print("-" * 70)
    
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "config.json"
    template_path = repo_root / "config_template.json"
    
    if config_path.exists():
        print(f"  ✓ config.json exists")
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check important fields
            if 'generator_type' in config:
                print(f"    generator_type: {config['generator_type']}")
            else:
                print(f"    ⚠ generator_type not set")
            
            if 'base_generator_path' in config:
                gen_path = Path(config['base_generator_path'])
                if gen_path.exists():
                    print(f"    ✓ base_generator_path exists: {config['base_generator_path']}")
                else:
                    print(f"    ✗ base_generator_path missing: {config['base_generator_path']}")
            
            return True
        except Exception as e:
            print(f"  ✗ Error reading config.json: {e}")
            return False
    else:
        print(f"  ✗ config.json MISSING")
        if template_path.exists():
            print(f"    → Copy from template: cp config_template.json config.json")
        return False

def check_code_imports():
    """Check if LaPep code can be imported."""
    print("\n4. CHECKING CODE IMPORTS:")
    print("-" * 70)
    
    imports_to_test = [
        ('generators.dfm_wrapper', 'load_dfm_model', 'PepDFM wrapper'),
        ('generators.peptune_wrapper', 'load_peptune_generator', 'PepMDLM wrapper'),
        ('predictors.loader', 'load_predictors', 'Predictor loader'),
        ('language.text_encoder', 'load_text_encoder', 'Text encoder'),
        ('language.preference_net', 'load_preference_net', 'Preference network'),
        ('lapep.sampler', 'sample_peptide', 'LaPep sampler'),
    ]
    
    all_ok = True
    for module_name, func_name, description in imports_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, func_name):
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description} - {func_name} not found")
                all_ok = False
        except Exception as e:
            print(f"  ✗ {description} - Import error: {e}")
            all_ok = False
    
    return all_ok

def check_device():
    """Check available compute devices."""
    print("\n5. CHECKING COMPUTE DEVICES:")
    print("-" * 70)
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  - CUDA not available")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ✓ MPS (Apple Silicon) available")
        else:
            print(f"  - MPS not available")
        
        print(f"  → Will use: CPU (always available)")
        return True
    except ImportError:
        print(f"  ✗ PyTorch not installed")
        return False

def main():
    print("=" * 70)
    print("LaPep Setup Checker")
    print("=" * 70)
    
    results = {
        'dependencies': check_imports(),
        'models': check_models(),
        'config': check_config(),
        'code': check_code_imports(),
        'device': check_device(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("-" * 70)
    
    all_ready = True
    for check_name, passed in results.items():
        status = "✓ READY" if passed else "✗ NOT READY"
        print(f"  {check_name.upper()}: {status}")
        if not passed:
            all_ready = False
    
    print("\n" + "=" * 70)
    if all_ready:
        print("✅ ALL CHECKS PASSED - You're ready to run experiments!")
        print("\nNext steps:")
        print("  python scripts/run_eval.py --config config.json")
    else:
        print("⚠️  SOME CHECKS FAILED - See details above")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Create config: cp config_template.json config.json")
        print("  3. Download models: python scripts/download_pepdfm.py")
        print("  4. Train preference net: python scripts/train_preferences.py")
    print("=" * 70)

if __name__ == '__main__':
    main()

