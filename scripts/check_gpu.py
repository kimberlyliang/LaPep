#!/usr/bin/env python3
"""
Quick GPU availability checker for LaPep.
"""

import sys

def main():
    print("=" * 70)
    print("GPU Availability Check")
    print("=" * 70)
    
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            
            # Check current device
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                print(f"\n  Current device: {current_device}")
                print(f"  Memory allocated: {torch.cuda.memory_allocated(current_device) / 1e9:.2f} GB")
                print(f"  Memory reserved: {torch.cuda.memory_reserved(current_device) / 1e9:.2f} GB")
        else:
            print("  → Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        # Check MPS (Apple Silicon)
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        print(f"\nMPS (Apple Silicon) available: {mps_available}")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("Recommendations:")
        print("-" * 70)
        
        if cuda_available:
            print("✓ Use --device cuda for GPU acceleration")
            print("  Example: python scripts/train_preferences.py --config config.json --device cuda")
        elif mps_available:
            print("✓ Use --device mps for Apple Silicon acceleration")
            print("  Example: python scripts/train_preferences.py --config config.json --device mps")
        else:
            print("⚠ No GPU available - will use CPU")
            print("  Example: python scripts/train_preferences.py --config config.json --device cpu")
            print("\n  To enable CUDA:")
            print("    1. Install NVIDIA drivers")
            print("    2. Install CUDA toolkit")
            print("    3. Reinstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        print("=" * 70)
        
    except ImportError:
        print("\n✗ PyTorch not installed")
        print("  Install with: pip install torch")
        sys.exit(1)

if __name__ == '__main__':
    main()

