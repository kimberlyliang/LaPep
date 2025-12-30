"""
Extract proteins without pre-existing binders from Benchmark_moPPIt_v3.xlsx.

This script reads the first sheet (proteins without pre-existing binders)
and creates a JSON file with protein targets for de novo design.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extract_benchmark_binders import extract_binders


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract proteins without pre-existing binders from benchmark Excel file"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/Benchmark_moPPIt_v3.xlsx',
        help='Path to benchmark Excel file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/proteins_without_binders.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Extracting proteins WITHOUT pre-existing binders")
    print("(First sheet of Benchmark_moPPIt_v3.xlsx)")
    print("="*80)
    
    # Extract from first sheet (index 0)
    extract_binders(
        args.input,
        args.output,
        sheet_index=0
    )
    
    print("\n" + "="*80)
    print("Note: These proteins don't have pre-existing binders.")
    print("Use them for de novo peptide design experiments.")
    print("="*80)


if __name__ == '__main__':
    main()

