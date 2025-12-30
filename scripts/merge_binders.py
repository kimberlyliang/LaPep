"""
Merge benchmark binders with existing wt_peptides.json file.

This script combines protein binders from the benchmark with existing
peptide data for use in controlled optimization experiments.
"""

import json
import argparse
from pathlib import Path


def merge_binders(
    benchmark_file: str,
    existing_file: str,
    output_file: str,
    overwrite: bool = False
):
    """
    Merge benchmark binders with existing peptide data.
    
    Args:
        benchmark_file: Path to benchmark_binders.json
        existing_file: Path to existing wt_peptides.json (or similar)
        output_file: Path to output merged file
        overwrite: If True, overwrite existing entries with benchmark data
    """
    # Load benchmark binders
    benchmark_path = Path(benchmark_file)
    if not benchmark_path.exists():
        print(f"Error: Benchmark file not found: {benchmark_file}")
        return
    
    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)
    
    print(f"Loaded {len(benchmark_data)} entries from benchmark file")
    
    # Load existing data
    existing_path = Path(existing_file)
    if existing_path.exists():
        with open(existing_path, 'r') as f:
            existing_data = json.load(f)
        print(f"Loaded {len(existing_data)} entries from existing file")
    else:
        existing_data = {}
        print("No existing file found, starting fresh")
    
    # Merge data
    merged_data = existing_data.copy()
    
    for protein_id, binder_info in benchmark_data.items():
        if protein_id in merged_data and not overwrite:
            print(f"  Skipping {protein_id} (already exists, use --overwrite to replace)")
        else:
            # Convert benchmark format to match existing format
            merged_data[protein_id] = {
                'protein_id': binder_info.get('protein_id', protein_id),
                'starting_peptide': binder_info.get('starting_peptide', '')
            }
            print(f"  Added/Updated {protein_id}")
    
    # Save merged data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\n✓ Merged data saved to: {output_file}")
    print(f"  Total entries: {len(merged_data)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge benchmark binders with existing peptide data"
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='data/benchmark_binders.json',
        help='Path to benchmark_binders.json'
    )
    parser.add_argument(
        '--existing',
        type=str,
        default='data/dnu_wt_peptides.json',
        help='Path to existing wt_peptides.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/all_protein_binders.json',
        help='Output file path'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing entries with benchmark data'
    )
    
    args = parser.parse_args()
    
    merge_binders(
        args.benchmark,
        args.existing,
        args.output,
        args.overwrite
    )

