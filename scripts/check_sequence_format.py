"""
Quick script to check if sequences in benchmark data are WT or SMILES.
"""

import json
import re
from pathlib import Path

def is_wt_sequence(seq: str) -> bool:
    """Check if sequence is WT (only standard amino acids)."""
    if not seq:
        return False
    # WT sequences only contain standard amino acid one-letter codes
    wt_chars = set('ACDEFGHIKLMNPQRSTVWY')
    return all(c.upper() in wt_chars for c in seq if c.isalpha())

def is_smiles_sequence(seq: str) -> bool:
    """Check if sequence looks like SMILES."""
    if not seq:
        return False
    # SMILES contains special characters like (, ), =, [, ], @, numbers
    smiles_indicators = ['(', ')', '=', '[', ']', '@', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    return any(char in seq for char in smiles_indicators)

# Load benchmark data
benchmark_path = Path("data/benchmark_binders.json")
if benchmark_path.exists():
    with open(benchmark_path, 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("CHECKING SEQUENCE FORMATS IN BENCHMARK DATA")
    print("=" * 80)
    
    wt_count = 0
    smiles_count = 0
    unknown_count = 0
    
    sample_wt = []
    sample_smiles = []
    
    for protein_id, entry in list(data.items())[:10]:  # Check first 10
        # Check starting peptide
        if 'starting_peptide' in entry:
            seq = entry['starting_peptide']
            if seq:
                if is_wt_sequence(seq):
                    wt_count += 1
                    if len(sample_wt) < 3:
                        sample_wt.append((protein_id, 'starting_peptide', seq))
                elif is_smiles_sequence(seq):
                    smiles_count += 1
                    if len(sample_smiles) < 3:
                        sample_smiles.append((protein_id, 'starting_peptide', seq))
                else:
                    unknown_count += 1
        
        # Check designed binder
        if 'designed_binder' in entry:
            seq = entry['designed_binder']
            if seq:
                if is_wt_sequence(seq):
                    wt_count += 1
                    if len(sample_wt) < 3:
                        sample_wt.append((protein_id, 'designed_binder', seq))
                elif is_smiles_sequence(seq):
                    smiles_count += 1
                    if len(sample_smiles) < 3:
                        sample_smiles.append((protein_id, 'designed_binder', seq))
                else:
                    unknown_count += 1
    
    print(f"\nFormat Analysis (first 10 proteins):")
    print(f"  WT sequences (amino acids): {wt_count}")
    print(f"  SMILES sequences: {smiles_count}")
    print(f"  Unknown format: {unknown_count}")
    
    if sample_wt:
        print(f"\nSample WT sequences:")
        for protein_id, field, seq in sample_wt:
            print(f"  {protein_id} ({field}): {seq}")
    
    if sample_smiles:
        print(f"\nSample SMILES sequences:")
        for protein_id, field, seq in sample_smiles:
            print(f"  {protein_id} ({field}): {seq[:80]}..." if len(seq) > 80 else f"  {protein_id} ({field}): {seq}")
    
    if not sample_smiles and sample_wt:
        print(f"\n✓ All sequences appear to be WT (amino acid) format, not SMILES.")
        print(f"  To use them with SMILES predictors, you'll need to convert them.")
        print(f"  Use: python scripts/get_sample_smiles.py --method generate")
    elif sample_smiles:
        print(f"\n✓ Found SMILES sequences in the data!")
    
    print("=" * 80)
else:
    print(f"Benchmark file not found at {benchmark_path}")

