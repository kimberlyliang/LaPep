"""
Helper script to get sample SMILES sequences for testing.

This script provides several ways to obtain SMILES sequences:
1. Generate samples from PepMDLM (recommended)
2. Convert WT sequences from benchmark data to SMILES
3. Load from existing test sets
4. Create a small sample set from generated peptides
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.peptune_wrapper import load_peptune_generator
from lapep.sampler import sample_peptide
from predictors.loader import load_predictors
from lapep.tr2d2.utils.app import PeptideAnalyzer


def generate_smiles_samples(
    peptune_model_path: str,
    num_samples: int = 100,
    device: str = 'cuda',
    config_path: Optional[str] = None
) -> List[str]:
    """
    Generate SMILES samples using PepMDLM.
    
    Args:
        peptune_model_path: Path to PepMDLM model
        num_samples: Number of samples to generate
        device: Device to run on
        config_path: Optional config path for predictors (if you want conditioned samples)
    
    Returns:
        List of SMILES strings
    """
    print(f"Loading PepMDLM from {peptune_model_path}...")
    generator = load_peptune_generator(peptune_model_path, device=device)
    
    if config_path:
        # Load predictors for conditioned generation
        with open(config_path, 'r') as f:
            config = json.load(f)
        predictors = load_predictors(config, format_type='smiles', device=device)
        print("Generating conditioned samples...")
    else:
        # Unconditioned generation
        predictors = {}
        print("Generating unconditioned samples...")
    
    samples = []
    for i in range(num_samples):
        try:
            peptide = sample_peptide(
                generator,
                prompt=None,
                predictors=predictors if predictors else None,
                constraints={'strength': 1.0} if predictors else None,
                text_encoder=None,
                preference_net=None,
                num_steps=50
            )
            samples.append(peptide)
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples...")
        except Exception as e:
            print(f"  Warning: Failed to generate sample {i}: {e}")
            continue
    
    print(f"✓ Generated {len(samples)} SMILES samples")
    return samples


def convert_wt_to_smiles(wt_sequences: List[str]) -> List[str]:
    """
    Convert WT amino acid sequences to SMILES format.
    
    Args:
        wt_sequences: List of WT amino acid sequences
    
    Returns:
        List of SMILES strings
    """
    analyzer = PeptideAnalyzer()
    smiles_list = []
    
    for wt_seq in wt_sequences:
        try:
            # Use PeptideAnalyzer to convert WT to SMILES
            # This is a simplified approach - you may need to adjust based on your needs
            seq_list = analyzer.return_sequence(wt_seq)
            # Reconstruct SMILES from sequence
            # Note: This is a placeholder - you may need a proper WT->SMILES converter
            smiles = analyzer.sequence_to_smiles(seq_list) if hasattr(analyzer, 'sequence_to_smiles') else None
            if smiles:
                smiles_list.append(smiles)
            else:
                print(f"  Warning: Could not convert WT sequence: {wt_seq[:20]}...")
        except Exception as e:
            print(f"  Warning: Failed to convert {wt_seq[:20]}...: {e}")
            continue
    
    return smiles_list


def extract_from_benchmark(
    benchmark_path: str = "data/benchmark_binders.json",
    max_samples: Optional[int] = None
) -> List[str]:
    """
    Extract peptide sequences from benchmark data.
    
    Note: These are WT sequences - you'll need to convert them to SMILES.
    
    Args:
        benchmark_path: Path to benchmark JSON file
        max_samples: Maximum number of samples to extract
    
    Returns:
        List of peptide sequences (WT format)
    """
    with open(benchmark_path, 'r') as f:
        data = json.load(f)
    
    peptides = []
    for protein_id, entry in data.items():
        # Get starting peptide (pre-existing binder)
        if 'starting_peptide' in entry and entry['starting_peptide']:
            peptides.append(entry['starting_peptide'])
        
        # Get designed binder if available
        if 'designed_binder' in entry and entry['designed_binder']:
            peptides.append(entry['designed_binder'])
        
        if max_samples and len(peptides) >= max_samples:
            break
    
    print(f"✓ Extracted {len(peptides)} peptide sequences from benchmark")
    return peptides


def save_smiles_samples(smiles_list: List[str], output_path: str, format: str = 'json'):
    """
    Save SMILES samples to file.
    
    Args:
        smiles_list: List of SMILES strings
        output_path: Output file path
        format: 'json' or 'txt' (one per line)
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        data = {
            'smiles_samples': smiles_list,
            'num_samples': len(smiles_list)
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        with open(output_path, 'w') as f:
            for smiles in smiles_list:
                f.write(f"{smiles}\n")
    
    print(f"✓ Saved {len(smiles_list)} SMILES samples to {output_path}")


def create_test_set_from_smiles(
    smiles_list: List[str],
    protein_seq: str,
    output_path: str
):
    """
    Create a test set CSV file from SMILES sequences.
    
    Args:
        smiles_list: List of SMILES strings
        protein_seq: Protein sequence to pair with peptides
        output_path: Output CSV path
    """
    import pandas as pd
    
    data = {
        'protein_sequence': [protein_seq] * len(smiles_list),
        'peptide_sequence': smiles_list,
        'binding_affinity': [0.0] * len(smiles_list)  # Placeholder
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ Created test set CSV with {len(smiles_list)} pairs at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Get sample SMILES sequences for testing"
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['generate', 'extract', 'convert'],
        default='generate',
        help='Method to get SMILES: generate (from PepMDLM), extract (from benchmark), convert (WT to SMILES)'
    )
    parser.add_argument(
        '--peptune_model',
        type=str,
        help='Path to PepMDLM model (required for generate method)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.json (optional, for conditioned generation)'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default='data/benchmark_binders.json',
        help='Path to benchmark JSON (for extract method)'
    )
    parser.add_argument(
        '--wt_sequences',
        type=str,
        nargs='+',
        help='WT sequences to convert (for convert method)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to generate/extract'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path (JSON or TXT)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'txt', 'csv'],
        default='json',
        help='Output format'
    )
    parser.add_argument(
        '--protein_seq',
        type=str,
        help='Protein sequence (required for CSV format)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )
    
    args = parser.parse_args()
    
    smiles_samples = []
    
    if args.method == 'generate':
        if not args.peptune_model:
            print("Error: --peptune_model is required for generate method")
            return
        smiles_samples = generate_smiles_samples(
            args.peptune_model,
            num_samples=args.num_samples,
            device=args.device,
            config_path=args.config
        )
    
    elif args.method == 'extract':
        wt_sequences = extract_from_benchmark(args.benchmark, max_samples=args.num_samples)
        print(f"\nNote: Extracted {len(wt_sequences)} WT sequences.")
        print("To convert to SMILES, use --method convert with these sequences.")
        print("Or use the PepMDLM generator to generate SMILES directly.")
        # For now, we'll just save the WT sequences
        if args.format == 'json':
            data = {
                'wt_sequences': wt_sequences,
                'num_samples': len(wt_sequences),
                'note': 'These are WT sequences. Convert to SMILES using a converter or PepMDLM generator.'
            }
            with open(args.output, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            with open(args.output, 'w') as f:
                for seq in wt_sequences:
                    f.write(f"{seq}\n")
        print(f"✓ Saved WT sequences to {args.output}")
        return
    
    elif args.method == 'convert':
        if not args.wt_sequences:
            print("Error: --wt_sequences is required for convert method")
            return
        smiles_samples = convert_wt_to_smiles(args.wt_sequences)
    
    if not smiles_samples:
        print("No SMILES samples generated. Exiting.")
        return
    
    # Save samples
    if args.format == 'csv':
        if not args.protein_seq:
            print("Error: --protein_seq is required for CSV format")
            return
        create_test_set_from_smiles(smiles_samples, args.protein_seq, args.output)
    else:
        save_smiles_samples(smiles_samples, args.output, format=args.format)
    
    print(f"\n✓ Done! Generated {len(smiles_samples)} SMILES samples")
    print(f"  First few samples:")
    for i, smiles in enumerate(smiles_samples[:5]):
        print(f"    {i+1}. {smiles[:80]}..." if len(smiles) > 80 else f"    {i+1}. {smiles}")


if __name__ == '__main__':
    main()

