"""
Helper script to load test set protein-peptide pairs.

Common sources for test sets:
1. TR2-D2 benchmark datasets
2. Peptide-protein binding databases (e.g., PepBDB, PeptideDB)
3. Published benchmark datasets from papers
4. Your own curated test set

This script provides utilities to load test sets in various formats.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv


def load_test_set_csv(
    csv_path: str,
    protein_col: str = 'protein_sequence',
    peptide_col: str = 'peptide_sequence',
    label_col: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Load test set from CSV file.
    
    Expected CSV format:
    - protein_sequence: Protein sequence (amino acids)
    - peptide_sequence: Peptide sequence (SMILES or amino acids)
    - label (optional): Binding affinity or other label
    
    Args:
        csv_path: Path to CSV file
        protein_col: Column name for protein sequences
        peptide_col: Column name for peptide sequences
        label_col: Optional column name for labels
        
    Returns:
        List of dicts with 'protein', 'peptide', and optionally 'label'
    """
    df = pd.read_csv(csv_path)
    
    test_pairs = []
    for _, row in df.iterrows():
        pair = {
            'protein': str(row[protein_col]),
            'peptide': str(row[peptide_col])
        }
        if label_col and label_col in row:
            pair['label'] = row[label_col]
        test_pairs.append(pair)
    
    return test_pairs


def load_test_set_json(
    json_path: str,
    protein_key: str = 'protein',
    peptide_key: str = 'peptide'
) -> List[Dict[str, str]]:
    """
    Load test set from JSON file.
    
    Expected JSON format:
    [
        {"protein": "MMDQARSAF...", "peptide": "CC(=O)N..."},
        ...
    ]
    
    Args:
        json_path: Path to JSON file
        protein_key: Key for protein sequences
        peptide_key: Key for peptide sequences
        
    Returns:
        List of dicts with 'protein' and 'peptide'
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'test_pairs' in data:
        return data['test_pairs']
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}")


def create_test_set_template(output_path: str, num_examples: int = 10):
    """
    Create a template CSV file for test set.
    
    Args:
        output_path: Where to save the template
        num_examples: Number of example rows to include
    """
    template_data = {
        'protein_sequence': ['MMDQARSAFSNLFGGEPLSYTRFSLARQVDGDNSHVEMKLAVDEEENADNNTKANVTKPKRCSGSICYGTIAVIVFFLIGFMIGYLGYCKGVEPKTECERLAGTESPVREEPGEDFPAARRLYWDDLKRKLSEKLDSTDFTGTIKLLNENSYVPREAGSQKDENLALYVENQFREFKLSKVWRDQHFVKIQVKDSAQNSVIIVDKNGRLVYLVENPGGYVAYSKAATVTGKLVHANFGTKKDFEDLYTPVNGSIVIVRAGKITFAEKVANAESLNAIGVLIYMDQTKFPIVNAELSFFGHAHLGTGDPYTPGFPSFNHTQFPPSRSSGLPNIPVQTISRAAAEKLFGNMEGDCPSDWKTDSTCRMVTSESKNVKLTVSNVLKEIKILNIFGVIKGFVEPDHYVVVGAQRDAWGPGAAKSGVGTALLLKLAQMFSDMVLKDGFQPSRSIIFASWSAGDFGSVGATEWLEGYLSSLHLKAFTYINLDKAVLGTSNFKVSASPLLYTLIEKTMQNVKHPVTGQFLYQDSNWASKVEKLTLDNAAFPFLAYSGIPAVSFCFCEDTDYPYLGTTMDTYKELIERIPELNKVARAAAEVAGQFVIKLTHDVELNLDYERYNSQLLSFVRDLNQYRADIKEMGLSLQWLYSARGDFFRATSRLTTDFGNAEKTDRFVMKKLNDRVMRVEYHFLSPYVSPKESPFRHVFWGSGSHTLPALLENLKLRKQNNGAFNETLFRNQLALATWTIQGAANALSGDVWDIDNEF'] * num_examples,
        'peptide_sequence': [''] * num_examples,  # Fill in your peptide sequences
        'binding_affinity': [0.0] * num_examples,  # Optional: experimental binding values
        'notes': [''] * num_examples
    }
    
    df = pd.DataFrame(template_data)
    df.to_csv(output_path, index=False)
    print(f"Template saved to {output_path}")
    print(f"Please fill in the peptide_sequence column with your test peptides (SMILES format)")


def print_test_set_sources():
    """Print information about common test set sources."""
    print("=" * 80)
    print("COMMON TEST SET SOURCES FOR PROTEIN-PEPTIDE BINDING")
    print("=" * 80)
    print("""
1. TR2-D2 Benchmark Datasets
   - Check the TR2-D2 repository for benchmark datasets
   - Usually includes protein-peptide binding pairs with experimental data

2. Peptide-Protein Binding Databases:
   - PepBDB: Peptide binding database
   - PeptideDB: Comprehensive peptide database
   - BindingDB: Protein-ligand binding database

3. Published Benchmark Datasets:
   - Check papers on peptide-protein binding prediction
   - Common benchmarks: PepBind, PepNN, etc.

4. Your Own Test Set:
   - Curate from literature
   - Use experimentally validated binding pairs
   - Format: CSV or JSON with protein and peptide sequences

5. Format Requirements:
   - Protein sequences: Amino acid sequences (one-letter codes)
   - Peptide sequences: SMILES format (for binding predictor compatibility)
   - Optional: Experimental binding affinities (pIC50, Kd, etc.)

To create a template CSV file, run:
    python scripts/load_test_set.py --create_template test_set_template.csv
    """)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load or create test set for protein-peptide pairs"
    )
    parser.add_argument(
        '--load_csv',
        type=str,
        help='Load test set from CSV file'
    )
    parser.add_argument(
        '--load_json',
        type=str,
        help='Load test set from JSON file'
    )
    parser.add_argument(
        '--create_template',
        type=str,
        help='Create a template CSV file for test set'
    )
    parser.add_argument(
        '--sources',
        action='store_true',
        help='Print information about test set sources'
    )
    
    args = parser.parse_args()
    
    if args.sources:
        print_test_set_sources()
    elif args.create_template:
        create_test_set_template(args.create_template)
    elif args.load_csv:
        pairs = load_test_set_csv(args.load_csv)
        print(f"Loaded {len(pairs)} test pairs from {args.load_csv}")
        print(f"First pair: protein={pairs[0]['protein'][:50]}..., peptide={pairs[0]['peptide'][:50]}...")
    elif args.load_json:
        pairs = load_test_set_json(args.load_json)
        print(f"Loaded {len(pairs)} test pairs from {args.load_json}")
        print(f"First pair: protein={pairs[0]['protein'][:50]}..., peptide={pairs[0]['peptide'][:50]}...")
    else:
        parser.print_help()
        print("\nUse --sources to see information about test set sources")

