"""
Prototype Experiment: Comparing PepDFM (WT) vs PepMDLM (SMILES)

This script runs the experimental setup comparing:
- PepDFM: WT amino acid sequences (20 amino acids, discrete flow matching)
- PepMDLM: SMILES sequences (discrete diffusion from PepTune)

Under three conditions:
1. No language (predictor-only)
2. Neutral language prompt
3. Stability language prompt

IMPORTANT: Predictors (binding, hemolysis) expect SMILES format.
- PepDFM outputs WT sequences → converted to SMILES for predictor evaluation
- PepMDLM outputs SMILES → used directly for predictor evaluation
- This ensures fair comparison of predictor distributions across both generators

Evaluates:
- Predictor distributions (binding, hemolysis) - evaluated in SMILES format for both
- PepDFM metrics: protease motifs, proline fraction, Shannon entropy (on WT sequences)
- PepMDLM metrics: D-amino acids, N-methylation, terminal capping, cyclization (on SMILES)
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.base_generator import BaseGenerator
from generators.peptune_wrapper import PepMDLMWrapper, load_peptune_generator
from generators.dfm_wrapper import DFMWrapper, load_dfm_model
from language.text_encoder import load_text_encoder
from language.preference_net import load_preference_net
from predictors.loader import load_predictors
from lapep.sampler import sample_peptide, sample_from_fixed_seeds, mask_sequence
from scripts.load_test_set import load_test_set_csv, load_test_set_json


# ============================================================================
# Evaluation Metrics for WT Sequences (PepDFM)
# ============================================================================

def count_protease_motifs(sequence: str) -> int:
    """
    Count common protease cleavage motifs in WT amino acid sequence.
    
    Common motifs:
    - Trypsin: K/R (except when followed by P)
    - Chymotrypsin: F/Y/W/L (except when followed by P)
    - Elastase: A/V/L/I
    """
    count = 0
    
    # Trypsin: K or R not followed by P
    trypsin_pattern = r'[KR](?!P)'
    count += len(re.findall(trypsin_pattern, sequence))
    
    # Chymotrypsin: F/Y/W/L not followed by P
    chymotrypsin_pattern = r'[FYW](?!P)'
    count += len(re.findall(chymotrypsin_pattern, sequence))
    
    # Elastase: A/V/L/I
    elastase_pattern = r'[AVLI]'
    count += len(re.findall(elastase_pattern, sequence))
    
    return count


def compute_proline_fraction(sequence: str) -> float:
    """Compute fraction of proline residues in WT sequence."""
    if len(sequence) == 0:
        return 0.0
    return sequence.count('P') / len(sequence)


def compute_shannon_entropy(sequence: str) -> float:
    """
    Compute Shannon entropy of amino acid sequence.
    
    Higher entropy = more diverse sequence.
    """
    if len(sequence) == 0:
        return 0.0
    
    # Count amino acid frequencies
    counts = Counter(sequence)
    total = len(sequence)
    
    # Compute entropy
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def evaluate_wt_sequences(sequences: List[str]) -> Dict[str, np.ndarray]:
    """
    Evaluate WT amino acid sequences.
    
    Returns:
        Dict with arrays of metrics for each sequence
    """
    protease_counts = [count_protease_motifs(seq) for seq in sequences]
    proline_fractions = [compute_proline_fraction(seq) for seq in sequences]
    shannon_entropies = [compute_shannon_entropy(seq) for seq in sequences]
    
    return {
        'protease_motifs': np.array(protease_counts),
        'proline_fraction': np.array(proline_fractions),
        'shannon_entropy': np.array(shannon_entropies)
    }


# ============================================================================
# Evaluation Metrics for SMILES Sequences (PepMDLM)
# ============================================================================

def detect_d_amino_acids(smiles: str) -> bool:
    """
    Detect D-amino acids in SMILES.
    
    D-amino acids are typically represented with '@' or '@@' in SMILES.
    """
    # Look for @ symbols which indicate stereochemistry
    return '@' in smiles


def detect_n_methylation(smiles: str) -> bool:
    """
    Detect N-methylation in SMILES.
    
    N-methylation: N(C) or [N+](C) patterns
    """
    # Look for N-methyl patterns
    patterns = [
        r'N\(C\)',  # N(C)
        r'\[N\+\]\(C\)',  # [N+](C)
        r'N\[C\]',  # N[C]
    ]
    for pattern in patterns:
        if re.search(pattern, smiles):
            return True
    return False


def detect_terminal_capping(smiles: str) -> bool:
    """
    Detect terminal capping (acetylation or amidation) in SMILES.
    
    Acetylation: CC(=O)N- at N-terminus
    Amidation: -C(=O)N at C-terminus
    """
    # Acetylation: CC(=O)N at start
    if re.match(r'CC\(=O\)N', smiles):
        return True
    
    # Amidation: C(=O)N at end
    if re.search(r'C\(=O\)N$', smiles):
        return True
    
    return False


def detect_cyclization(smiles: str) -> bool:
    """
    Detect cyclization or stapling in SMILES.
    
    Cyclization: ring structures, often indicated by numbers in SMILES
    """
    # Look for ring closures (numbers in SMILES indicate ring bonds)
    if re.search(r'\d', smiles):
        return True
    
    # Look for explicit ring patterns
    if 'cyclo' in smiles.lower():
        return True
    
    return False


def evaluate_smiles_sequences(sequences: List[str]) -> Dict[str, np.ndarray]:
    """
    Evaluate SMILES sequences.
    
    Returns:
        Dict with arrays of metrics for each sequence
    """
    d_amino_acids = np.array([detect_d_amino_acids(s) for s in sequences], dtype=float)
    n_methylation = np.array([detect_n_methylation(s) for s in sequences], dtype=float)
    terminal_capping = np.array([detect_terminal_capping(s) for s in sequences], dtype=float)
    cyclization = np.array([detect_cyclization(s) for s in sequences], dtype=float)
    
    # Also compute protease motifs (convert SMILES to sequence first)
    protease_counts = []
    for smiles in sequences:
        # Try to extract sequence from SMILES
        # This is a simplified approach - in practice, use proper SMILES parser
        seq = _smiles_to_sequence_simple(smiles)
        protease_counts.append(count_protease_motifs(seq))
    
    return {
        'd_amino_acids': d_amino_acids,
        'n_methylation': n_methylation,
        'terminal_capping': terminal_capping,
        'cyclization': cyclization,
        'protease_motifs': np.array(protease_counts)
    }


def _smiles_to_sequence_simple(smiles: str) -> str:
    """
    Convert SMILES to WT amino acid sequence.
    
    Returns one-letter amino acid sequence.
    """
    try:
        from lapep.tr2d2.utils.app import PeptideAnalyzer
        analyzer = PeptideAnalyzer()
        seq_list = analyzer.return_sequence(smiles)
        one_letter = ''.join(
            analyzer.three_to_one.get(aa.split('(')[0], 'X') 
            for aa in seq_list
        )
        return one_letter
    except Exception:
        # Fallback: return empty string
        return ""


def _wt_to_smiles(wt_sequence: str) -> str:
    """
    Convert WT amino acid sequence to SMILES format.
    
    Uses RDKit to create linear peptide SMILES from one-letter amino acid codes.
    This creates a simple linear peptide without modifications.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        # Standard amino acid side chain SMILES (without N and C termini)
        # These are the side chains that get attached to the peptide backbone
        aa_side_chains = {
            'A': 'C',  # Alanine: CH3
            'C': 'CS',  # Cysteine: CH2SH
            'D': 'CC(=O)O',  # Aspartic acid: CH2COOH
            'E': 'CCC(=O)O',  # Glutamic acid: CH2CH2COOH
            'F': 'CC1=CC=CC=C1',  # Phenylalanine: CH2C6H5
            'G': '',  # Glycine: H (no side chain)
            'H': 'CC1=CNC=N1',  # Histidine: CH2-imidazole
            'I': 'CC(C)CC',  # Isoleucine: CH(CH3)CH2CH3
            'K': 'CCCCN',  # Lysine: CH2CH2CH2CH2NH2
            'L': 'CC(C)CC',  # Leucine: CH2CH(CH3)2
            'M': 'CCSC',  # Methionine: CH2CH2SCH3
            'N': 'CC(=O)N',  # Asparagine: CH2CONH2
            'P': 'CC1CCNC1',  # Proline: cyclic (special case)
            'Q': 'CCC(=O)N',  # Glutamine: CH2CH2CONH2
            'R': 'CCCCNC(=N)N',  # Arginine: CH2CH2CH2NHC(=NH)NH2
            'S': 'CO',  # Serine: CH2OH
            'T': 'CC(C)O',  # Threonine: CH(CH3)OH
            'V': 'CC(C)C',  # Valine: CH(CH3)2
            'W': 'CC1=CNC2=C1C=CC=C2',  # Tryptophan: CH2-indole
            'Y': 'CC1=CC=C(C=C1)O',  # Tyrosine: CH2C6H4OH
        }
        
        if not wt_sequence:
            return ""
        
        # Build peptide SMILES: N-term-[AA1]-[AA2]-...-[AA_n]-C-term
        # Peptide backbone: N-C(=O)-N-C(=O)-...-N-C(=O)-C(=O)O
        # Each amino acid contributes: N-C(=O) (backbone) + side chain
        
        smiles_parts = []
        for i, aa in enumerate(wt_sequence.upper()):
            if aa not in aa_side_chains:
                continue
            
            side_chain = aa_side_chains[aa]
            
            if i == 0:
                # N-terminal: H2N-CH(side_chain)-C(=O)
                if side_chain:
                    smiles_parts.append(f"NC({side_chain})C(=O)")
                else:
                    smiles_parts.append("NCC(=O)")  # Glycine
            else:
                # Middle residues: N-CH(side_chain)-C(=O)
                if side_chain:
                    smiles_parts.append(f"NC({side_chain})C(=O)")
                else:
                    smiles_parts.append("NCC(=O)")  # Glycine
        
        # C-terminal: add -OH
        if smiles_parts:
            # Remove last C(=O) and add C(=O)O
            smiles = ''.join(smiles_parts[:-1])
            if smiles_parts:
                last_part = smiles_parts[-1]
                # Replace last C(=O) with C(=O)O
                smiles = smiles + last_part.replace("C(=O)", "C(=O)O", 1)
            else:
                smiles = smiles_parts[0].replace("C(=O)", "C(=O)O", 1)
        else:
            return ""
        
        # Validate and canonicalize SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Try alternative: use simpler approach
            return ""
        
        return Chem.MolToSmiles(mol)
        
    except ImportError:
        # RDKit not available - return placeholder
        print("Warning: RDKit not available. Cannot convert WT to SMILES.")
        return ""
    except Exception as e:
        print(f"Warning: Failed to convert WT '{wt_sequence}' to SMILES: {e}")
        return ""


# ============================================================================
# Main Experimental Function
# ============================================================================

def run_prototype_experiment(
    config_path: str,
    peptune_model_path: str,
    dfm_model_path: Optional[str] = None,
    num_samples: int = 500,
    output_dir: Optional[str] = None,
    device: str = 'cuda',
    test_set_path: Optional[str] = None,
    constraint_strength: float = 2.0
):
    """
    Run the prototype experiment comparing PepDFM vs PepMDLM.
    
    Args:
        config_path: Path to config.json
        peptune_model_path: Path to PepMDLM (PepTune) model
        dfm_model_path: Path to PepDFM model (optional, if None will skip PepDFM)
        num_samples: Number of samples per condition
        output_dir: Output directory for results
        device: Device to run on
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"eval_results/prototype_experiment_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PROTOTYPE EXPERIMENT: PepDFM vs PepMDLM")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Samples per condition: {num_samples}")
    print("=" * 80)
    
    # Load predictors using unified loader
    # We'll load SMILES predictors for both generators since predictors expect SMILES
    # WT sequences will be converted to SMILES for predictor evaluation
    print("\nLoading predictors (SMILES format for both generators)...")
    
    # Create a temporary config for SMILES predictors (since both need SMILES format)
    smiles_config = config.copy()
    smiles_config['generator_type'] = 'pepmdlm'  # Force SMILES format
    
    predictors = load_predictors(
        smiles_config,
        format_type='smiles',  # Use SMILES predictors for both
        device=device,
        protein_seq=config.get('protein_seq')
    )
    
    # Ensure we have binding and hemolysis
    if 'binding' not in predictors:
        print("⚠ Warning: Binding predictor not found in config")
    if 'hemolysis' not in predictors:
        print("⚠ Warning: Hemolysis predictor not found in config")
        # Create placeholder if needed
        from predictors.smiles.hemolysis import HemolysisPredictor
        predictors['hemolysis'] = HemolysisPredictor(device=device)
        print("  Created placeholder hemolysis predictor")
    
    print(f"✓ Loaded {len(predictors)} predictor(s): {list(predictors.keys())}")
    
    # Load text encoder and preference network
    print("\nLoading language models...")
    text_encoder = load_text_encoder(config.get('text_encoder_name', 'Qwen/Qwen3-Embedding-0.6B'), device=device)
    preference_net_path = config.get('preference_net_path')
    if preference_net_path:
        preference_net = load_preference_net(preference_net_path, device=device)
        print("✓ Preference network loaded")
    else:
        preference_net = None
        print("⚠ No preference network found - will use predictor-only conditioning")
    
    # Load test set seeds if provided (for Algorithm 3)
    seed_set = None
    use_algorithm_3 = False
    if test_set_path:
        print(f"\nLoading test set from {test_set_path}...")
        test_set_path_obj = Path(test_set_path)
        if test_set_path_obj.suffix == '.csv':
            test_pairs = load_test_set_csv(test_set_path)
            # Extract peptide sequences as seeds (convert to appropriate format)
            seed_set = [pair['peptide'] for pair in test_pairs]
            print(f"✓ Loaded {len(seed_set)} seeds from test set")
            use_algorithm_3 = True
        elif test_set_path_obj.suffix == '.json':
            test_pairs = load_test_set_json(test_set_path)
            seed_set = [pair['peptide'] for pair in test_pairs]
            print(f"✓ Loaded {len(seed_set)} seeds from test set")
            use_algorithm_3 = True
        else:
            print(f"⚠ Unknown file format: {test_set_path_obj.suffix}. Using Algorithm 2 instead.")
    
    if use_algorithm_3:
        print("\n" + "="*80)
        print("USING ALGORITHM 3: Controlled Evaluation with Fixed Seeds")
        print("="*80)
        print(f"Seed set size: {len(seed_set)}")
        print(f"Completions per seed: {num_samples}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("USING ALGORITHM 2: Standard Sampling")
        print("="*80)
    
    # Define prompts
    prompts = {
        'no_language': None,
        'neutral': "Design a peptide with good binding and low hemolysis.",
        'stability': "Design a peptide that is protease-resistant and stable in vivo, while maintaining good binding and low hemolysis."
    }
    
    # Load generators
    print("\nLoading generators...")
    generators = {}
    
    # PepMDLM (SMILES)
    print("Loading PepMDLM (SMILES)...")
    try:
        generators['pepmdlm'] = load_peptune_generator(peptune_model_path, device=device)
        if generators['pepmdlm'].model is None:
            raise RuntimeError("Failed to load PepMDLM model")
        print("✓ PepMDLM loaded")
    except Exception as e:
        print(f"✗ Failed to load PepMDLM: {e}")
        generators['pepmdlm'] = None
    
    # PepDFM (WT)
    if dfm_model_path:
        print("Loading PepDFM (WT)...")
        try:
            generators['pepdfm'] = load_dfm_model(dfm_model_path)
            if generators['pepdfm'] is None:
                raise NotImplementedError("PepDFM loading returned None")
            print("✓ PepDFM loaded")
        except NotImplementedError as e:
            print(f"✗ PepDFM not implemented: {e}")
            print("  Note: PepDFM wrapper needs to be implemented in generators/dfm_wrapper.py")
            generators['pepdfm'] = None
        except Exception as e:
            print(f"✗ Failed to load PepDFM: {e}")
            generators['pepdfm'] = None
    else:
        print("⚠ PepDFM model path not provided - skipping")
        generators['pepdfm'] = None
    
    # Run experiments
    results = {}
    
    for generator_name, generator in generators.items():
        if generator is None:
            continue
        
        print(f"\n{'='*80}")
        print(f"Running experiments for {generator_name.upper()}")
        print(f"{'='*80}")
        
        generator_results = {}
        
        for condition_name, prompt in prompts.items():
            print(f"\nCondition: {condition_name}")
            print(f"Prompt: {prompt if prompt else 'None (predictor-only)'}")
            
            completion_metadata = None  # Initialize for Algorithm 3
            
            # Generate samples using Algorithm 3 (controlled) or Algorithm 2 (standard)
            if use_algorithm_3 and seed_set:
                # Algorithm 3: Controlled evaluation from fixed seeds
                print(f"Generating {num_samples} completions per seed from {len(seed_set)} seeds...")
                print(f"  Using fixed mask pattern for fair comparison across conditions")
                
                # Use fixed mask rate (50%) for all seeds
                mask_rate = 0.5
                
                completions_data = sample_from_fixed_seeds(
                    base_generator=generator,
                    seed_set=seed_set,
                    prompt=prompt,
                    predictors=predictors,
                    constraints={'strength': constraint_strength, 'weights': {'binding': 1.0, 'hemolysis': 1.0}},  # Tight constraints
                    text_encoder=text_encoder if prompt else None,
                    preference_net=preference_net if prompt else None,
                    num_steps=50,
                    use_linear_preferences=False,
                    schedule=None,
                    mask_rate=mask_rate,
                    mask_positions=None,  # Random but fixed per seed
                    completions_per_seed=num_samples,
                    seed=42,  # Fixed seed for reproducibility
                    language_weight=1.0  # Default language weight
                )
                
                # Extract just the completions
                samples = [comp['completion'] for comp in completions_data]
                print(f"  Generated {len(samples)} completions from {len(seed_set)} seeds")
                
                # Store completion metadata for analysis
                completion_metadata = completions_data
            else:
                # Algorithm 2: Standard sampling
                print(f"Generating {num_samples} samples...")
                samples = []
                for i in range(num_samples):
                    try:
                        if prompt is None or preference_net is None:
                            # Predictor-only conditioning
                            peptide = sample_peptide(
                                generator,
                                prompt=None,
                                predictors=predictors,
                                constraints={'strength': constraint_strength, 'weights': {'binding': 1.0, 'hemolysis': 1.0}},  # Tight constraints
                                text_encoder=None,
                                preference_net=None,
                                num_steps=50,
                                language_weight=0.0  # No language effect
                            )
                        else:
                            # Language conditioning
                            peptide = sample_peptide(
                                generator,
                                prompt=prompt,
                                predictors=predictors,
                                constraints={'strength': constraint_strength, 'weights': {'binding': 1.0, 'hemolysis': 1.0}},  # Tight constraints
                                text_encoder=text_encoder,
                                preference_net=preference_net,
                                num_steps=50,
                                language_weight=1.0  # Normal language weight
                            )
                        samples.append(peptide)
                    except Exception as e:
                        print(f"  Warning: Failed to generate sample {i}: {e}")
                        continue
                
                print(f"  Generated {len(samples)} valid samples")
            
            # IMPORTANT: Predictors expect SMILES format
            # So we need to convert WT sequences to SMILES for predictor evaluation
            print("Computing predictor scores (predictors expect SMILES format)...")
            binding_scores = []
            hemolysis_scores = []
            
            # Determine sample format and convert to SMILES for predictors
            smiles_for_predictors = []
            wt_sequences = []  # For WT-specific metrics
            
            for sample in samples:
                # Check if sample is WT sequence (only standard amino acids) or SMILES
                is_wt = all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in sample.upper())
                
                if is_wt:
                    # PepDFM output: WT sequence
                    wt_seq = sample.upper()
                    wt_sequences.append(wt_seq)
                    # Convert WT to SMILES for predictor evaluation
                    smiles = _wt_to_smiles(wt_seq)
                    if smiles:
                        smiles_for_predictors.append(smiles)
                    else:
                        print(f"  Warning: Could not convert WT sequence to SMILES: {wt_seq[:20]}...")
                        smiles_for_predictors.append("")  # Will skip predictor evaluation
                else:
                    # PepMDLM output: SMILES
                    smiles_for_predictors.append(sample)
                    # Extract WT sequence from SMILES for protease motif analysis
                    wt_seq = _smiles_to_sequence_simple(sample)
                    wt_sequences.append(wt_seq if wt_seq else "")
            
            # Evaluate predictors on SMILES format
            for smiles in smiles_for_predictors:
                if smiles:
                    try:
                        binding_scores.append(predictors['binding'].predict(smiles))
                        hemolysis_scores.append(predictors['hemolysis'].predict(smiles))
                    except Exception as e:
                        print(f"  Warning: Predictor evaluation failed: {e}")
                        binding_scores.append(np.nan)
                        hemolysis_scores.append(np.nan)
                else:
                    binding_scores.append(np.nan)
                    hemolysis_scores.append(np.nan)
            
            binding_scores = np.array(binding_scores)
            hemolysis_scores = np.array(hemolysis_scores)
            
            # Compute evaluation metrics based on generator type
            print("Computing generator-specific evaluation metrics...")
            if generator_name == 'pepdfm':
                # PepDFM: Evaluate as WT sequences (protease motifs, proline, entropy)
                valid_wt = [s for s in wt_sequences if s]
                print(f"  Evaluating {len(valid_wt)} WT sequences")
                metrics = evaluate_wt_sequences(valid_wt)
            else:
                # PepMDLM: Evaluate as SMILES (modifications, etc.)
                print(f"  Evaluating {len(samples)} SMILES sequences")
                metrics = evaluate_smiles_sequences(samples)
            
            condition_data = {
                'samples': samples,
                'binding_scores': binding_scores,
                'hemolysis_scores': hemolysis_scores,
                'metrics': metrics
            }
            
            # Store completion metadata if using Algorithm 3
            if use_algorithm_3 and completion_metadata is not None:
                condition_data['completion_metadata'] = completion_metadata
            
            generator_results[condition_name] = condition_data
        
        results[generator_name] = generator_results
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    # Save raw results (without samples to save space)
    results_to_save = {}
    for gen_name, gen_results in results.items():
        results_to_save[gen_name] = {}
        for cond_name, cond_data in gen_results.items():
            results_to_save[gen_name][cond_name] = {
                'binding_scores': cond_data['binding_scores'].tolist(),
                'hemolysis_scores': cond_data['hemolysis_scores'].tolist(),
                'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in cond_data['metrics'].items()}
            }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Create tables and visualizations
    print("Creating tables and visualizations...")
    create_tables(results, output_dir)
    create_histograms(results, output_dir)
    create_combined_table(results, output_dir)
    
    print(f"\n{'='*80}")
    print("Experiment complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


def create_tables(results: Dict, output_dir: Path):
    """Create comparison tables for all metrics."""
    tables = {}
    
    for generator_name, generator_results in results.items():
        table_data = []
        
        for condition_name, condition_data in generator_results.items():
            metrics = condition_data['metrics']
            binding_mean = np.mean(condition_data['binding_scores'])
            hemolysis_mean = np.mean(condition_data['hemolysis_scores'])
            
            row = {
                'Condition': condition_name,
                'Binding (mean)': f"{binding_mean:.4f}",
                'Hemolysis (mean)': f"{hemolysis_mean:.4f}"
            }
            
            # Add generator-specific metrics
            for metric_name, metric_values in metrics.items():
                if isinstance(metric_values, np.ndarray):
                    row[metric_name] = f"{np.mean(metric_values):.4f}"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        tables[generator_name] = df
        
        # Save as CSV
        df.to_csv(output_dir / f"table_{generator_name}.csv", index=False)
        
        # Save as LaTeX
        latex_table = df.to_latex(index=False, float_format="%.4f")
        with open(output_dir / f"table_{generator_name}.tex", 'w') as f:
            f.write(latex_table)
    
    print(f"✓ Tables saved to {output_dir}")


def create_histograms(results: Dict, output_dir: Path):
    """Create predictor distribution histograms showing overlap across conditions."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Create combined histograms comparing both generators
    for predictor_name in ['binding', 'hemolysis']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, (generator_name, generator_results) in enumerate(results.items()):
            ax = axes[idx]
            
            # Collect all scores for this generator across conditions
            all_scores = []
            condition_labels = []
            
            for condition_name, condition_data in generator_results.items():
                scores = condition_data[f'{predictor_name}_scores']
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) > 0:
                    all_scores.append(valid_scores)
                    condition_labels.append(condition_name)
            
            # Create histogram with overlapping distributions
            if all_scores:
                # Use same bins for all conditions
                all_values = np.concatenate(all_scores)
                bins = np.linspace(all_values.min(), all_values.max(), 30)
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
                for i, (scores, label) in enumerate(zip(all_scores, condition_labels)):
                    ax.hist(scores, bins=bins, alpha=0.6, label=label, 
                           color=colors[i % len(colors)], density=True, edgecolor='black', linewidth=0.5)
                
                ax.set_xlabel(f'{predictor_name.capitalize()} Score', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f'{predictor_name.capitalize()} Distribution - {generator_name.upper()}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f"{predictor_name}_histogram_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create individual histograms per generator
    for generator_name, generator_results in results.items():
        for predictor_name in ['binding', 'hemolysis']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            all_scores = []
            condition_labels = []
            for condition_name, condition_data in generator_results.items():
                scores = condition_data[f'{predictor_name}_scores']
                valid_scores = scores[~np.isnan(scores)]
                if len(valid_scores) > 0:
                    all_scores.append(valid_scores)
                    condition_labels.append(condition_name)
            
            if all_scores:
                all_values = np.concatenate(all_scores)
                bins = np.linspace(all_values.min(), all_values.max(), 30)
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                for i, (scores, label) in enumerate(zip(all_scores, condition_labels)):
                    ax.hist(scores, bins=bins, alpha=0.6, label=label,
                           color=colors[i % len(colors)], density=True, edgecolor='black', linewidth=0.5)
                
                ax.set_xlabel(f'{predictor_name.capitalize()} Score', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f'{predictor_name.capitalize()} Distribution - {generator_name.upper()}\n(Overlapping distributions show constraint matching)', 
                            fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(fig_dir / f"{predictor_name}_histogram_{generator_name}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Histograms saved to {fig_dir}")


def create_combined_table(results: Dict, output_dir: Path):
    """Create a combined table comparing both generators."""
    combined_data = []
    
    for generator_name, generator_results in results.items():
        for condition_name, condition_data in generator_results.items():
            metrics = condition_data['metrics']
            binding_scores = condition_data['binding_scores']
            hemolysis_scores = condition_data['hemolysis_scores']
            
            binding_valid = binding_scores[~np.isnan(binding_scores)]
            hemolysis_valid = hemolysis_scores[~np.isnan(hemolysis_scores)]
            
            row = {
                'Generator': generator_name.upper(),
                'Condition': condition_name,
                'Binding (mean±std)': f"{np.mean(binding_valid):.4f}±{np.std(binding_valid):.4f}" if len(binding_valid) > 0 else "N/A",
                'Hemolysis (mean±std)': f"{np.mean(hemolysis_valid):.4f}±{np.std(hemolysis_valid):.4f}" if len(hemolysis_valid) > 0 else "N/A"
            }
            
            # Add generator-specific metrics
            if generator_name == 'pepdfm':
                for metric_name in ['protease_motifs', 'proline_fraction', 'shannon_entropy']:
                    if metric_name in metrics:
                        metric_values = metrics[metric_name]
                        if isinstance(metric_values, np.ndarray):
                            valid_values = metric_values[~np.isnan(metric_values)]
                            if len(valid_values) > 0:
                                row[metric_name] = f"{np.mean(valid_values):.4f}"
            else:
                for metric_name in ['d_amino_acids', 'n_methylation', 'terminal_capping', 'cyclization', 'protease_motifs']:
                    if metric_name in metrics:
                        metric_values = metrics[metric_name]
                        if isinstance(metric_values, np.ndarray):
                            valid_values = metric_values[~np.isnan(metric_values)]
                            if len(valid_values) > 0:
                                row[metric_name] = f"{np.mean(valid_values):.4f}"
            
            combined_data.append(row)
    
    df = pd.DataFrame(combined_data)
    df.to_csv(output_dir / "table_combined.csv", index=False)
    
    # LaTeX version
    latex_str = "\\begin{table}[h]\n\\centering\n"
    latex_str += df.to_latex(index=False, escape=False)
    latex_str += "\\caption{Combined metrics comparison: PepDFM vs PepMDLM}\n"
    latex_str += "\\end{table}\n"
    with open(output_dir / "table_combined.tex", 'w') as f:
        f.write(latex_str)


def main():
    parser = argparse.ArgumentParser(
        description="Run prototype experiment comparing PepDFM vs PepMDLM"
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--peptune_model', type=str, required=True, help='Path to PepMDLM (PepTune) model')
    parser.add_argument('--dfm_model', type=str, default=None, help='Path to PepDFM model (optional)')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples per condition')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--test_set', type=str, default=None, help='Path to test set (CSV or JSON) for Algorithm 3')
    parser.add_argument('--constraint_strength', type=float, default=2.0, help='Constraint strength (higher = tighter enforcement)')
    
    args = parser.parse_args()
    
    run_prototype_experiment(
        config_path=args.config,
        peptune_model_path=args.peptune_model,
        dfm_model_path=args.dfm_model,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device,
        test_set_path=args.test_set,
        constraint_strength=args.constraint_strength
    )


if __name__ == '__main__':
    main()

