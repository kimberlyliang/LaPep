"""
Test Set Peptide Optimization Experiment

This experiment:
1. Uses known protein-peptide pairs from test set (benchmark_binders.json)
2. Optimizes existing peptides (not de novo design) using Algorithm 3
3. Tests properties WITH predictors (binding, hemolysis, toxicity)
4. Tests properties WITHOUT predictors but empirically known (protease resistance, stability motifs)

This is the "biggest test" - can language guidance help optimize for properties
we don't have predictors for but empirically know how to achieve?
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd
import sys
import re
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.peptune_wrapper import load_peptune_generator
from generators.dfm_wrapper import load_dfm_model
from language.text_encoder import load_text_encoder
from language.preference_net import load_preference_net
from predictors.loader import load_predictors
from predictors import detect_sequence_format, is_wt_sequence, is_smiles_sequence
from lapep.sampler import sample_from_fixed_seeds
from lapep.tr2d2.utils.app import PeptideAnalyzer


# ============================================================================
# Empirical Metrics (No Predictors)
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
    """Compute fraction of proline residues (stability indicator)."""
    if not sequence:
        return 0.0
    return sequence.count('P') / len(sequence)


def detect_stability_patterns(sequence: str) -> Dict[str, float]:
    """
    Detect empirically known stability patterns.
    
    Returns:
        - proline_fraction: Higher = more stable
        - protease_motifs: Lower = more stable (fewer cleavage sites)
        - charged_residue_fraction: Moderate = better (too high = aggregation)
        - hydrophobic_fraction: Moderate = better
    """
    if not sequence:
        return {
            'proline_fraction': 0.0,
            'protease_motifs': 0,
            'charged_fraction': 0.0,
            'hydrophobic_fraction': 0.0
        }
    
    charged = set('KRDE')
    hydrophobic = set('AVILMFYW')
    
    return {
        'proline_fraction': compute_proline_fraction(sequence),
        'protease_motifs': count_protease_motifs(sequence),
        'charged_fraction': sum(1 for aa in sequence if aa in charged) / len(sequence),
        'hydrophobic_fraction': sum(1 for aa in sequence if aa in hydrophobic) / len(sequence)
    }


def smiles_to_wt_sequence(smiles: str) -> Optional[str]:
    """Convert SMILES to WT amino acid sequence."""
    try:
        analyzer = PeptideAnalyzer()
        wt_seq = analyzer.smiles_to_sequence(smiles)
        return wt_seq if wt_seq else None
    except Exception as e:
        print(f"Warning: Could not convert SMILES to WT: {e}")
        return None


# ============================================================================
# Main Experiment
# ============================================================================

def run_test_set_optimization(
    config_path: str,
    test_set_path: str,
    generator_type: str = 'pepmdlm',
    generator_path: Optional[str] = None,
    num_optimization_steps: int = 50,
    num_completions_per_seed: int = 10,
    output_dir: Optional[str] = None,
    device: str = 'cuda',
    constraint_strength: float = 2.0,
    language_weight: float = 1.0,
    max_proteins: Optional[int] = None
):
    """
    Run test set optimization experiment.
    
    Args:
        config_path: Path to config.json
        test_set_path: Path to benchmark_binders.json (test set)
        generator_type: 'pepmdlm' (SMILES) or 'pepdfm' (WT)
        generator_path: Path to generator model
        num_optimization_steps: Number of optimization steps
        num_completions_per_seed: Number of completions per starting peptide
        output_dir: Output directory for results
        device: Device to run on
        constraint_strength: Strength of predictor constraints
        language_weight: Weight for language conditioning
        max_proteins: Maximum number of proteins to process (None = all)
    """
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"eval_results/test_set_optimization_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TEST SET PEPTIDE OPTIMIZATION EXPERIMENT")
    print("=" * 80)
    print(f"Test set: {test_set_path}")
    print(f"Generator: {generator_type}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load test set
    print(f"\nLoading test set from {test_set_path}...")
    with open(test_set_path, 'r') as f:
        test_set = json.load(f)
    
    protein_ids = list(test_set.keys())
    if max_proteins:
        protein_ids = protein_ids[:max_proteins]
    print(f"✓ Loaded {len(protein_ids)} protein targets")
    
    # Load generator
    print(f"\nLoading {generator_type.upper()} generator...")
    if generator_type == 'pepmdlm':
        if generator_path is None:
            generator_path = config.get('base_generator_path')
        generator = load_peptune_generator(generator_path, device=device)
    elif generator_type == 'pepdfm':
        if generator_path is None:
            generator_path = config.get('dfm_model_path')
        generator = load_dfm_model(generator_path)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
    print(f"✓ Generator loaded")
    
    # Load predictors
    print("\nLoading predictors...")
    format_type = 'wt' if generator_type == 'pepdfm' else 'smiles'
    predictors = load_predictors(
        config,
        format_type=format_type,
        device=device,
        protein_seq=config.get('protein_seq')
    )
    print(f"✓ Loaded {len(predictors)} predictor(s): {list(predictors.keys())}")
    
    # Load language models
    print("\nLoading language models...")
    text_encoder = load_text_encoder(config.get('text_encoder_name'), device=device)
    preference_net = load_preference_net(config.get('preference_net_path'), device=device)
    print("✓ Language models loaded")
    
    # Define language conditions
    conditions = {
        'no_language': None,
        'binding_optimization': "Optimize this peptide for high binding affinity to the target protein while maintaining low toxicity and hemolysis.",
        'stability_optimization': "Optimize this peptide to be protease-resistant and stable in vivo. Reduce protease cleavage sites, increase proline content, and maintain good binding affinity.",
        'binding_and_stability': "Optimize this peptide for both high binding affinity and protease resistance. Increase stability while maintaining strong binding."
    }
    
    # Constraints (same across all conditions)
    constraints = {
        'strength': constraint_strength,
        'weights': {
            'binding': 1.0,
            'hemolysis': 1.0,
            'toxicity': 1.0 if 'toxicity' in predictors else 0.0
        }
    }
    
    # Store all results
    all_results = []
    
    # Process each protein target
    for protein_idx, protein_id in enumerate(protein_ids):
        print(f"\n{'='*80}")
        print(f"Processing {protein_idx+1}/{len(protein_ids)}: {protein_id}")
        print(f"{'='*80}")
        
        protein_data = test_set[protein_id]
        starting_peptide = protein_data.get('starting_peptide')
        
        if not starting_peptide:
            print(f"  ⚠ Skipping {protein_id}: No starting peptide")
            continue
        
        print(f"  Starting peptide: {starting_peptide}")
        
        # Convert to appropriate format if needed
        if generator_type == 'pepmdlm' and is_wt_sequence(starting_peptide):
            # Need to convert WT to SMILES (this is a limitation - we'd need a converter)
            print(f"  ⚠ Warning: Starting peptide is WT but generator expects SMILES")
            print(f"     Skipping {protein_id} - need WT→SMILES converter")
            continue
        elif generator_type == 'pepdfm' and is_smiles_sequence(starting_peptide):
            # Convert SMILES to WT
            wt_seq = smiles_to_wt_sequence(starting_peptide)
            if wt_seq:
                starting_peptide = wt_seq
            else:
                print(f"  ⚠ Warning: Could not convert SMILES to WT")
                continue
        
        # Evaluate starting peptide
        print(f"\n  Evaluating starting peptide...")
        starting_metrics = evaluate_peptide(
            starting_peptide,
            predictors,
            generator_type,
            protein_id
        )
        
        print(f"    Binding: {starting_metrics['predictor_scores']['binding']:.3f}")
        print(f"    Hemolysis: {starting_metrics['predictor_scores']['hemolysis']:.3f}")
        print(f"    Protease motifs: {starting_metrics['empirical_metrics']['protease_motifs']}")
        print(f"    Proline fraction: {starting_metrics['empirical_metrics']['proline_fraction']:.3f}")
        
        # Run optimization for each condition
        for condition_name, prompt in conditions.items():
            print(f"\n  Condition: {condition_name}")
            if prompt:
                print(f"    Prompt: {prompt[:80]}...")
            
            # Generate optimized completions using Algorithm 3
            try:
                completions_data = sample_from_fixed_seeds(
                    base_generator=generator,
                    seed_set=[starting_peptide],
                    prompt=prompt,
                    predictors=predictors,
                    constraints=constraints,
                    text_encoder=text_encoder if prompt else None,
                    preference_net=preference_net if prompt else None,
                    num_steps=num_optimization_steps,
                    use_linear_preferences=False,
                    schedule=None,
                    mask_rate=0.5,  # Mask 50% of starting peptide
                    mask_positions=None,
                    completions_per_seed=num_completions_per_seed,
                    seed=42,  # Fixed seed for reproducibility
                    language_weight=language_weight
                )
                
                optimized_peptides = [comp['completion'] for comp in completions_data]
                print(f"    ✓ Generated {len(optimized_peptides)} optimized variants")
                
            except Exception as e:
                print(f"    ✗ Error during optimization: {e}")
                import traceback
                traceback.print_exc()
                optimized_peptides = []
            
            # Evaluate optimized peptides
            if optimized_peptides:
                optimized_metrics_list = []
                for opt_pep in optimized_peptides:
                    metrics = evaluate_peptide(
                        opt_pep,
                        predictors,
                        generator_type,
                        protein_id
                    )
                    optimized_metrics_list.append(metrics)
                
                # Aggregate metrics
                avg_metrics = aggregate_metrics(optimized_metrics_list)
                
                # Compare to starting peptide
                improvement = compute_improvement(starting_metrics, avg_metrics)
                
                # Store results
                result = {
                    'protein_id': protein_id,
                    'condition': condition_name,
                    'prompt': prompt,
                    'starting_peptide': starting_peptide,
                    'num_optimized': len(optimized_peptides),
                    'starting_metrics': starting_metrics,
                    'optimized_metrics': avg_metrics,
                    'improvement': improvement
                }
                all_results.append(result)
                
                # Print summary
                print(f"    Improvement:")
                print(f"      Binding: {improvement['predictor_improvement']['binding']:+.3f}")
                print(f"      Hemolysis: {improvement['predictor_improvement']['hemolysis']:+.3f}")
                print(f"      Protease motifs: {improvement['empirical_improvement']['protease_motifs']:+.1f} (lower is better)")
                print(f"      Proline fraction: {improvement['empirical_improvement']['proline_fraction']:+.3f}")
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    # Save raw results
    results_path = output_dir / "optimization_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved raw results to {results_path}")
    
    # Create summary table
    summary_df = create_summary_table(all_results)
    summary_path = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary table to {summary_path}")
    
    # Create LaTeX table
    latex_path = output_dir / "summary_table.tex"
    with open(latex_path, 'w') as f:
        f.write(summary_df.to_latex(index=False, float_format="%.3f"))
    print(f"✓ Saved LaTeX table to {latex_path}")
    
    # Print overall summary
    print_summary(all_results)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")


def evaluate_peptide(
    peptide: str,
    predictors: Dict,
    generator_type: str,
    protein_id: str
) -> Dict:
    """
    Evaluate a peptide on both predictor-based and empirical metrics.
    
    Returns:
        Dict with 'predictor_scores' and 'empirical_metrics'
    """
    # Get sequence in appropriate format
    if generator_type == 'pepdfm':
        # WT sequence
        wt_seq = peptide
    else:
        # SMILES - convert to WT for empirical metrics
        wt_seq = smiles_to_wt_sequence(peptide)
        if not wt_seq:
            wt_seq = peptide  # Fallback
    
    # Predictor scores
    predictor_scores = {}
    for pred_name, predictor in predictors.items():
        try:
            raw_score = predictor.predict(peptide)
            normalized = predictor.normalize(raw_score)
            predictor_scores[pred_name] = {
                'raw': float(raw_score),
                'normalized': float(normalized)
            }
        except Exception as e:
            print(f"    Warning: Predictor {pred_name} failed: {e}")
            predictor_scores[pred_name] = {'raw': np.nan, 'normalized': np.nan}
    
    # Empirical metrics (no predictors)
    empirical_metrics = detect_stability_patterns(wt_seq)
    
    return {
        'peptide': peptide,
        'wt_sequence': wt_seq,
        'predictor_scores': predictor_scores,
        'empirical_metrics': empirical_metrics
    }


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """Aggregate metrics across multiple peptides."""
    if not metrics_list:
        return {}
    
    # Aggregate predictor scores
    predictor_names = list(metrics_list[0]['predictor_scores'].keys())
    aggregated_predictors = {}
    for pred_name in predictor_names:
        raw_scores = [m['predictor_scores'][pred_name]['raw'] for m in metrics_list]
        normalized_scores = [m['predictor_scores'][pred_name]['normalized'] for m in metrics_list]
        
        aggregated_predictors[pred_name] = {
            'raw_mean': float(np.nanmean(raw_scores)),
            'raw_std': float(np.nanstd(raw_scores)),
            'normalized_mean': float(np.nanmean(normalized_scores)),
            'normalized_std': float(np.nanstd(normalized_scores))
        }
    
    # Aggregate empirical metrics
    empirical_keys = list(metrics_list[0]['empirical_metrics'].keys())
    aggregated_empirical = {}
    for key in empirical_keys:
        values = [m['empirical_metrics'][key] for m in metrics_list]
        aggregated_empirical[key] = {
            'mean': float(np.nanmean(values)),
            'std': float(np.nanstd(values))
        }
    
    return {
        'predictor_scores': aggregated_predictors,
        'empirical_metrics': aggregated_empirical
    }


def compute_improvement(starting: Dict, optimized: Dict) -> Dict:
    """Compute improvement from starting to optimized peptide."""
    improvement = {
        'predictor_improvement': {},
        'empirical_improvement': {}
    }
    
    # Predictor improvements (higher normalized = better for binding, lower = better for hemolysis/toxicity)
    for pred_name in starting['predictor_scores'].keys():
        if pred_name in optimized['predictor_scores']:
            start_norm = starting['predictor_scores'][pred_name]['normalized']
            opt_norm = optimized['predictor_scores'][pred_name]['normalized_mean']
            
            # For binding: higher is better
            # For hemolysis/toxicity: lower is better
            if pred_name == 'binding':
                improvement['predictor_improvement'][pred_name] = opt_norm - start_norm
            else:
                improvement['predictor_improvement'][pred_name] = start_norm - opt_norm  # Negative = improvement
    
    # Empirical improvements
    for key in starting['empirical_metrics'].keys():
        if key in optimized['empirical_metrics']:
            start_val = starting['empirical_metrics'][key]
            opt_val = optimized['empirical_metrics'][key]['mean']
            
            if key == 'protease_motifs':
                # Lower is better
                improvement['empirical_improvement'][key] = start_val - opt_val
            elif key == 'proline_fraction':
                # Higher is better
                improvement['empirical_improvement'][key] = opt_val - start_val
            else:
                # Neutral (just report difference)
                improvement['empirical_improvement'][key] = opt_val - start_val
    
    return improvement


def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    """Create summary table of results."""
    rows = []
    
    for result in results:
        row = {
            'protein_id': result['protein_id'],
            'condition': result['condition'],
            'binding_improvement': result['improvement']['predictor_improvement'].get('binding', np.nan),
            'hemolysis_improvement': result['improvement']['predictor_improvement'].get('hemolysis', np.nan),
            'protease_motifs_change': result['improvement']['empirical_improvement'].get('protease_motifs', np.nan),
            'proline_fraction_change': result['improvement']['empirical_improvement'].get('proline_fraction', np.nan),
            'num_optimized': result['num_optimized']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_summary(results: List[Dict]):
    """Print overall summary statistics."""
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    # Group by condition
    by_condition = defaultdict(list)
    for result in results:
        by_condition[result['condition']].append(result)
    
    for condition_name, condition_results in by_condition.items():
        print(f"\n{condition_name.upper()}:")
        print(f"  Number of proteins: {len(condition_results)}")
        
        # Average improvements
        binding_improvements = [r['improvement']['predictor_improvement'].get('binding', 0) for r in condition_results]
        hemolysis_improvements = [r['improvement']['predictor_improvement'].get('hemolysis', 0) for r in condition_results]
        protease_changes = [r['improvement']['empirical_improvement'].get('protease_motifs', 0) for r in condition_results]
        proline_changes = [r['improvement']['empirical_improvement'].get('proline_fraction', 0) for r in condition_results]
        
        print(f"  Avg binding improvement: {np.mean(binding_improvements):+.3f}")
        print(f"  Avg hemolysis improvement: {np.mean(hemolysis_improvements):+.3f}")
        print(f"  Avg protease motifs change: {np.mean(protease_changes):+.1f} (negative = fewer motifs)")
        print(f"  Avg proline fraction change: {np.mean(proline_changes):+.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test set peptide optimization experiment"
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--test_set', type=str, default='data/benchmark_binders.json', help='Path to test set JSON')
    parser.add_argument('--generator_type', type=str, choices=['pepmdlm', 'pepdfm'], default='pepmdlm', help='Generator type')
    parser.add_argument('--generator_path', type=str, default=None, help='Path to generator model')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of optimization steps')
    parser.add_argument('--num_completions', type=int, default=10, help='Number of completions per seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--constraint_strength', type=float, default=2.0, help='Constraint strength')
    parser.add_argument('--language_weight', type=float, default=1.0, help='Language weight')
    parser.add_argument('--max_proteins', type=int, default=None, help='Maximum number of proteins to process')
    
    args = parser.parse_args()
    
    run_test_set_optimization(
        config_path=args.config,
        test_set_path=args.test_set,
        generator_type=args.generator_type,
        generator_path=args.generator_path,
        num_optimization_steps=args.num_steps,
        num_completions_per_seed=args.num_completions,
        output_dir=args.output_dir,
        device=args.device,
        constraint_strength=args.constraint_strength,
        language_weight=args.language_weight,
        max_proteins=args.max_proteins
    )


if __name__ == '__main__':
    main()

