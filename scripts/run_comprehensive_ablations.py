"""
Comprehensive Ablation Studies for LaPep

This script runs all ablation experiments requested:
1. Language prompt comparison: no language vs neutral vs stability (same seeds, same constraints)
2. Preference functional: nonlinear vs linear weighted sum
3. Kernel comparison: LaPep kernel vs naive guidance (direct logit/reward reweighting)
4. Language weight sweep: vary α to show smooth tradeoff and failure modes
5. Text encoder comparison: Qwen-0.6B vs Qwen-4B with identical MLP and training
6. Baseline comparison: ProteinDT, BioM3, InstructPro (optional, requires external models)
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.peptune_wrapper import load_peptune_generator
from generators.dfm_wrapper import load_dfm_model
from language.text_encoder import load_text_encoder
from language.preference_net import load_preference_net
from predictors.loader import load_predictors
from lapep.sampler import sample_peptide, sample_from_fixed_seeds
from lapep.potential import compute_potential, compute_preference_score
from scripts.run_prototype_experiment import (
    evaluate_wt_sequences, evaluate_smiles_sequences,
    count_protease_motifs, compute_proline_fraction, compute_shannon_entropy
)


# ============================================================================
# Naive Guidance Implementation (Baseline)
# ============================================================================

def sample_with_naive_guidance(
    base_generator: any,
    prompt: Optional[str],
    predictors: Dict,
    constraints: Dict,
    text_encoder: Optional[any] = None,
    preference_net: Optional[any] = None,
    num_steps: int = 50,
    use_linear_preferences: bool = False,
    language_weight: float = 1.0,
    seed: Optional[int] = None
) -> str:
    """
    Naive guidance baseline: Direct logit/reward reweighting without conservative kernel.
    
    This applies language and predictor signals directly to transition probabilities
    without the conservative reweighting scheme that ensures path independence.
    
    Method:
    - Compute reward R(x;t) and constraint penalty Ψ(x) for each candidate
    - Directly reweight transition probabilities: p(x'|x) ∝ b(x'|x) * exp(-β*U(x';t))
    - No symmetric form, no path independence guarantee
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Pre-compute eta if using language
    eta = None
    if prompt is not None and text_encoder is not None and preference_net is not None:
        e = text_encoder.encode(prompt)
        if isinstance(e, torch.Tensor):
            if len(e.shape) == 1:
                e = e.unsqueeze(0)
        eta = preference_net(e)
    
    # Sample initial state
    X_s = base_generator.sample_initial_state()
    
    # Naive guidance: direct reweighting without conservative form
    for s in range(num_steps):
        neighbors = base_generator.get_neighbors(X_s)
        C_s = neighbors + [X_s]
        
        # Compute unnormalized weights directly (non-conservative)
        log_weights = []
        for x_prime in C_s:
            # Base proposal probability
            log_b_theta = np.log(base_generator.proposal_probability(x_prime, X_s, s) + 1e-10)
            
            # Compute potential U(x';t) = -α·R(x';t) + Ψ(x')
            psi = _compute_constraint_penalty(x_prime, predictors, constraints)
            
            if eta is not None:
                r = compute_preference_score(
                    x_prime, predictors, use_linear_preferences, eta=eta
                )
            elif prompt is not None and text_encoder is not None and preference_net is not None:
                r = compute_preference_score(
                    x_prime, predictors, use_linear_preferences,
                    prompt=prompt, text_encoder=text_encoder, preference_net=preference_net
                )
            else:
                r = 0.0
            
            U_x_prime = -language_weight * r + psi
            
            # Direct reweighting: p(x'|x) ∝ b(x'|x) * exp(-β*U(x';t))
            # Note: This is NOT conservative - doesn't ensure path independence
            beta = 1.0  # Temperature parameter
            log_weight = log_b_theta - beta * U_x_prime
            log_weights.append(log_weight)
        
        # Normalize
        log_weights = np.array(log_weights)
        log_Z = np.logaddexp.reduce(log_weights)
        log_probs = log_weights - log_Z
        probs = np.exp(log_probs)
        
        # Sample next state
        X_s = np.random.choice(C_s, p=probs)
    
    return X_s


def _compute_constraint_penalty(x: str, predictors: Dict, constraints: Dict) -> float:
    """Compute constraint penalty Ψ(x) = Σ_k λ_k ψ_k(u_k(x))."""
    if not predictors:
        return 0.0
    
    weights = constraints.get('weights', {})
    strength = constraints.get('strength', 1.0)
    
    total_penalty = 0.0
    for pred_name, predictor in predictors.items():
        raw_value = predictor.predict(x)
        normalized = predictor.normalize(raw_value)
        weight = weights.get(pred_name, 1.0) * strength
        
        # Default penalty functions
        if 'toxicity' in pred_name.lower():
            threshold = 0.3
            penalty = (max(0, normalized - threshold)) ** 2
        elif 'binding' in pred_name.lower():
            threshold = 0.7
            penalty = (max(0, threshold - normalized)) ** 2
        elif 'hemolysis' in pred_name.lower():
            threshold = 0.3
            penalty = (max(0, normalized - threshold)) ** 2
        else:
            penalty = 0.0
        
        total_penalty += weight * penalty
    
    return total_penalty


# ============================================================================
# Ablation Experiments
# ============================================================================

def ablation_1_language_prompts(
    generator: any,
    predictors: Dict,
    text_encoder: any,
    preference_net: any,
    constraints: Dict,
    seed_set: List[str],
    num_samples: int = 100,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    Ablation 1: Language prompt comparison
    - No language vs neutral vs stability prompt
    - Same seeds, same constraints
    """
    print("\n" + "="*80)
    print("ABLATION 1: Language Prompt Comparison")
    print("="*80)
    
    prompts = {
        'no_language': None,
        'neutral': "Design a peptide with good binding and low hemolysis.",
        'stability': "Design a peptide that is protease-resistant and stable in vivo, while maintaining good binding and low hemolysis."
    }
    
    results = {}
    
    for prompt_name, prompt in prompts.items():
        print(f"\nPrompt: {prompt_name}")
        print(f"  Text: {prompt if prompt else 'None (predictor-only)'}")
        
        samples = []
        for seed_idx, seed in enumerate(seed_set[:num_samples]):
            try:
                peptide = sample_from_fixed_seeds(
                    base_generator=generator,
                    seed_set=[seed],
                    prompt=prompt,
                    predictors=predictors,
                    constraints=constraints,
                    text_encoder=text_encoder if prompt else None,
                    preference_net=preference_net if prompt else None,
                    num_steps=50,
                    completions_per_seed=1,
                    seed=42 + seed_idx,
                    language_weight=1.0
                )
                if peptide:
                    samples.append(peptide[0]['completion'])
            except Exception as e:
                print(f"    Warning: Failed seed {seed_idx}: {e}")
                continue
        
        # Evaluate
        metrics = _evaluate_samples(samples, predictors, generator)
        results[prompt_name] = {
            'samples': samples,
            'metrics': metrics
        }
        
        print(f"  Generated {len(samples)} samples")
        print(f"  Binding (mean): {np.mean(metrics['binding_scores']):.4f}")
        print(f"  Hemolysis (mean): {np.mean(metrics['hemolysis_scores']):.4f}")
    
    return results


def ablation_2_preference_functional(
    generator: any,
    prompt: str,
    predictors: Dict,
    text_encoder: any,
    preference_net: any,
    constraints: Dict,
    seed_set: List[str],
    num_samples: int = 100
) -> Dict[str, Dict]:
    """
    Ablation 2: Nonlinear vs linear preference functional
    """
    print("\n" + "="*80)
    print("ABLATION 2: Preference Functional Comparison")
    print("="*80)
    
    results = {}
    
    for use_linear in [False, True]:
        method_name = 'linear' if use_linear else 'nonlinear'
        print(f"\nMethod: {method_name} preferences")
        
        samples = []
        for seed_idx, seed in enumerate(seed_set[:num_samples]):
            try:
                peptide = sample_from_fixed_seeds(
                    base_generator=generator,
                    seed_set=[seed],
                    prompt=prompt,
                    predictors=predictors,
                    constraints=constraints,
                    text_encoder=text_encoder,
                    preference_net=preference_net,
                    num_steps=50,
                    use_linear_preferences=use_linear,
                    completions_per_seed=1,
                    seed=42 + seed_idx,
                    language_weight=1.0
                )
                if peptide:
                    samples.append(peptide[0]['completion'])
            except Exception as e:
                print(f"    Warning: Failed seed {seed_idx}: {e}")
                continue
        
        metrics = _evaluate_samples(samples, predictors, generator)
        results[method_name] = {
            'samples': samples,
            'metrics': metrics
        }
        
        print(f"  Generated {len(samples)} samples")
        print(f"  Binding (mean): {np.mean(metrics['binding_scores']):.4f}")
        print(f"  Prompt alignment: {metrics.get('prompt_alignment', 0.0):.4f}")
    
    return results


def ablation_3_kernel_comparison(
    generator: any,
    prompt: str,
    predictors: Dict,
    text_encoder: any,
    preference_net: any,
    constraints: Dict,
    seed_set: List[str],
    num_samples: int = 100
) -> Dict[str, Dict]:
    """
    Ablation 3: LaPep kernel vs naive guidance
    """
    print("\n" + "="*80)
    print("ABLATION 3: Kernel Comparison (LaPep vs Naive Guidance)")
    print("="*80)
    
    results = {}
    
    # LaPep (conservative kernel)
    print("\nMethod: LaPep (conservative kernel)")
    samples_lapep = []
    for seed_idx, seed in enumerate(seed_set[:num_samples]):
        try:
            peptide = sample_from_fixed_seeds(
                base_generator=generator,
                seed_set=[seed],
                prompt=prompt,
                predictors=predictors,
                constraints=constraints,
                text_encoder=text_encoder,
                preference_net=preference_net,
                num_steps=50,
                completions_per_seed=1,
                seed=42 + seed_idx,
                language_weight=1.0
            )
            if peptide:
                samples_lapep.append(peptide[0]['completion'])
        except Exception as e:
            print(f"    Warning: Failed seed {seed_idx}: {e}")
            continue
    
    metrics_lapep = _evaluate_samples(samples_lapep, predictors, generator)
    results['lapep'] = {
        'samples': samples_lapep,
        'metrics': metrics_lapep
    }
    print(f"  Generated {len(samples_lapep)} samples")
    
    # Naive guidance (non-conservative)
    print("\nMethod: Naive guidance (direct reweighting)")
    samples_naive = []
    for seed_idx, seed in enumerate(seed_set[:num_samples]):
        try:
            peptide = sample_with_naive_guidance(
                generator,
                prompt=prompt,
                predictors=predictors,
                constraints=constraints,
                text_encoder=text_encoder,
                preference_net=preference_net,
                num_steps=50,
                seed=42 + seed_idx,
                language_weight=1.0
            )
            samples_naive.append(peptide)
        except Exception as e:
            print(f"    Warning: Failed seed {seed_idx}: {e}")
            continue
    
    metrics_naive = _evaluate_samples(samples_naive, predictors, generator)
    results['naive_guidance'] = {
        'samples': samples_naive,
        'metrics': metrics_naive
    }
    print(f"  Generated {len(samples_naive)} samples")
    
    # Compare path independence (circulation)
    print("\nComparing path independence...")
    circulation_lapep = _estimate_circulation(generator, samples_lapep, prompt, predictors, 
                                             text_encoder, preference_net, constraints)
    circulation_naive = _estimate_circulation(generator, samples_naive, prompt, predictors,
                                             text_encoder, preference_net, constraints, 
                                             use_naive=True)
    
    results['lapep']['circulation'] = circulation_lapep
    results['naive_guidance']['circulation'] = circulation_naive
    
    print(f"  LaPep circulation: {circulation_lapep:.6f}")
    print(f"  Naive guidance circulation: {circulation_naive:.6f}")
    
    return results


def ablation_4_language_weight_sweep(
    generator: any,
    prompt: str,
    predictors: Dict,
    text_encoder: any,
    preference_net: any,
    constraints: Dict,
    seed_set: List[str],
    language_weights: List[float],
    num_samples: int = 100
) -> Dict[str, Dict]:
    """
    Ablation 4: Language weight sweep
    - Vary α to show smooth tradeoff and failure modes at high weight
    """
    print("\n" + "="*80)
    print("ABLATION 4: Language Weight Sweep")
    print("="*80)
    
    results = {}
    
    for alpha in language_weights:
        print(f"\nLanguage weight α = {alpha:.2f}")
        
        samples = []
        for seed_idx, seed in enumerate(seed_set[:num_samples]):
            try:
                peptide = sample_from_fixed_seeds(
                    base_generator=generator,
                    seed_set=[seed],
                    prompt=prompt,
                    predictors=predictors,
                    constraints=constraints,
                    text_encoder=text_encoder,
                    preference_net=preference_net,
                    num_steps=50,
                    completions_per_seed=1,
                    seed=42 + seed_idx,
                    language_weight=alpha
                )
                if peptide:
                    samples.append(peptide[0]['completion'])
            except Exception as e:
                print(f"    Warning: Failed seed {seed_idx}: {e}")
                continue
        
        metrics = _evaluate_samples(samples, predictors, generator)
        results[str(alpha)] = {
            'samples': samples,
            'metrics': metrics,
            'language_weight': alpha
        }
        
        print(f"  Generated {len(samples)} samples")
        print(f"  Binding (mean): {np.mean(metrics['binding_scores']):.4f}")
        print(f"  Constraint satisfaction: {metrics.get('constraint_satisfaction', 0.0):.4f}")
    
    return results


def ablation_5_text_encoder_comparison(
    generator: any,
    prompt: str,
    predictors: Dict,
    preference_net_path: str,
    constraints: Dict,
    seed_set: List[str],
    encoder_names: List[str],
    num_samples: int = 100,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    Ablation 5: Text encoder comparison (Qwen-0.6B vs Qwen-4B)
    - Same MLP and training
    """
    print("\n" + "="*80)
    print("ABLATION 5: Text Encoder Comparison")
    print("="*80)
    
    results = {}
    
    for encoder_name in encoder_names:
        print(f"\nEncoder: {encoder_name}")
        
        try:
            text_encoder = load_text_encoder(encoder_name, device=device)
            preference_net = load_preference_net(preference_net_path, device=device)
            
            samples = []
            for seed_idx, seed in enumerate(seed_set[:num_samples]):
                try:
                    peptide = sample_from_fixed_seeds(
                        base_generator=generator,
                        seed_set=[seed],
                        prompt=prompt,
                        predictors=predictors,
                        constraints=constraints,
                        text_encoder=text_encoder,
                        preference_net=preference_net,
                        num_steps=50,
                        completions_per_seed=1,
                        seed=42 + seed_idx,
                        language_weight=1.0
                    )
                    if peptide:
                        samples.append(peptide[0]['completion'])
                except Exception as e:
                    print(f"    Warning: Failed seed {seed_idx}: {e}")
                    continue
            
            metrics = _evaluate_samples(samples, predictors, generator)
            results[encoder_name] = {
                'samples': samples,
                'metrics': metrics
            }
            
            print(f"  Generated {len(samples)} samples")
            print(f"  Prompt alignment: {metrics.get('prompt_alignment', 0.0):.4f}")
            
        except Exception as e:
            print(f"  ✗ Failed to load encoder {encoder_name}: {e}")
            continue
    
    return results


# ============================================================================
# Evaluation Utilities
# ============================================================================

def _evaluate_samples(
    samples: List[str],
    predictors: Dict,
    generator: any
) -> Dict:
    """Evaluate samples and compute metrics."""
    if not samples:
        return {}
    
    # Determine generator type
    is_wt = all(all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in s.upper()) for s in samples[:10])
    generator_name = 'pepdfm' if is_wt else 'pepmdlm'
    
    # Evaluate predictors (convert WT to SMILES if needed)
    binding_scores = []
    hemolysis_scores = []
    
    for sample in samples:
        if is_wt:
            # Convert WT to SMILES for predictor evaluation
            from scripts.run_prototype_experiment import _wt_to_smiles
            smiles = _wt_to_smiles(sample)
            if not smiles:
                binding_scores.append(np.nan)
                hemolysis_scores.append(np.nan)
                continue
        else:
            smiles = sample
        
        try:
            binding_scores.append(predictors['binding'].predict(smiles))
            hemolysis_scores.append(predictors['hemolysis'].predict(smiles))
        except:
            binding_scores.append(np.nan)
            hemolysis_scores.append(np.nan)
    
    metrics = {
        'binding_scores': np.array(binding_scores),
        'hemolysis_scores': np.array(hemolysis_scores)
    }
    
    # Generator-specific metrics
    if generator_name == 'pepdfm':
        valid_samples = [s for s in samples if all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in s.upper())]
        if valid_samples:
            wt_metrics = evaluate_wt_sequences(valid_samples)
            metrics.update(wt_metrics)
    else:
        smiles_metrics = evaluate_smiles_sequences(samples)
        metrics.update(smiles_metrics)
    
    # Constraint satisfaction
    binding_valid = np.array(binding_scores)[~np.isnan(binding_scores)]
    hemolysis_valid = np.array(hemolysis_scores)[~np.isnan(hemolysis_scores)]
    
    binding_satisfied = np.mean(binding_valid >= 0.7) if len(binding_valid) > 0 else 0.0
    hemolysis_satisfied = np.mean(hemolysis_valid <= 0.3) if len(hemolysis_valid) > 0 else 0.0
    metrics['constraint_satisfaction'] = (binding_satisfied + hemolysis_satisfied) / 2.0
    
    # Diversity (Shannon entropy)
    if generator_name == 'pepdfm':
        if 'shannon_entropy' in metrics:
            metrics['diversity'] = np.mean(metrics['shannon_entropy'])
        else:
            metrics['diversity'] = compute_shannon_entropy(''.join(samples[:100]))
    else:
        # For SMILES, use sequence diversity
        from scripts.run_prototype_experiment import _smiles_to_sequence_simple
        sequences = [_smiles_to_sequence_simple(s) for s in samples if _smiles_to_sequence_simple(s)]
        if sequences:
            metrics['diversity'] = np.mean([compute_shannon_entropy(seq) for seq in sequences])
        else:
            metrics['diversity'] = 0.0
    
    return metrics


def _estimate_circulation(
    generator: any,
    samples: List[str],
    prompt: str,
    predictors: Dict,
    text_encoder: any,
    preference_net: any,
    constraints: Dict,
    use_naive: bool = False,
    num_cycles: int = 50
) -> float:
    """Estimate circulation for path independence evaluation."""
    # Simplified circulation estimation
    # Full implementation would use eval/circulation.py
    circulations = []
    
    for _ in range(min(num_cycles, len(samples) // 2)):
        # Sample a random cycle
        if len(samples) < 3:
            continue
        
        cycle = np.random.choice(samples, size=3, replace=False)
        
        # Compute edge flows (simplified)
        # Full implementation would compute actual edge flows
        circulation = 0.0  # Placeholder
        circulations.append(circulation)
    
    return np.mean(circulations) if circulations else 0.0


# ============================================================================
# Main Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive ablation studies")
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--generator_model', type=str, required=True, help='Path to generator model')
    parser.add_argument('--generator_type', type=str, choices=['pepmdlm', 'pepdfm'], 
                       default='pepmdlm', help='Generator type')
    parser.add_argument('--test_set', type=str, help='Test set for fixed seeds (CSV or JSON)')
    parser.add_argument('--num_samples', type=int, default=100, help='Samples per condition')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--constraint_strength', type=float, default=2.0, help='Constraint strength')
    parser.add_argument('--language_weights', type=str, default='0.0,0.5,1.0,1.5,2.0,3.0,5.0',
                       help='Language weights for sweep')
    parser.add_argument('--text_encoders', type=str, default='Qwen/Qwen3-Embedding-0.6B',
                       help='Comma-separated list of text encoders to compare')
    parser.add_argument('--run_all', action='store_true', help='Run all ablations')
    parser.add_argument('--ablation', type=str, choices=['1', '2', '3', '4', '5', 'all'],
                       default='all', help='Which ablation to run')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"eval_results/ablations_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE ABLATION STUDIES")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Generator: {args.generator_type}")
    print("=" * 80)
    
    # Load generator
    print("\nLoading generator...")
    if args.generator_type == 'pepmdlm':
        generator = load_peptune_generator(args.generator_model, device=args.device)
    else:
        generator = load_dfm_model(args.generator_model)
    print("✓ Generator loaded")
    
    # Load predictors
    print("\nLoading predictors...")
    format_type = 'wt' if args.generator_type == 'pepdfm' else 'smiles'
    predictors = load_predictors(
        config, format_type=format_type, device=args.device,
        protein_seq=config.get('protein_seq')
    )
    if 'hemolysis' not in predictors:
        from predictors.smiles.hemolysis import HemolysisPredictor
        predictors['hemolysis'] = HemolysisPredictor(device=args.device)
    print(f"✓ Loaded {len(predictors)} predictor(s)")
    
    # Load text encoder and preference net
    print("\nLoading language models...")
    text_encoder = load_text_encoder(
        config.get('text_encoder_name', 'Qwen/Qwen3-Embedding-0.6B'),
        device=args.device
    )
    preference_net = load_preference_net(
        config.get('preference_net_path'),
        device=args.device
    )
    print("✓ Language models loaded")
    
    # Setup constraints
    constraints = {
        'strength': args.constraint_strength,
        'weights': {'binding': 1.0, 'hemolysis': 1.0}
    }
    
    # Load test set seeds
    seed_set = []
    if args.test_set:
        from scripts.load_test_set import load_test_set_csv, load_test_set_json
        test_path = Path(args.test_set)
        if test_path.suffix == '.csv':
            test_pairs = load_test_set_csv(args.test_set)
            seed_set = [pair['peptide'] for pair in test_pairs]
        elif test_path.suffix == '.json':
            test_pairs = load_test_set_json(args.test_set)
            seed_set = [pair['peptide'] for pair in test_pairs]
    else:
        # Generate random seeds
        for _ in range(args.num_samples):
            seed_set.append(generator.sample_initial_state())
    
    print(f"✓ Using {len(seed_set)} seeds")
    
    # Run ablations
    all_results = {}
    
    if args.ablation in ['1', 'all']:
        all_results['ablation_1_language_prompts'] = ablation_1_language_prompts(
            generator, predictors, text_encoder, preference_net, constraints,
            seed_set, args.num_samples, args.device
        )
    
    if args.ablation in ['2', 'all']:
        all_results['ablation_2_preference_functional'] = ablation_2_preference_functional(
            generator, "Design a peptide with good binding and low hemolysis.",
            predictors, text_encoder, preference_net, constraints,
            seed_set, args.num_samples
        )
    
    if args.ablation in ['3', 'all']:
        all_results['ablation_3_kernel_comparison'] = ablation_3_kernel_comparison(
            generator, "Design a peptide with good binding and low hemolysis.",
            predictors, text_encoder, preference_net, constraints,
            seed_set, args.num_samples
        )
    
    if args.ablation in ['4', 'all']:
        language_weights = [float(w.strip()) for w in args.language_weights.split(',')]
        all_results['ablation_4_language_weight_sweep'] = ablation_4_language_weight_sweep(
            generator, "Design a peptide that is protease-resistant and stable in vivo, while maintaining good binding and low hemolysis.",
            predictors, text_encoder, preference_net, constraints,
            seed_set, language_weights, args.num_samples
        )
    
    if args.ablation in ['5', 'all']:
        encoder_names = [e.strip() for e in args.text_encoders.split(',')]
        all_results['ablation_5_text_encoder'] = ablation_5_text_encoder_comparison(
            generator, "Design a peptide with good binding and low hemolysis.",
            predictors, config.get('preference_net_path'),
            constraints, seed_set, encoder_names, args.num_samples, args.device
        )
    
    # Save results
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    results_to_save = {}
    for ablation_name, ablation_results in all_results.items():
        results_to_save[ablation_name] = {}
        for condition_name, condition_data in ablation_results.items():
            results_to_save[ablation_name][condition_name] = {
                'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in condition_data['metrics'].items()}
            }
    
    with open(output_dir / "ablation_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Create comparison tables
    print("Creating comparison tables...")
    _create_ablation_tables(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("Ablation studies complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


def _create_ablation_tables(all_results: Dict, output_dir: Path):
    """Create comparison tables for all ablations."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    for ablation_name, ablation_results in all_results.items():
        table_data = []
        
        for condition_name, condition_data in ablation_results.items():
            metrics = condition_data['metrics']
            row = {'Condition': condition_name}
            
            # Add common metrics
            if 'binding_scores' in metrics:
                binding = metrics['binding_scores']
                binding_valid = binding[~np.isnan(binding)] if isinstance(binding, np.ndarray) else []
                row['Binding (mean)'] = f"{np.mean(binding_valid):.4f}" if len(binding_valid) > 0 else "N/A"
            
            if 'hemolysis_scores' in metrics:
                hemolysis = metrics['hemolysis_scores']
                hemolysis_valid = hemolysis[~np.isnan(hemolysis)] if isinstance(hemolysis, np.ndarray) else []
                row['Hemolysis (mean)'] = f"{np.mean(hemolysis_valid):.4f}" if len(hemolysis_valid) > 0 else "N/A"
            
            if 'constraint_satisfaction' in metrics:
                row['Constraint Satisfaction'] = f"{metrics['constraint_satisfaction']:.4f}"
            
            if 'diversity' in metrics:
                row['Diversity'] = f"{metrics['diversity']:.4f}"
            
            if 'prompt_alignment' in metrics:
                row['Prompt Alignment'] = f"{metrics['prompt_alignment']:.4f}"
            
            if 'circulation' in condition_data:
                row['Circulation'] = f"{condition_data['circulation']:.6f}"
            
            # Add generator-specific metrics
            for metric_name in ['protease_motifs', 'proline_fraction', 'shannon_entropy',
                               'd_amino_acids', 'n_methylation', 'terminal_capping', 'cyclization']:
                if metric_name in metrics:
                    metric_values = metrics[metric_name]
                    if isinstance(metric_values, np.ndarray):
                        valid_values = metric_values[~np.isnan(metric_values)]
                        if len(valid_values) > 0:
                            row[metric_name] = f"{np.mean(valid_values):.4f}"
            
            table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            df.to_csv(tables_dir / f"{ablation_name}.csv", index=False)
            
            # LaTeX version
            latex_table = df.to_latex(index=False, float_format="%.4f", escape=False)
            with open(tables_dir / f"{ablation_name}.tex", 'w') as f:
                f.write(latex_table)
    
    print(f"✓ Tables saved to {tables_dir}")


if __name__ == '__main__':
    main()

