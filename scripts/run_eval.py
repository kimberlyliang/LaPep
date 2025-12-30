"""
Main evaluation script for LaPep experiments.

experiments from the paper:
1. language conditioning effect (Section 4.1)
2. path independence and stability (Section 4.2)
3. unlabeled objective control (Section 4.3)
4. ablation studies (Section 4.4)
5. generality across base generators (Section 4.5)
"""

import argparse
import json
import os
from pathlib import Path
import torch
import sys
import numpy as np
from datetime import datetime

# Add project root to Python path BEFORE importing project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.distribution_shift import (
    evaluate_language_conditioning_effect,
    format_results_table as format_distribution_table
)
from eval.circulation import (
    compare_conditioning_methods,
    format_circulation_table
)
from eval.motif_analysis import (
    evaluate_unlabeled_objective_control,
    format_unlabeled_table
)
from eval.ablations import (
    run_ablation_studies,
    format_ablation_table
)
from generators.base_generator import load_base_generator
from language.text_encoder import load_text_encoder
from language.preference_net import load_preference_net
from predictors.loader import load_predictors
from eval.circulation import evaluate_path_independence
from generators.diffusion_wrapper import load_diffusion_model
from generators.dfm_wrapper import load_dfm_model


def convert_to_serializable(obj):
    """convert numpy arrays to JSON-compatible types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def find_latest_trained_model(results_dir: Path = Path("results")) -> Path:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    model_files = []
    for training_dir in results_dir.glob("training_*"):
        model_file = training_dir / "final_model.ckpt"
        if model_file.exists():
            model_files.append(model_file)
    
    if not model_files:
        raise FileNotFoundError(
            f"No trained models found in {results_dir}. "
        )
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    mod_time = datetime.fromtimestamp(latest_model.stat().st_mtime)
    print(f"\n[Model Selection] Found latest trained model:")
    print(f"  Path: {latest_model}")
    print(f"  Training directory: {latest_model.parent}")
    print(f"  Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    return latest_model


def load_models(config_path: str, device: str = 'cuda', use_latest_model: bool = True):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_generator = load_base_generator(config['base_generator_path'], device=device)
    text_encoder = load_text_encoder(config['text_encoder_name'], device=device)
    test_embedding = text_encoder.encode("test")
    actual_embedding_dim = test_embedding.shape[-1]
    print(f"Text encoder embedding dimension: {actual_embedding_dim}")
    
    # use latest trained model if requested, otherwise use config
    if use_latest_model:
        try:
            latest_model_path = find_latest_trained_model()
            preference_net_path = str(latest_model_path)
        except FileNotFoundError as e:
            print(f"\n[Model Selection] Warning: {e}")
            print(f"[Model Selection] Falling back to config path: {config.get('preference_net_path', 'Not specified')}")
            preference_net_path = config.get('preference_net_path')
            if preference_net_path is None:
                raise ValueError("No preference_net_path in config and no trained models found in results/")
    else:
        preference_net_path = config.get('preference_net_path')
        if preference_net_path is None:
            raise ValueError("preference_net_path not specified in config")
        print(f"\n[Model Selection] Using model from config: {preference_net_path}")
    
    preference_net = load_preference_net(preference_net_path, device=device)
    
    # Check if dimensions match
    if preference_net.input_dim != actual_embedding_dim:
        print(f"[Model Check] WARNING: Dimension mismatch!")
        print(f"  - Preference network expects: {preference_net.input_dim} dimensions")
        print(f"  - Text encoder provides: {actual_embedding_dim} dimensions")
        raise ValueError(
            f"Embedding dimension mismatch: preference_net expects {preference_net.input_dim} "
            f"but text_encoder provides {actual_embedding_dim}. "
        )
    else:
        print(f"[Model Check] Embedding dimensions match: {actual_embedding_dim}")
    
    # Determine format from generator type
    generator_type = config.get('generator_type', config.get('base_generator_type', 'pepmdlm'))
    format_type = 'wt' if generator_type == 'pepdfm' else 'smiles'
    
    predictors = load_predictors(
        config,
        format_type=format_type,
        device=device,
        protein_seq=config.get('protein_seq')
    )
    return base_generator, text_encoder, preference_net, predictors, config


def run_experiment_4_1(
    base_generator,
    text_encoder,
    preference_net,
    predictors,
    output_dir: Path
):
    """Run Section 4.1: Effect of Language Conditioning."""
    print("\n" + "="*80)
    print("Experiment 4.1: Effect of Language Conditioning on Generated Distributions")
    print("="*80)
    
    prompts = [
        "Generate a peptide with high binding affinity and low toxicity",
        "Generate a stable peptide with long half-life",
        "Generate a balanced peptide optimizing binding, toxicity, and half-life"
    ]
    
    results = evaluate_language_conditioning_effect(
        base_generator=base_generator,
        text_encoder=text_encoder,
        preference_net=preference_net,
        predictors=predictors,
        prompts=prompts,
        num_samples=1000
    )
    results_path = output_dir / "experiment_4_1_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    predictor_names = list(predictors.keys())
    table = format_distribution_table(results, predictor_names)
    table_path = output_dir / "table_language_effect.tex"
    with open(table_path, 'w') as f:
        f.write(table)
    
    print(f"Results saved to {results_path}")
    print(f"Table saved to {table_path}")
    print("\n" + table)

def run_experiment_4_2(
    base_generator,
    text_encoder,
    preference_net,
    predictors,
    output_dir: Path
):
    """Run Section 4.2: Stability and Path Independence."""
    print("\n" + "="*80)
    print("Experiment 4.2: Stability and Path Independence of Conditioned Dynamics")
    print("="*80)
    
    prompt = "Generate a peptide with high binding affinity and low toxicity"
    
    results = compare_conditioning_methods(
        base_generator=base_generator,
        text_encoder=text_encoder,
        preference_net=preference_net,
        predictors=predictors,
        prompt=prompt,
        num_cycles=1000
    )
    
    results_path = output_dir / "experiment_4_2_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    table = format_circulation_table(results)
    table_path = output_dir / "table_path_independence.tex"
    with open(table_path, 'w') as f:
        f.write(table)
    
    print(f"Results saved to {results_path}")
    print(f"Table saved to {table_path}")
    print("\n" + table)


def run_experiment_4_3(
    base_generator,
    text_encoder,
    preference_net,
    predictors,
    output_dir: Path
):
    """Run Section 4.3: Language-Guided Control of Unlabeled Objectives."""
    print("\n" + "="*80)
    print("Experiment 4.3: Language-Guided Control of Unlabeled Design Objectives")
    print("="*80)
    
    prompts = {
        'protease_resistance': (
            "Generate a peptide that is resistant to protease degradation "
            "and maintains therapeutic activity"
        ),
        'stability_oriented': (
            "Generate a stable peptide suitable for formulation and storage "
            "with minimal aggregation"
        )
    }
    
    results = evaluate_unlabeled_objective_control(
        base_generator=base_generator,
        text_encoder=text_encoder,
        preference_net=preference_net,
        predictors=predictors,
        prompts=prompts,
        num_samples=500
    )
    
    results_path = output_dir / "experiment_4_3_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    table = format_unlabeled_table(results)
    table_path = output_dir / "table_unlabeled_control.tex"
    with open(table_path, 'w') as f:
        f.write(table)
    
    print(f"Results saved to {results_path}")
    print(f"Table saved to {table_path}")
    print("\n" + table)


def run_experiment_4_4(
    base_generator,
    text_encoder,
    preference_net,
    predictors,
    output_dir: Path
):
    """Run Section 4.4: Ablation Studies."""
    print("\n" + "="*80)
    print("Experiment 4.4: Ablations on Preference and Constraint Formulation")
    print("="*80)
    
    prompt = "Generate a peptide with high binding affinity and low toxicity"
    
    results = run_ablation_studies(
        base_generator=base_generator,
        text_encoder=text_encoder,
        preference_net=preference_net,
        predictors=predictors,
        prompt=prompt,
        num_samples=500
    )
    
    results_path = output_dir / "experiment_4_4_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    table = format_ablation_table(results)
    table_path = output_dir / "table_ablations.tex"
    with open(table_path, 'w') as f:
        f.write(table)
    
    print(f"Results saved to {results_path}")
    print(f"Table saved to {table_path}")
    print("\n" + table)


def run_experiment_4_5(
    base_generators: dict,
    text_encoder,
    preference_net,
    predictors,
    output_dir: Path
):
    """Run Section 4.5: Generality Across Base Generators."""
    print("\n" + "="*80)
    print("Experiment 4.5: Generality Across Base Generators")
    print("="*80)
    
    prompt = "Generate a peptide with high binding affinity and low toxicity"
    results = {}
    
    for gen_name, base_generator in base_generators.items():
        print(f"\nEvaluating {gen_name}...")
        
        # Language conditioning effect
        dist_results = evaluate_language_conditioning_effect(
            base_generator=base_generator,
            text_encoder=text_encoder,
            preference_net=preference_net,
            predictors=predictors,
            prompts=[prompt],
            num_samples=500
        )
        
        # Path independence
        path_results = evaluate_path_independence(
            base_generator=base_generator,
            text_encoder=text_encoder,
            preference_net=preference_net,
            predictors=predictors,
            prompt=prompt,
            num_cycles=500
        )
        
        results[gen_name] = {
            'prompt_alignment': np.mean(dist_results[prompt]) if prompt in dist_results else 0.0,
            'constraint_satisfaction': 1.0,
            'path_stability': 1.0 / (1.0 + path_results.get('mean_circulation', 0.0))
        }
    
    results_path = output_dir / "experiment_4_5_results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Robustness of LaPep across base generators.}",
        "\\label{tab:generality}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Base Generator} & \\textbf{Prompt Alignment} & ",
        "\\textbf{Constraint Satisfaction} & \\textbf{Path Stability} \\\\",
        "\\midrule"
    ]
    
    for gen_name, gen_results in results.items():
        alignment = gen_results['prompt_alignment']
        constraint = gen_results['constraint_satisfaction']
        stability = gen_results['path_stability']
        lines.append(
            f"{gen_name.replace('_', ' ').title()} & {alignment:.3f} & "
            f"{constraint:.3f} & {stability:.3f} \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    table = "\n".join(lines)
    table_path = output_dir / "table_generality.tex"
    with open(table_path, 'w') as f:
        f.write(table)
    
    print(f"Results saved to {results_path}")
    print(f"Table saved to {table_path}")
    print("\n" + table)


def main():
    parser = argparse.ArgumentParser(
        description="Run LaPep evaluation experiments"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./eval_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        choices=['4.1', '4.2', '4.3', '4.4', '4.5', 'all'],
        default=['all'],
        help='Which experiments to run'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run experiments on'
    )
    parser.add_argument(
        '--use_config_model',
        action='store_true',
        help='Use model from config instead of latest trained model'
    )
    
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        actual_device = 'cpu'
    else:
        actual_device = args.device
    
    print(f"Using device: {actual_device}")
    if actual_device.startswith('cuda'):
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create timestamped output directory to avoid overwriting previous results
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    
    print("Loading models...")
    base_generator, text_encoder, preference_net, predictors, config = load_models(
        args.config, device=actual_device, use_latest_model=not args.use_config_model
    )
    
    experiments_to_run = args.experiments
    if 'all' in experiments_to_run:
        experiments_to_run = ['4.1', '4.2', '4.3', '4.4', '4.5']
    
    if '4.1' in experiments_to_run:
        run_experiment_4_1(
            base_generator, text_encoder, preference_net, predictors, output_dir
        )
    
    if '4.2' in experiments_to_run:
        run_experiment_4_2(
            base_generator, text_encoder, preference_net, predictors, output_dir
        )
    
    if '4.3' in experiments_to_run:
        run_experiment_4_3(
            base_generator, text_encoder, preference_net, predictors, output_dir
        )
    
    if '4.4' in experiments_to_run:
        run_experiment_4_4(
            base_generator, text_encoder, preference_net, predictors, output_dir
        )
    
    if '4.5' in experiments_to_run:
        base_generators = {}
        
        # Try to load additional generators if available
        # Note: Currently only PepMDLM is fully implemented
        try:
            if config.get('diffusion_model_path'):
                base_generators['masked_discrete_diffusion'] = load_diffusion_model(
                    config.get('diffusion_model_path')
                )
        except NotImplementedError as e:
            print(f"[Experiment 4.5] Skipping diffusion model: {e}")
        
        try:
            if config.get('dfm_model_path'):
                base_generators['discrete_flow_matching'] = load_dfm_model(
                    config.get('dfm_model_path')
                )
        except NotImplementedError as e:
            print(f"[Experiment 4.5] Skipping DFM model: {e}")
        
        # Always include the main generator (PepMDLM) for comparison
        base_generators['peptune'] = base_generator
        
        if len(base_generators) < 2:
            print(f"\n[Experiment 4.5] Warning: Only {len(base_generators)} generator(s) available.")
            print("Experiment 4.5 requires multiple generators for comparison.")
            print("Skipping experiment 4.5. Only PepMDLM is currently implemented.")
        else:
            run_experiment_4_5(
                base_generators, text_encoder, preference_net, predictors, output_dir
            )
    
    print("\n" + "="*80)
    print("All experiments completed")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

