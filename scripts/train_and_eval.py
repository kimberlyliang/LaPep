#!/usr/bin/env python3
"""
Combined training and evaluation script for LaPep.

This script:
1. Trains the preference network for 10 epochs (default, configurable)
2. Saves the final trained model to results/training_TIMESTAMP/final_model.ckpt
3. Immediately uses that final model for evaluation (not the one in config)
4. Runs all evaluation experiments (4.1-4.5) by default

Usage:
    python scripts/train_and_eval.py --config config.json
    
Note: HuggingFace model downloads may show progress bars that continue after
"loaded" messages - this is normal, downloads happen asynchronously.
"""

import argparse
import json
import torch
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import training functions
import importlib.util
train_spec = importlib.util.spec_from_file_location("train_preferences", Path(__file__).parent / "train_preferences.py")
train_module = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(train_module)

eval_spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).parent / "run_eval.py")
eval_module = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(eval_module)
from language.preference_net import PreferenceNet, load_preference_net
from generators.base_generator import load_base_generator
from predictors.binding import BindingPredictor
from predictors.toxicity import ToxicityPredictor
from predictors.halflife import HalfLifePredictor


def find_latest_model(results_dir: Path = Path("results")) -> Path:
    """
    Find the latest trained model in the results directory.
    
    Looks for final_model.ckpt files in training_* subdirectories,
    returns the most recently modified one.
    
    Args:
        results_dir: Base results directory (default: "results")
        
    Returns:
        Path to the latest model checkpoint
        
    Raises:
        FileNotFoundError: If no trained models are found
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all final_model.ckpt files in training_* directories
    model_files = []
    for training_dir in results_dir.glob("training_*"):
        model_file = training_dir / "final_model.ckpt"
        if model_file.exists():
            model_files.append(model_file)
    
    if not model_files:
        raise FileNotFoundError(
            f"No trained models found in {results_dir}. "
            f"Please train a model first or check the results directory."
        )
    
    # Return the most recently modified one
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Found latest model: {latest_model}")
    print(f"  Modified: {datetime.fromtimestamp(latest_model.stat().st_mtime)}")
    
    return latest_model


def load_models_with_custom_preference_net(
    config_path: str,
    preference_net_path: str,
    device: str = 'cuda'
):
    """
    Load models for evaluation, using a custom preference network path.
    
    This is similar to load_models from run_eval.py but allows specifying
    the preference network path instead of reading from config.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    from generators.base_generator import load_base_generator
    base_generator = load_base_generator(config['base_generator_path'], device=device)
    
    from language.text_encoder import load_text_encoder
    text_encoder = load_text_encoder(config['text_encoder_name'], device=device)
    
    # Check embedding dimension
    test_embedding = text_encoder.encode("test")
    actual_embedding_dim = test_embedding.shape[-1]
    print(f"[Model Check] Text encoder embedding dimension: {actual_embedding_dim}")
    
    # Load preference network from specified path
    from language.preference_net import load_preference_net
    preference_net = load_preference_net(preference_net_path, device=device)
    
    # Check if dimensions match
    if preference_net.input_dim != actual_embedding_dim:
        print(f"[Model Check] WARNING: Dimension mismatch!")
        print(f"  - Preference network expects: {preference_net.input_dim} dimensions")
        print(f"  - Text encoder provides: {actual_embedding_dim} dimensions")
        print(f"  - This will cause a RuntimeError during inference")
        raise ValueError(
            f"Embedding dimension mismatch: preference_net expects {preference_net.input_dim} "
            f"but text_encoder provides {actual_embedding_dim}."
        )
    else:
        print(f"[Model Check] ✓ Embedding dimensions match: {actual_embedding_dim}")
    
    predictors = {}
    for pred_name, pred_config in config['predictors'].items():
        if pred_name == 'binding':
            predictors['binding'] = BindingPredictor.load(
                pred_config['path'], 
                device=device,
                protein_seq=config.get('protein_seq')
            )
        elif pred_name == 'toxicity':
            predictors['toxicity'] = ToxicityPredictor.load(pred_config['path'])
        elif pred_name == 'halflife':
            predictors['halflife'] = HalfLifePredictor.load(pred_config['path'])
    
    return base_generator, text_encoder, preference_net, predictors, config


def main():
    parser = argparse.ArgumentParser(
        description="Train LaPep preference network and run evaluation"
    )
    
    # Training arguments
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training pairs')
    parser.add_argument('--peptides_per_batch', type=int, default=16, help='Peptides to sample per batch')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--use_linear', action='store_true', help='Use linear preferences')
    parser.add_argument('--judge_method', type=str, default='rule_based', 
                       choices=['rule_based', 'llm_judge'], 
                       help='Method for creating pairwise comparisons')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='L2 regularization weight')
    parser.add_argument('--training_output_dir', type=str, default=None, 
                       help='Output directory for training results (default: results/training_TIMESTAMP)')
    
    # Evaluation arguments
    parser.add_argument('--eval_output_dir', type=str, default=None,
                       help='Output directory for evaluation results (default: eval_results)')
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['4.1', '4.2', '4.3', '4.4', '4.5', 'all'],
                       default=['all'],
                       help='Which experiments to run (default: all)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only run evaluation with latest model')
    parser.add_argument('--use_specific_model', type=str, default=None,
                       help='Use a specific model path instead of latest (for evaluation only)')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        actual_device = 'cpu'
    else:
        actual_device = args.device
    
    print("=" * 80)
    print("LaPep: Train and Evaluate")
    print("=" * 80)
    print(f"Device: {actual_device}")
    if actual_device.startswith('cuda') and torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # ========================================================================
    # PART 1: TRAINING
    # ========================================================================
    if not args.skip_training:
        print("\n" + "=" * 80)
        print("PART 1: TRAINING")
        print("=" * 80)
        
        # Load models for training
        print("\n[Training] Loading models...")
        from language.text_encoder import load_text_encoder as train_load_text_encoder
        from generators.base_generator import load_base_generator as train_load_base_generator
        
        text_encoder = train_load_text_encoder(config['text_encoder_name'], device=actual_device)
        
        predictors = {}
        protein_seq = config.get('protein_seq', None)
        for pred_name, pred_config in config['predictors'].items():
            if pred_name == 'binding':
                predictors['binding'] = BindingPredictor.load(
                    pred_config['path'], protein_seq=protein_seq, device=actual_device
                )
            elif pred_name == 'toxicity':
                predictors['toxicity'] = ToxicityPredictor.load(pred_config['path'])
            elif pred_name == 'halflife':
                predictors['halflife'] = HalfLifePredictor.load(pred_config['path'])
        
        base_generator = train_load_base_generator(config['base_generator_path'], device=actual_device)
        
        # Get prompts
        prompts = config.get('training_prompts', [
            "Generate a peptide with high binding affinity and low toxicity",
            "Generate a stable peptide with long half-life",
            "Generate a balanced peptide optimizing binding, toxicity, and half-life"
        ])
        
        # Initialize preference network
        print("\n[Training] Initializing preference network...")
        test_embedding = text_encoder.encode("test")
        embedding_dim = test_embedding.shape[-1]
        
        hidden_dim = config.get('hidden_dim', 256)
        output_dim = config.get('output_dim', 64)
        num_predictors = len(predictors)
        
        preference_net = PreferenceNet(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_predictors=num_predictors
        )
        preference_net = preference_net.to(actual_device)
        
        # Optional LLM judge
        llm_judge = None
        if args.judge_method == 'llm_judge':
            try:
                from language.llm_judge import create_llm_judge
                llm_api_key = config.get('llm_api_key', None)
                llm_model = config.get('llm_model', 'gpt-4')
                llm_judge = create_llm_judge(model_name=llm_model, api_key=llm_api_key)
            except ImportError:
                print("Warning: LLM judge dependencies not available, falling back to rule-based")
                args.judge_method = 'rule_based'
        
        # Train
        print(f"\n[Training] Starting training for {args.epochs} epochs...")
        
        # Determine training output directory before training (so we know where model will be saved)
        if args.training_output_dir:
            training_output_dir = Path(args.training_output_dir)
        else:
            # Create a timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            training_output_dir = Path(f"results/training_{timestamp}")
        
        trained_net = train_module.train_preference_network(
            preference_net,
            text_encoder,
            base_generator,
            predictors,
            prompts,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=actual_device,
            use_linear=args.use_linear,
            reg_weight=args.reg_weight,
            peptides_per_batch=args.peptides_per_batch,
            judge_method=args.judge_method,
            llm_judge=llm_judge,
            output_dir=str(training_output_dir)
        )
        
        # train_preference_network returns the trained network but doesn't save final_model.ckpt
        # We need to save it ourselves
        final_model_path = training_output_dir / "final_model.ckpt"
        print(f"[Training] Saving final model to {final_model_path}...")
        torch.save({
            'state_dict': trained_net.state_dict(),
            'input_dim': trained_net.input_dim,
            'hidden_dim': trained_net.hidden_dim,
            'output_dim': trained_net.output_dim,
            'num_predictors': trained_net.num_predictors
        }, final_model_path)
        
        print(f"\n[Training] ✓ Training completed!")
        print(f"  Model saved to: {final_model_path}")
        print(f"  Training results: {training_output_dir}")
        
        # Use the newly trained model for evaluation
        preference_net_path = str(final_model_path)
        
    else:
        print("\n[Training] Skipping training (--skip_training flag set)")
        # Find latest model or use specified one
        if args.use_specific_model:
            preference_net_path = args.use_specific_model
            print(f"[Evaluation] Using specified model: {preference_net_path}")
        else:
            preference_net_path = str(find_latest_model())
            print(f"[Evaluation] Using latest model: {preference_net_path}")
    
    # ========================================================================
    # PART 2: EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 2: EVALUATION")
    print("=" * 80)
    
    # Set up evaluation output directory
    if args.eval_output_dir:
        eval_output_dir = Path(args.eval_output_dir)
    else:
        eval_output_dir = Path("eval_results")
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Evaluation] Loading models with trained preference network...")
    print(f"  Preference network: {preference_net_path}")
    
    # Load models for evaluation using the trained preference network
    base_generator, text_encoder, preference_net, predictors, config = \
        load_models_with_custom_preference_net(
            args.config,
            preference_net_path,
            device=actual_device
        )
    
    # Determine which experiments to run
    experiments_to_run = args.experiments
    if 'all' in experiments_to_run:
        experiments_to_run = ['4.1', '4.2', '4.3', '4.4', '4.5']
    
    print(f"\n[Evaluation] Running experiments: {experiments_to_run}")
    
    # Run experiments
    if '4.1' in experiments_to_run:
        print("\n[Evaluation] Running Experiment 4.1: Language Conditioning Effect...")
        eval_module.run_experiment_4_1(
            base_generator, text_encoder, preference_net, predictors, eval_output_dir
        )
    
    if '4.2' in experiments_to_run:
        print("\n[Evaluation] Running Experiment 4.2: Path Independence...")
        eval_module.run_experiment_4_2(
            base_generator, text_encoder, preference_net, predictors, eval_output_dir
        )
    
    if '4.3' in experiments_to_run:
        print("\n[Evaluation] Running Experiment 4.3: Unlabeled Objective Control...")
        eval_module.run_experiment_4_3(
            base_generator, text_encoder, preference_net, predictors, eval_output_dir
        )
    
    if '4.4' in experiments_to_run:
        print("\n[Evaluation] Running Experiment 4.4: Ablation Studies...")
        eval_module.run_experiment_4_4(
            base_generator, text_encoder, preference_net, predictors, eval_output_dir
        )
    
    if '4.5' in experiments_to_run:
        print("\n[Evaluation] Running Experiment 4.5: Generality Across Base Generators...")
        from generators.diffusion_wrapper import load_diffusion_model
        from generators.dfm_wrapper import load_dfm_model
        
        base_generators = {
            'masked_discrete_diffusion': load_diffusion_model(config.get('diffusion_model_path')),
            'discrete_flow_matching': load_dfm_model(config.get('dfm_model_path'))
        }
        
        eval_module.run_experiment_4_5(
            base_generators, text_encoder, preference_net, predictors, eval_output_dir
        )
    
    print("\n" + "=" * 80)
    print("✓ All experiments completed!")
    print(f"  Evaluation results saved to: {eval_output_dir}")
    print(f"  Trained model: {preference_net_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()

