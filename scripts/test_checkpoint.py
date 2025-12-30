"""
Script to test a specific checkpoint from training.

This allows you to evaluate a checkpoint (e.g., epoch 20) to see if it performs
better than the final model.
"""

import argparse
import json
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_eval import load_models, run_experiment_4_1
from language.preference_net import PreferenceNet


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load a checkpoint and return the preference network."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model parameters
    input_dim = checkpoint.get('input_dim')
    hidden_dim = checkpoint.get('hidden_dim')
    output_dim = checkpoint.get('output_dim')
    num_predictors = checkpoint.get('num_predictors')
    
    if not all([input_dim, hidden_dim, output_dim, num_predictors]):
        raise ValueError(f"Checkpoint missing required parameters. Found: {checkpoint.keys()}")
    
    # Create preference network
    preference_net = PreferenceNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_predictors=num_predictors
    )
    
    # Load state dict
    preference_net.load_state_dict(checkpoint['state_dict'])
    preference_net = preference_net.to(device)
    preference_net.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
    
    print(f"Loaded checkpoint:")
    print(f"  - Epoch: {epoch}")
    print(f"  - Loss: {loss}")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Output dim: {output_dim}")
    print(f"  - Num predictors: {num_predictors}")
    
    return preference_net


def main():
    parser = argparse.ArgumentParser(
        description="Test a specific checkpoint from training"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (e.g., results/training_XXX/checkpoints/checkpoint_epoch_20.ckpt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.json file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for evaluation results (default: eval_results/checkpoint_test_TIMESTAMP)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='4.1',
        choices=['4.1', '4.2', '4.3', '4.4'],
        help='Which experiment to run (default: 4.1)'
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Determine output directory
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = checkpoint_path.stem
        output_dir = Path(f"eval_results/checkpoint_test_{checkpoint_name}_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TESTING CHECKPOINT")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    preference_net = load_checkpoint(str(checkpoint_path), device=args.device)
    
    # Load other models (base generator, text encoder, predictors)
    print("\nLoading other models...")
    # We need to load models but skip preference_net loading
    # Let's load config and models manually
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    from generators.peptune_wrapper import load_peptune_generator
    from generators.dfm_wrapper import load_dfm_model
    from language.text_encoder import load_text_encoder
    from predictors.binding import BindingPredictor
    from predictors.toxicity import ToxicityPredictor
    from predictors.halflife import HalfLifePredictor
    
    generator_type = config.get('generator_type', config.get('base_generator_type', 'pepmdlm'))
    
    if generator_type == 'pepdfm':
        base_generator = load_dfm_model(
            config.get('dfm_model_path'),
            device=args.device
        )
        if base_generator is None:
            raise RuntimeError(f"Failed to load PepDFM model from {config.get('dfm_model_path')}")
    else:
        base_generator = load_peptune_generator(
            config['base_generator_path'],
            device=args.device
        )
        if base_generator.model is None:
            raise RuntimeError(f"Failed to load PepMDLM model from {config['base_generator_path']}")
    text_encoder = load_text_encoder(config['text_encoder_name'], device=args.device)
    
    # Check embedding dimension matches checkpoint
    test_embedding = text_encoder.encode("test")
    actual_embedding_dim = test_embedding.shape[-1]
    print(f"Text encoder embedding dimension: {actual_embedding_dim}")
    
    if preference_net.input_dim != actual_embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: checkpoint expects {preference_net.input_dim} "
            f"but text encoder provides {actual_embedding_dim}"
        )
    print(f"✓ Embedding dimensions match: {actual_embedding_dim}")
    
    predictors = {}
    for pred_name, pred_config in config['predictors'].items():
        if pred_name == 'binding':
            predictors['binding'] = BindingPredictor.load(
                pred_config['path'], 
                device=args.device,
                protein_seq=config.get('protein_seq')
            )
        elif pred_name == 'toxicity':
            predictors['toxicity'] = ToxicityPredictor.load(pred_config['path'], device=args.device)
        elif pred_name == 'halflife':
            predictors['halflife'] = HalfLifePredictor.load(pred_config['path'])
    
    print("\nUsing checkpoint preference network for evaluation...")
    
    # Run experiment
    print(f"\nRunning Experiment {args.experiment}...")
    if args.experiment == '4.1':
        run_experiment_4_1(
            base_generator, text_encoder, preference_net, predictors, output_dir
        )
    elif args.experiment == '4.2':
        from scripts.run_eval import run_experiment_4_2
        run_experiment_4_2(
            base_generator, text_encoder, preference_net, predictors, output_dir
        )
    elif args.experiment == '4.3':
        from scripts.run_eval import run_experiment_4_3
        run_experiment_4_3(
            base_generator, text_encoder, preference_net, predictors, output_dir
        )
    elif args.experiment == '4.4':
        from scripts.run_eval import run_experiment_4_4
        run_experiment_4_4(
            base_generator, text_encoder, preference_net, predictors, output_dir
        )
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

