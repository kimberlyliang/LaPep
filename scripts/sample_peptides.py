import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.base_generator import load_base_generator
from language.text_encoder import load_text_encoder
from language.preference_net import load_preference_net
from predictors.binding import BindingPredictor
from predictors.toxicity import ToxicityPredictor
from predictors.halflife import HalfLifePredictor
from lapep.sampler import sample_peptide


def main():
    parser = argparse.ArgumentParser(
        description="Sample peptides using LaPep"
    )
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--prompt', type=str, required=True, help='Natural language prompt')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--output', type=str, default='peptides.txt', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("Loading models...")
    base_generator = load_base_generator(config['base_generator_path'])
    text_encoder = load_text_encoder(config['text_encoder_name'])
    preference_net = load_preference_net(config['preference_net_path'])
    
    predictors = {}
    for pred_name, pred_config in config['predictors'].items():
        if pred_name == 'binding':
            predictors['binding'] = BindingPredictor.load(pred_config['path'])
        elif pred_name == 'toxicity':
            predictors['toxicity'] = ToxicityPredictor.load(pred_config['path'])
        elif pred_name == 'halflife':
            predictors['halflife'] = HalfLifePredictor.load(pred_config['path'])
    
    constraints = config.get('constraints', {})
    
    print(f"Generating {args.num_samples} peptides...")
    peptides = []
    for i in range(args.num_samples):
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{args.num_samples}...")
        
        peptide = sample_peptide(
            base_generator=base_generator,
            prompt=args.prompt,
            predictors=predictors,
            constraints=constraints,
            text_encoder=text_encoder,
            preference_net=preference_net,
            seed=args.seed + i
        )
        peptides.append(peptide)
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for peptide in peptides:
            f.write(peptide + '\n')
    
    print(f"Saved {len(peptides)} peptides to {output_path}")


if __name__ == '__main__':
    main()

