#!/usr/bin/env python3
"""
Validate and fix config.json file.

Checks for common issues and suggests fixes.
"""

import json
import sys
from pathlib import Path

def validate_config(config_path: str = "config.json"):
    """Validate config.json and suggest fixes."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"✗ {config_path} not found")
        print(f"  → Create from template: cp config_template.json config.json")
        return False
    
    print(f"Validating {config_path}...")
    print("=" * 70)
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False
    
    issues = []
    warnings = []
    
    # Check required fields
    required_fields = {
        'base_generator_path': 'PepMDLM generator path',
        'text_encoder_name': 'Text encoder model name',
        'dfm_model_path': 'PepDFM generator path',
    }
    
    for field, description in required_fields.items():
        if field not in config:
            issues.append(f"Missing required field: {field} ({description})")
        else:
            print(f"✓ {field}: {config[field]}")
    
    # Check generator_type
    if 'generator_type' not in config:
        warnings.append("Missing 'generator_type' (will default to 'pepmdlm')")
        print("⚠ generator_type: Not set (will default to 'pepmdlm')")
    else:
        gen_type = config['generator_type']
        if gen_type not in ['pepmdlm', 'pepdfm']:
            issues.append(f"Invalid generator_type: {gen_type} (must be 'pepmdlm' or 'pepdfm')")
        else:
            print(f"✓ generator_type: {gen_type}")
    
    # Check paths exist
    path_fields = {
        'base_generator_path': 'PepMDLM checkpoint',
        'dfm_model_path': 'PepDFM checkpoint',
        'preference_net_path': 'Preference network (optional, will be created if missing)',
    }
    
    repo_root = Path(__file__).parent.parent
    
    for field, description in path_fields.items():
        if field in config:
            path = Path(config[field])
            if not path.is_absolute():
                path = repo_root / path
            
            if path.exists():
                size_mb = path.stat().st_size / 1e6
                print(f"✓ {field}: {path} ({size_mb:.2f} MB)")
            else:
                if field == 'preference_net_path':
                    warnings.append(f"{field}: {config[field]} not found (will be created during training)")
                    print(f"⚠ {field}: {config[field]} (will be created)")
                else:
                    issues.append(f"{field}: {config[field]} not found")
                    print(f"✗ {field}: {config[field]} NOT FOUND")
    
    # Check dfm_model_path specifically
    if 'dfm_model_path' in config:
        dfm_path = config['dfm_model_path']
        if 'dfm_model.ckpt' in dfm_path or 'mog-dfm' in dfm_path.lower():
            warnings.append(f"dfm_model_path might be wrong: {dfm_path} (should be 'pretrained/generators/pepdfm.ckpt')")
            print(f"⚠ dfm_model_path: {dfm_path} (should this be 'pretrained/generators/pepdfm.ckpt'?)")
    
    # Check predictors
    if 'predictors' in config:
        print(f"\n✓ predictors: {len(config['predictors'])} predictor(s) configured")
        for pred_name, pred_config in config['predictors'].items():
            if 'path' in pred_config:
                pred_path = Path(pred_config['path'])
                if not pred_path.is_absolute():
                    pred_path = repo_root / pred_path
                if pred_path.exists():
                    print(f"  ✓ {pred_name}: {pred_config['path']}")
                else:
                    warnings.append(f"Predictor {pred_name}: {pred_config['path']} not found (will use fallback)")
                    print(f"  ⚠ {pred_name}: {pred_config['path']} (will use fallback)")
    else:
        warnings.append("No predictors configured")
    
    # Check constraints
    if 'constraints' in config:
        if 'weights' in config['constraints']:
            weights = config['constraints']['weights']
            print(f"✓ constraints.weights: {list(weights.keys())}")
        if 'strength' in config['constraints']:
            print(f"✓ constraints.strength: {config['constraints']['strength']}")
    
    # Check training_prompts
    if 'training_prompts' in config:
        print(f"✓ training_prompts: {len(config['training_prompts'])} prompt(s)")
    else:
        warnings.append("No training_prompts configured")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("-" * 70)
    
    if issues:
        print(f"✗ {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print(f"⚠ {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("✓ Config is valid!")
        return True
    elif not issues:
        print("✓ Config is valid (some warnings above)")
        return True
    else:
        print("\n✗ Config has issues that need to be fixed")
        return False

def fix_config(config_path: str = "config.json"):
    """Attempt to fix common config issues."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"Config file not found: {config_path}")
        return False
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    fixes_applied = []
    
    # Fix dfm_model_path
    if 'dfm_model_path' in config:
        if config['dfm_model_path'] == 'pretrained/generators/dfm_model.ckpt':
            config['dfm_model_path'] = 'pretrained/generators/pepdfm.ckpt'
            fixes_applied.append("Fixed dfm_model_path")
    
    # Add generator_type if missing
    if 'generator_type' not in config:
        config['generator_type'] = 'pepmdlm'
        fixes_applied.append("Added generator_type: 'pepmdlm'")
    
    # Update preference_net_path if it's an old training result
    if 'preference_net_path' in config:
        old_path = config['preference_net_path']
        if 'results/training' in old_path or 'training_' in old_path:
            config['preference_net_path'] = 'pretrained/language/preference_net.ckpt'
            fixes_applied.append("Updated preference_net_path to standard location")
    
    if fixes_applied:
        # Backup original
        backup_path = config_file.with_suffix('.json.backup')
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Backed up original to: {backup_path}")
        
        # Write fixed config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nApplied {len(fixes_applied)} fix(es):")
        for fix in fixes_applied:
            print(f"  ✓ {fix}")
        return True
    else:
        print("No fixes needed")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate and fix config.json")
    parser.add_argument('--config', type=str, default='config.json', help='Config file to validate')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix common issues')
    
    args = parser.parse_args()
    
    if args.fix:
        print("Attempting to fix config...")
        print("=" * 70)
        fix_config(args.config)
        print()
    
    is_valid = validate_config(args.config)
    sys.exit(0 if is_valid else 1)

if __name__ == '__main__':
    main()

