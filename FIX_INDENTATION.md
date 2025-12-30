# Fix Indentation Error in scripts/run_eval.py

## Problem

After removing bad import lines with `sed`, there's an indentation error at line 137. This happens when `sed` removes a line that was part of a code block, leaving mismatched indentation.

## Solution

The issue is likely in the `load_models()` function around lines 115-137. Here's what it should look like:

```python
def load_models(config_path: str, device: str = 'cuda', use_latest_model: bool = True):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine generator type and load appropriate generator
    generator_type = config.get('generator_type', config.get('base_generator_type', 'pepmdlm'))
    
    if generator_type == 'pepdfm':
        base_generator = load_dfm_model(
            config.get('dfm_model_path'),
            device=device
        )
        if base_generator is None:
            raise RuntimeError(f"Failed to load PepDFM model from {config.get('dfm_model_path')}")
    else:
        base_generator = load_peptune_generator(
            config['base_generator_path'],
            device=device
        )
        if base_generator.model is None:
            raise RuntimeError(f"Failed to load PepMDLM model from {config['base_generator_path']}")
    
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
```

## Quick Fix on GPU Server

The easiest fix is to check what's around line 137 and fix the indentation:

```bash
# On GPU server, check lines around 137
sed -n '130,145p' scripts/run_eval.py

# If you see a closing paren `)` that's misaligned, fix it
# The issue is likely that a line was removed, leaving orphaned closing parens
```

## Better Solution: Pull from Git

If you've pushed your fixed local version:

```bash
# On GPU server
cd /scratch/pranamlab/kimberly/LaPep
git checkout scripts/run_eval.py  # Restore from git
git pull origin main  # Pull latest
```

Or copy the fixed file from local to GPU server.

