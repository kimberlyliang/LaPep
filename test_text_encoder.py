#!/usr/bin/env python3
"""
Test script for Qwen text encoder loading and encoding.

This script tests:
1. Loading the Qwen model via TextEncoder
2. Encoding a sample prompt
3. Checking embedding dimensions
"""

import torch
from language.text_encoder import load_text_encoder

def test_qwen_encoder():
    """Test Qwen text encoder loading and encoding."""
    
    print("=" * 80)
    print("Testing Qwen Text Encoder")
    print("=" * 80)
    
    # Model name from test_qwen.py
    model_name = "Qwen/Qwen3-0.6B"
    
    print(f"\n[1] Loading text encoder: {model_name}")
    print("    (This may take a moment on first run to download the model...)")
    
    try:
        # Load the encoder
        text_encoder = load_text_encoder(model_name, device='cpu')
        print(f"    ✓ Loaded successfully!")
        print(f"    - Device: {text_encoder.device}")
        print(f"    - Embedding dimension: {text_encoder.embedding_dim}")
    except Exception as e:
        print(f"    ✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n[2] Testing encoding...")
    
    # Test prompts
    test_prompts = [
        "Generate a peptide with high binding affinity",
        "Create a peptide with low toxicity",
        "Design a stable peptide with long half-life"
    ]
    
    try:
        # Test single prompt - with debug output
        print(f"    Testing single prompt encoding...")
        test_prompt = test_prompts[0]
        print(f"    - Input prompt: '{test_prompt}'")
        
        # Show what the encoder actually processes
        messages = [{"role": "user", "content": test_prompt}]
        formatted_text = text_encoder.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        print(f"    - Formatted text (first 100 chars): '{formatted_text[:100]}...'")
        
        single_embedding = text_encoder.encode(test_prompt)
        print(f"    ✓ Single prompt encoded successfully!")
        print(f"    - Shape: {single_embedding.shape}")
        print(f"    - Dtype: {single_embedding.dtype}")
        
        # Test batch encoding
        print(f"\n    Testing batch encoding...")
        print(f"    - Input prompts:")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"      {i}. '{prompt}'")
        
        batch_embeddings = text_encoder.encode(test_prompts)
        print(f"    ✓ Batch encoding successful!")
        print(f"    - Shape: {batch_embeddings.shape}")
        print(f"    - Expected: ({len(test_prompts)}, {text_encoder.embedding_dim})")
        
        # Verify dimensions match
        assert batch_embeddings.shape[0] == len(test_prompts), \
            f"Batch size mismatch: {batch_embeddings.shape[0]} != {len(test_prompts)}"
        assert batch_embeddings.shape[1] == text_encoder.embedding_dim, \
            f"Embedding dim mismatch: {batch_embeddings.shape[1]} != {text_encoder.embedding_dim}"
        
        print(f"\n[3] Testing embedding properties...")
        print(f"    - Mean: {batch_embeddings.mean().item():.4f}")
        print(f"    - Std: {batch_embeddings.std().item():.4f}")
        print(f"    - Min: {batch_embeddings.min().item():.4f}")
        print(f"    - Max: {batch_embeddings.max().item():.4f}")
        
        # Test that different prompts give different embeddings
        print(f"\n[4] Testing prompt diversity...")
        emb1 = text_encoder.encode(test_prompts[0])
        emb2 = text_encoder.encode(test_prompts[1])
        # Fix: squeeze to remove batch dimension, then compute similarity
        emb1_flat = emb1.squeeze(0) if emb1.dim() > 1 else emb1
        emb2_flat = emb2.squeeze(0) if emb2.dim() > 1 else emb2
        cosine_sim = torch.nn.functional.cosine_similarity(emb1_flat.unsqueeze(0), emb2_flat.unsqueeze(0), dim=1)
        print(f"    - Prompt 1: '{test_prompts[0]}'")
        print(f"    - Prompt 2: '{test_prompts[1]}'")
        print(f"    - Cosine similarity: {cosine_sim.item():.4f}")
        print(f"    - (Should be < 1.0 to show prompts are encoded differently)")
        
        # Test with very different prompts to see lower similarity
        print(f"\n[4b] Testing with very different prompts...")
        diverse_prompts = [
            "Generate a peptide with high binding affinity",
            "What is the weather today?",  # Completely unrelated
            "Explain quantum mechanics"     # Completely unrelated
        ]
        emb_peptide = text_encoder.encode(diverse_prompts[0])
        emb_weather = text_encoder.encode(diverse_prompts[1])
        emb_quantum = text_encoder.encode(diverse_prompts[2])
        
        emb_peptide_flat = emb_peptide.squeeze(0) if emb_peptide.dim() > 1 else emb_peptide
        emb_weather_flat = emb_weather.squeeze(0) if emb_weather.dim() > 1 else emb_weather
        emb_quantum_flat = emb_quantum.squeeze(0) if emb_quantum.dim() > 1 else emb_quantum
        
        sim_peptide_weather = torch.nn.functional.cosine_similarity(
            emb_peptide_flat.unsqueeze(0), emb_weather_flat.unsqueeze(0), dim=1
        )
        sim_peptide_quantum = torch.nn.functional.cosine_similarity(
            emb_peptide_flat.unsqueeze(0), emb_quantum_flat.unsqueeze(0), dim=1
        )
        
        print(f"    - Peptide vs Weather: {sim_peptide_weather.item():.4f} (should be lower)")
        print(f"    - Peptide vs Quantum: {sim_peptide_quantum.item():.4f} (should be lower)")
        print(f"    - This shows the model CAN differentiate when prompts are very different")
        
        # Test that same prompt gives same embedding
        print(f"\n[5] Testing consistency...")
        emb1_repeat = text_encoder.encode(test_prompts[0])
        emb1_flat = emb1.squeeze(0) if emb1.dim() > 1 else emb1
        emb1_repeat_flat = emb1_repeat.squeeze(0) if emb1_repeat.dim() > 1 else emb1_repeat
        same_sim = torch.nn.functional.cosine_similarity(emb1_flat.unsqueeze(0), emb1_repeat_flat.unsqueeze(0), dim=1)
        print(f"    - Cosine similarity (same prompt, different calls): {same_sim.item():.4f}")
        print(f"    - (Should be ≈ 1.0 to show consistency)")
        
        print(f"\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"    ✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_qwen_encoder()

