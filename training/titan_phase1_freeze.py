import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import time
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "../core")))

from titan_config import SOURCE_MODEL, DEVICE, DTYPE, print_config

def load_and_freeze():
    print("Loading model and freezing weights...")
    
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL,
        torch_dtype=DTYPE,
        device_map=DEVICE
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    
    attn_params = 0
    mlp_params = 0
    embed_params = 0
    norm_params = 0
    other_params = 0
    
    attn_details = {}
    mlp_details = {}
    
    for name, p in model.named_parameters():
        numel = p.numel()
        if 'self_attn' in name:
            attn_params += numel
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    attn_details[layer_idx] = attn_details.get(layer_idx, 0) + numel
                    break
        elif 'mlp' in name:
            mlp_params += numel
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    mlp_details[layer_idx] = mlp_details.get(layer_idx, 0) + numel
                    break
        elif 'embed' in name or 'lm_head' in name:
            embed_params += numel
        elif 'norm' in name or 'layernorm' in name:
            norm_params += numel
        else:
            other_params += numel
            
    total_params = attn_params + mlp_params + embed_params + norm_params + other_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Attention parameters: {attn_params:,}")
    print(f"MLP parameters: {mlp_params:,}")
    
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert trainable == 0, f"Error: {trainable} parameters are still trainable."
    
    # Forward Pass Verification
    test_prompt = "If A > B and B > C, then is A > C?"
    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        gen_ids = model.generate(
            **inputs, max_new_tokens=30, temperature=0.1,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"Test Q: {test_prompt}")
    print(f"Test A: {response}")
    
    # Capture original MLP norms for calibration
    num_layers = model.config.num_hidden_layers
    original_mlp_norms = {}
    hooks = []
    
    def make_norm_hook(layer_idx):
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            original_mlp_norms[layer_idx] = o.float().norm(dim=-1).mean().item()
        return hook_fn
        
    for i in range(num_layers):
        h = model.model.layers[i].mlp.register_forward_hook(make_norm_hook(i))
        hooks.append(h)
        
    calibration_prompts = [
        "Solve: 3x + 7 = 22",
        "If P implies Q and Q is false, what can we say about P?",
        "What is 15 * 8 + 12?",
        "All cats are animals. Tom is a cat. Is Tom an animal?",
        "Write a function that checks if a number is prime.",
    ]
    
    all_layer_norms = {i: [] for i in range(num_layers)}
    
    for prompt in calibration_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            model(**inputs)
        for layer_idx, norm_val in original_mlp_norms.items():
            all_layer_norms[layer_idx].append(norm_val)
            
    for h in hooks:
        h.remove()
        
    avg_mlp_norms = {}
    for layer_idx in range(num_layers):
        norms = all_layer_norms[layer_idx]
        avg_mlp_norms[layer_idx] = sum(norms) / len(norms) if norms else 0.0
        
    breakdown = {
        "attn_params": attn_params,
        "mlp_params": mlp_params,
        "embed_params": embed_params,
        "norm_params": norm_params,
        "total_params": total_params,
        "avg_mlp_norms": avg_mlp_norms,
    }
    
    return model, tokenizer, breakdown

if __name__ == "__main__":
    print_config()
    model, tokenizer, breakdown = load_and_freeze()
