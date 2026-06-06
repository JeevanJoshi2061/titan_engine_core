import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import time
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "../core")))

from titan_config import (
    SOURCE_MODEL, OUTPUT_DIR, DEVICE, DTYPE,
    ORIGINAL_HIDDEN_DIM, ORIGINAL_INTERMEDIATE, NEW_INTERMEDIATE,
    CALIBRATION_SAMPLES, INIT_SCALE_FACTOR,
    print_config
)
from titan_phase1_freeze import load_and_freeze

class TitanLogicMLP(nn.Module):
    """
    Downsized SwiGLU MLP for logic computation.
    """
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def replace_all_mlps(model, tokenizer, avg_mlp_norms):
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    
    print(f"Original MLP intermediate: {model.config.intermediate_size}")
    print(f"New MLP intermediate:      {NEW_INTERMEDIATE}")
    
    original_mlp_per_layer = 3 * hidden_dim * model.config.intermediate_size
    new_mlp_per_layer = 3 * hidden_dim * NEW_INTERMEDIATE
    total_saved = num_layers * (original_mlp_per_layer - new_mlp_per_layer)
    print(f"Total params saved: {total_saved:,}")
    
    surgery_log = []
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        
        old_gate_shape = layer.mlp.gate_proj.weight.shape
        with torch.no_grad():
            old_gate_fnorm = layer.mlp.gate_proj.weight.data.float().norm(p='fro').item()
            old_up_fnorm = layer.mlp.up_proj.weight.data.float().norm(p='fro').item()
            old_down_fnorm = layer.mlp.down_proj.weight.data.float().norm(p='fro').item()
        
        new_mlp = TitanLogicMLP(hidden_dim, NEW_INTERMEDIATE)
        
        nn.init.kaiming_normal_(new_mlp.gate_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(new_mlp.up_proj.weight, nonlinearity='linear')
        nn.init.xavier_normal_(new_mlp.down_proj.weight)
        
        new_mlp = new_mlp.to(device=DEVICE, dtype=DTYPE)
        layer.mlp = new_mlp
        
        for param in layer.mlp.parameters():
            param.requires_grad = True
        
        new_gate_shape = layer.mlp.gate_proj.weight.shape
        new_params = sum(p.numel() for p in layer.mlp.parameters())
        
        surgery_log.append({
            "layer": layer_idx,
            "old_gate_shape": list(old_gate_shape),
            "new_gate_shape": list(new_gate_shape),
            "old_gate_fnorm": round(old_gate_fnorm, 4),
            "old_up_fnorm": round(old_up_fnorm, 4),
            "old_down_fnorm": round(old_down_fnorm, 4),
            "new_params": new_params,
        })
        
    gc.collect()
    torch.cuda.empty_cache()
    
    # Variance Calibration
    new_mlp_norms = {}
    hooks = []
    
    def make_norm_hook(layer_idx):
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            new_mlp_norms[layer_idx] = o.float().norm(dim=-1).mean().item()
        return hook_fn
    
    for i in range(num_layers):
        h = model.model.layers[i].mlp.register_forward_hook(make_norm_hook(i))
        hooks.append(h)
    
    cal_prompt = "If A > B and B > C, is A > C? Solve step by step."
    messages = [{"role": "user", "content": cal_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        model(**inputs)
    
    for h in hooks:
        h.remove()
        
    scale_factors = {}
    for layer_idx in range(num_layers):
        orig_norm = avg_mlp_norms.get(layer_idx, 1.0)
        new_norm = new_mlp_norms.get(layer_idx, 1.0)
        
        scale = (orig_norm / new_norm) * INIT_SCALE_FACTOR if new_norm > 1e-8 else 1.0
        scale_factors[layer_idx] = scale
        
        layer = model.model.layers[layer_idx]
        with torch.no_grad():
            layer.mlp.down_proj.weight.data.mul_(scale)
            
    # Verification
    calibrated_norms = {}
    hooks2 = []
    
    def make_cal_hook(layer_idx):
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            calibrated_norms[layer_idx] = o.float().norm(dim=-1).mean().item()
        return hook_fn
    
    for i in range(num_layers):
        h = model.model.layers[i].mlp.register_forward_hook(make_cal_hook(i))
        hooks2.append(h)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    for h in hooks2:
        h.remove()
        
    logits = outputs.logits
    has_nan = torch.isnan(logits).any().item()
    print(f"Logits NaN check: {'NaN detected!' if has_nan else 'No NaN'}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable (New MLP): {trainable_params:,}")
    
    return model, surgery_log, scale_factors

if __name__ == "__main__":
    print_config()
    model, tokenizer, breakdown = load_and_freeze()
    model, surgery_log, scale_factors = replace_all_mlps(
        model, tokenizer, breakdown["avg_mlp_norms"]
    )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = {
        "phase": "Phase 2 - MLP Replacement",
        "source_model": SOURCE_MODEL,
        "original_intermediate": ORIGINAL_INTERMEDIATE,
        "new_intermediate": NEW_INTERMEDIATE,
        "hidden_dim": ORIGINAL_HIDDEN_DIM,
        "total_params_original": breakdown["total_params"],
        "total_params_titan": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "surgery_log": surgery_log,
        "scale_factors": {str(k): round(v, 6) for k, v in scale_factors.items()},
    }
    
    meta_path = os.path.join(OUTPUT_DIR, "phase2_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")
