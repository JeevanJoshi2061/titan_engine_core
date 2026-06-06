import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
import math
import gc
import sys
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "../core")))

from titan_config import (
    SOURCE_MODEL, OUTPUT_DIR, DATASET_DIR, EXPORT_DIR,
    NEW_INTERMEDIATE, BATCH_SIZE, GRADIENT_ACCUMULATION,
    LEARNING_RATE, WEIGHT_DECAY, MAX_SEQ_LEN, EPOCHS,
    WARMUP_STEPS, MAX_GRAD_NORM, DEVICE, DTYPE,
    USE_GRADIENT_CHECKPOINTING
)

DATASET_PATH = os.path.join(DATASET_DIR, "titan_reasoning.jsonl")
SAVE_INTERVAL = 1000
KEEP_ONLY_LATEST = True
COMPILE_MODEL = True
RESUME_FROM_CHECKPOINT = True

class TitanLogicMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def calibrate_and_replace_mlps(model, tokenizer):
    print("Surgery: Replacing original MLPs with custom Logic MLPs...")
    
    orig_norms = []
    calibration_prompts = [
        "If A > B and B > C, then is A > C?",
        "Solve: 5 + 3 * 2",
        "Let x = 10. If x > 5, output True.",
        "Evaluate: NOT (True AND False)",
        "DFA transition trace: state q0 read 1"
    ]
    
    hooks = []
    layer_outputs = {}
    
    def get_hook(layer_idx):
        def hook_fn(module, input, output):
            if layer_idx not in layer_outputs:
                layer_outputs[layer_idx] = []
            layer_outputs[layer_idx].append(output.detach().norm(dim=-1).mean().item())
        return hook_fn

    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.register_forward_hook(get_hook(idx)))
        
    model.eval()
    for prompt in calibration_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model(**inputs)
            
    for h in hooks:
        h.remove()
        
    for idx in range(len(model.model.layers)):
        orig_norms.append(sum(layer_outputs[idx]) / len(calibration_prompts))
        
    hidden_dim = model.config.hidden_size
    for idx in range(model.config.num_hidden_layers):
        new_mlp = TitanLogicMLP(hidden_dim, NEW_INTERMEDIATE).to(DEVICE).to(DTYPE)
        
        with torch.no_grad():
            inputs = tokenizer(calibration_prompts[0], return_tensors="pt").to(DEVICE)
            orig_out = model.model.layers[idx].mlp(model.model.layers[idx].input_layernorm(model.model.layers[idx].self_attn(model.model.layers[idx].input_layernorm(inputs['input_ids']))[0]))
            new_out = new_mlp(model.model.layers[idx].input_layernorm(model.model.layers[idx].self_attn(model.model.layers[idx].input_layernorm(inputs['input_ids']))[0]))
            
            orig_norm = orig_out.norm(dim=-1).mean().item()
            new_norm = new_out.norm(dim=-1).mean().item()
            scale_factor = orig_norm / max(new_norm, 1e-5)
            
            new_mlp.down_proj.weight.data *= scale_factor
            
        model.model.layers[idx].mlp = new_mlp
        
    model.config.intermediate_size = NEW_INTERMEDIATE
    print(f"Replaced {model.config.num_hidden_layers} MLP layers.")
    return model

class TitanReasoningDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=512):
        self.samples = []
        print(f"Tokenizing dataset: {jsonl_path}")
        
        raw_texts = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    inp, out = item.get('input', ''), item.get('output', '')
                    if inp and out:
                        text = f"{inp} {out}{tokenizer.eos_token}"
                        raw_texts.append(text)
                except:
                    continue
        
        chunk_size = 50000
        total_samples = len(raw_texts)
        for i in range(0, total_samples, chunk_size):
            chunk = raw_texts[i : i + chunk_size]
            encoded = tokenizer(
                chunk,
                max_length=max_len,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            for j in range(len(chunk)):
                input_ids = torch.tensor(encoded['input_ids'][j], dtype=torch.long)
                attention_mask = torch.tensor(encoded['attention_mask'][j], dtype=torch.long)
                labels = input_ids.clone()
                self.samples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                })
            
        print(f"Dataset pre-tokenized: {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_lr(step, total_steps, warmup_steps, max_lr):
    if step < warmup_steps: return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

def save_checkpoint(model, optimizer, step, loss, epoch):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUTPUT_DIR, f"titan_mlp_step{step}.pt")
    temp_path = os.path.join(OUTPUT_DIR, "titan_mlp_temp.pt")
    
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    mlp_state = {name: param.data.cpu() for name, param in raw_model.named_parameters() if param.requires_grad}
    
    torch.save({
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'mlp_state_dict': mlp_state,
        'optimizer_state_dict': optimizer.state_dict()
    }, temp_path)
    
    if KEEP_ONLY_LATEST:
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith("titan_mlp_") and f.endswith(".pt") and f != "titan_mlp_temp.pt":
                try: os.remove(os.path.join(OUTPUT_DIR, f))
                except: pass
                
    if os.path.exists(ckpt_path):
        try: os.remove(ckpt_path)
        except: pass
    os.rename(temp_path, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path} (Loss: {loss:.4f})")

def find_latest_checkpoint():
    candidates = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith("titan_mlp_") and f.endswith(".pt"):
                candidates.append(os.path.join(OUTPUT_DIR, f))
    if not candidates:
        return None
    def get_step_number(path):
        filename = os.path.basename(path)
        match = re.search(r'(?:step|epoch)(\d+)', filename)
        if match:
            return int(match.group(1))
        try:
            return os.path.getmtime(path)
        except:
            return 0
    candidates.sort(key=get_step_number, reverse=True)
    return candidates[0]

def load_checkpoint(model, optimizer, ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model_state = model.state_dict()
    loaded_count = 0
    for name, param in checkpoint['mlp_state_dict'].items():
        if name in model_state:
            if model_state[name].shape == param.shape:
                model_state[name].copy_(param.to(DEVICE))
                loaded_count += 1
    print(f"Loaded {loaded_count} MLP parameter tensors.")
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        try:
            state_dict = checkpoint['optimizer_state_dict']
            optimizer.load_state_dict(state_dict)
            print("Optimizer state restored.")
        except Exception as e:
            print(f"Could not restore optimizer state ({e}). Re-initializing fresh optimizer.")
    return checkpoint.get('step', 0), checkpoint.get('epoch', 0), checkpoint.get('loss', 0.0)

def main():
    print(f"Starting Training Process on {DEVICE} | Dtype: {DTYPE}")
    
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(SOURCE_MODEL, torch_dtype=DTYPE, device_map=DEVICE, attn_implementation="sdpa")
    
    for p in model.parameters():
        p.requires_grad = False
        
    model = calibrate_and_replace_mlps(model, tokenizer)
    
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} | Frozen: {frozen_params:,}")
    
    dataset = TitanReasoningDataset(DATASET_PATH, tokenizer, max_len=MAX_SEQ_LEN)
    
    def collate_fn(batch):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        padded_attention = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {
            'input_ids': padded_inputs,
            'attention_mask': padded_attention,
            'labels': padded_labels
        }
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION
    total_steps = steps_per_epoch * EPOCHS
    
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_param_list, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
        print("Optimizer: 8-bit AdamW (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_param_list, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
        print("Optimizer: Standard AdamW")
    
    use_scaler = (DTYPE == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    
    start_epoch = 0
    resumed_step = 0
    resumed_loss = float('inf')
    if RESUME_FROM_CHECKPOINT:
        ckpt_path = find_latest_checkpoint()
        if ckpt_path:
            resumed_step, start_epoch, resumed_loss = load_checkpoint(model, optimizer, ckpt_path)
            print(f"Resuming from step {resumed_step}, loss={resumed_loss:.4f}")
            
    if COMPILE_MODEL:
        model = torch.compile(model, dynamic=True)
        
    global_step = resumed_step
    model.train()
    model.config.use_cache = False
    
    for epoch in range(start_epoch, EPOCHS):
        micro_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad()
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            with torch.amp.autocast('cuda', dtype=DTYPE):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUMULATION
                
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulated_loss += loss.item()
            micro_step += 1
            
            if micro_step % GRADIENT_ACCUMULATION == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_param_list, MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_param_list, MAX_GRAD_NORM)
                    optimizer.step()
                    
                current_lr = get_lr(global_step, total_steps, WARMUP_STEPS, LEARNING_RATE)
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
                    
                optimizer.zero_grad()
                global_step += 1
                
                step_loss = accumulated_loss
                accumulated_loss = 0.0
                
                print(f"Step {global_step:4d}/{total_steps} | Loss: {step_loss:.4f} | LR: {current_lr:.2e} | Grad: {grad_norm:.2f}")
                
                if global_step % SAVE_INTERVAL == 0:
                    save_checkpoint(model, optimizer, global_step, step_loss, epoch)
                    
                if global_step >= total_steps:
                    break
        
        if global_step >= total_steps:
            break
            
    print(f"Exporting final model to: {EXPORT_DIR}")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    if DTYPE == torch.float16:
        for p in raw_model.parameters():
            if p.requires_grad:
                p.data = p.data.to(DTYPE)
                
    raw_model.config.intermediate_size = NEW_INTERMEDIATE
    os.makedirs(EXPORT_DIR, exist_ok=True)
    raw_model.save_pretrained(EXPORT_DIR)
    tokenizer.save_pretrained(EXPORT_DIR)
    print("Training and Logic Engine export complete.")

if __name__ == "__main__":
    main()
