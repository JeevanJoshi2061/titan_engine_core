"""
TITAN ENGINE - PHASE 4: TRAINING LOOP (SYMPHONY ARCHITECTURE)
==============================================================
Trains the custom SymphonyASHCLayers on the reasoning dataset.
Everything else (Attention, Embeddings, Norms, LM Head) remains frozen.

VRAM Budget (RTX 4060 8GB):
  Model (frozen+trainable):  ~1.3 GB
  Optimizer states (FP32):   ~2.1 GB  (264M * 2 states * 4 bytes)
  Gradients (BF16):          ~0.5 GB
  Activations (checkpointed): ~1.0 GB
  TOTAL:                     ~4.9 GB  (comfortable under 8 GB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Adafactor
import json
import os
import time
import math
import gc
import logging
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("titan_train_fft")

from titan_config import (
    SOURCE_MODEL, OUTPUT_DIR, DATASET_DIR, EXPORT_DIR,
    DEVICE, DTYPE, BATCH_SIZE, GRADIENT_ACCUMULATION,
    LEARNING_RATE, WEIGHT_DECAY, MAX_SEQ_LEN, EPOCHS,
    WARMUP_STEPS, MAX_GRAD_NORM, USE_GRADIENT_CHECKPOINTING,
    print_config
)
from titan_phase1_freeze import load_and_freeze

# ============================================================
# LOCAL HYPERPARAMETERS & CONFIGS
# ============================================================
LAMBDA_DECAY = 0.99
# Set to None to train from scratch, or point to your step7500 checkpoint:
RESUME_FROM_CHECKPOINT = r"E:\titan Engine new\titan_symphony_step7500.pt"

# ============================================================
# SYMPHONY ARCHITECTURE COMPONENTS
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        return (x_f32 * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class ASHCRouter(nn.Module):
    """
    Pillar 1: The Gumbel-Softmax Router
    Evaluates each token's hidden state and outputs a discrete [0.0, 1.0] decision.
    Uses hard=True for Straight-Through Estimator (STE) during training,
    allowing discrete routing decisions while passing gradients back.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.salience_proj = nn.Linear(hidden_dim, 2) # [prob_background, prob_fact]
        
    def forward(self, hidden_states, temperature=1.0, hard=True):
        logits = self.salience_proj(hidden_states)
        
        if self.training:
            routing_weights = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        else:
            preds = torch.argmax(logits, dim=-1)
            routing_weights = F.one_hot(preds, num_classes=2).to(hidden_states.dtype)
            
        fact_selection = routing_weights[:, :, 1] # [B, S]
        return fact_selection


class PhantomKVCache(nn.Module):
    """
    Pillar 2 & 4: The Phantom Fact-Ledger (True Dynamic VRAM Sparsity)
    Filters out non-facts completely to shrink tensor dimensions.
    Pads batch items dynamically to the maximum number of facts in the current batch.
    """
    def __init__(self, hidden_dim, max_capacity=2048):
        super().__init__()
        self.max_capacity = max_capacity
        self.hidden_dim = hidden_dim
        self.reset_cache()
        
    def reset_cache(self):
        self.keys = None
        self.values = None
        self.masks = None
        self.num_tokens = 0
        
    def add_to_cache(self, new_keys, new_values, fact_selection):
        B, S, D = new_keys.shape
        device = new_keys.device
        
        batch_keys = []
        batch_values = []
        max_facts = 0
        
        for b in range(B):
            valid_indices = torch.nonzero(fact_selection[b] > 0.5).squeeze(-1)
            b_keys = new_keys[b, valid_indices]
            b_values = new_values[b, valid_indices]
            
            batch_keys.append(b_keys)
            batch_values.append(b_values)
            if b_keys.shape[0] > max_facts:
                max_facts = b_keys.shape[0]
                
        if max_facts == 0:
            return None, None
            
        padded_keys = torch.zeros(B, max_facts, D, device=device, dtype=new_keys.dtype)
        padded_values = torch.zeros(B, max_facts, D, device=device, dtype=new_values.dtype)
        padded_masks = torch.zeros(B, max_facts, device=device, dtype=torch.bool)
        
        for b in range(B):
            num_f = batch_keys[b].shape[0]
            if num_f > 0:
                padded_keys[b, :num_f] = batch_keys[b]
                padded_values[b, :num_f] = batch_values[b]
                padded_masks[b, :num_f] = True
                
        if self.keys is None:
            self.keys = padded_keys
            self.values = padded_values
            self.masks = padded_masks
        else:
            self.keys = torch.cat([self.keys, padded_keys], dim=1)
            self.values = torch.cat([self.values, padded_values], dim=1)
            self.masks = torch.cat([self.masks, padded_masks], dim=1)
            
        self.num_tokens = self.keys.shape[1]
        
        overflow_keys = None
        overflow_values = None
        
        if self.num_tokens > self.max_capacity:
            overflow = self.num_tokens - self.max_capacity
            overflow_keys = self.keys[:, :overflow, :]
            overflow_values = self.values[:, :overflow, :]
            
            self.keys = self.keys[:, overflow:, :]
            self.values = self.values[:, overflow:, :]
            self.masks = self.masks[:, overflow:, :]
            self.num_tokens = self.max_capacity
            
        return overflow_keys, overflow_values

    def exact_attention_retrieval(self, query):
        if self.keys is None or self.keys.shape[1] == 0:
            return torch.zeros_like(query)
            
        scores = torch.matmul(query, self.keys.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        mask_expanded = self.masks.unsqueeze(1)
        scores = scores.masked_fill(~mask_expanded, -10000.0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        exact_context = torch.matmul(attn_weights, self.values)
        return exact_context


class SymphonyASHCLayer(nn.Module):
    """
    The True Hardware-Optimized Symphony ASH-C Memory Layer.
    """
    def __init__(self, hidden_dim, original_mlp, lambda_decay=0.99, scale_factor=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.original_mlp = original_mlp
        self.lambda_decay = lambda_decay
        self.scale_factor = scale_factor
        self.freq_dim = hidden_dim // 2 + 1
        
        self.router = ASHCRouter(hidden_dim)
        self.phantom_cache = PhantomKVCache(hidden_dim, max_capacity=2048)
        
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.prism_w1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.prism_w2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.prism_w3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.norm = RMSNorm(hidden_dim)
        
        self.gate = nn.Parameter(torch.zeros(1))
        self.m_state = None
        self.last_contrastive_loss = None

        nn.init.kaiming_normal_(self.key_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.value_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.query_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.prism_w1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.prism_w2.weight, nonlinearity='linear')
        nn.init.xavier_normal_(self.prism_w3.weight)

    def reset_state(self):
        self.m_state = None
        self.phantom_cache.reset_cache()
        self.last_contrastive_loss = None

    def forward(self, x):
        if self.training:
            self.reset_state()
            
        B, S, D = x.shape
        device = x.device
        dtype = x.dtype
        
        mlp_out = self.original_mlp(x)
        
        # 1. Soft-Router routing decision
        fact_selection = self.router(x)
        
        # 2. Project K, V, Q in dtype and cast to float32 for high-precision math
        K = self.key_proj(x)
        V = self.value_proj(x)
        Q = self.query_proj(x)
        
        K_f32 = K.float()
        V_f32 = V.float()
        Q_f32 = Q.float()
        
        # Update dynamic VRAM-efficient Phantom Cache in float32
        overflow_K, overflow_V = self.phantom_cache.add_to_cache(K_f32, V_f32, fact_selection)
        
        # 3. FHRR Holographic Memory operations in float32
        K_freq = torch.fft.rfft(K_f32, dim=-1)
        V_freq = torch.fft.rfft(V_f32, dim=-1)
        Q_freq = torch.fft.rfft(Q_f32, dim=-1)
        
        K_abs = torch.sqrt(K_freq.real.pow(2) + K_freq.imag.pow(2) + 1e-12)
        K_norm_freq = K_freq / K_abs.to(dtype=torch.complex64)
        
        bound = K_norm_freq * V_freq
        
        if self.training or S > 1 or self.m_state is None or self.m_state.shape[0] != B:
            initial_state = torch.zeros(B, self.freq_dim, dtype=torch.complex64, device=device)
        else:
            initial_state = self.m_state
            
        t_indices = torch.arange(S, device=device).unsqueeze(1)
        i_indices = torch.arange(S, device=device).unsqueeze(0)
        power = torch.clamp(t_indices - i_indices, min=0)
        mask = (t_indices - i_indices >= 0).float()
        W = ((self.lambda_decay ** power) * mask).to(dtype=torch.complex64)
        
        scale = math.sqrt(1.0 - self.lambda_decay ** 2)
        outputs_rec_freq = scale * torch.matmul(W, bound)
        
        steps = torch.arange(1, S + 1, device=device, dtype=torch.float32)
        decay_factors = (self.lambda_decay ** steps).unsqueeze(0).unsqueeze(2).to(dtype=torch.complex64)
        outputs_rec_freq = outputs_rec_freq + initial_state.unsqueeze(1) * decay_factors
        
        if overflow_K is not None:
            O_K_freq = torch.fft.rfft(overflow_K, dim=-1)
            O_V_freq = torch.fft.rfft(overflow_V, dim=-1)
            O_K_abs = torch.sqrt(O_K_freq.real.pow(2) + O_K_freq.imag.pow(2) + 1e-12)
            O_K_norm = O_K_freq / O_K_abs.to(dtype=torch.complex64)
            
            overflow_bound = (O_K_norm * O_V_freq).mean(dim=1)
            outputs_rec_freq = outputs_rec_freq + overflow_bound.unsqueeze(1)
            
        if not self.training:
            self.m_state = outputs_rec_freq[:, -1, :].detach()
            
        Q_abs = torch.sqrt(Q_freq.real.pow(2) + Q_freq.imag.pow(2) + 1e-12)
        Q_norm_freq = Q_freq / Q_abs.to(dtype=torch.complex64)
        outputs_rec_freq = outputs_rec_freq * torch.conj(Q_norm_freq)
        
        fhrr_rec = torch.fft.irfft(outputs_rec_freq, n=D, dim=-1)
        exact_rec = self.phantom_cache.exact_attention_retrieval(Q_f32)
        
        # 5. Pillar 3: Contrastive Hopfield Prism Execution
        combined_signal = torch.cat([fhrr_rec, exact_rec], dim=-1).to(dtype)
        gated = self.prism_w1(combined_signal) * self.act(self.prism_w2(combined_signal))
        clean_out_proj = self.prism_w3(gated)
        
        # RMSNorm and loss are evaluated in float32 for training stability
        clean_out_f32 = self.norm(clean_out_proj.float()) * (self.scale_factor / math.sqrt(self.hidden_dim))
        
        if self.training:
            pos_dist = F.pairwise_distance(clean_out_f32, exact_rec, p=2)
            neg_dist = F.pairwise_distance(clean_out_f32, fhrr_rec, p=2)
            self.last_contrastive_loss = torch.clamp(1.0 + pos_dist - neg_dist, min=0.0).mean()
        else:
            self.last_contrastive_loss = None
            
        clean_out = clean_out_f32.to(dtype)
        gate_val = torch.sigmoid(self.gate).to(dtype)
        output = mlp_out + gate_val * clean_out
        
        return output

# ============================================================
# SURGERY FUNCTION
# ============================================================

def replace_all_mlps(model, tokenizer, avg_mlp_norms):
    log.info("\n[PHASE 2] REPLACE MLP WITH SYMPHONY ARCHITECTURE (PARALLEL HYBRID)")
    log.info("=" * 70)
    
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    
    log.info(f"  Replacing {num_layers} MLPs with SymphonyASHCLayers...")
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        orig_norm = avg_mlp_norms.get(layer_idx, 1.0)
        
        original_mlp = layer.mlp
        for param in original_mlp.parameters():
            param.requires_grad = False
            
        # Create Symphony ASH-C Layer (custom layers are initialized in float32 by default)
        ash_c_layer = SymphonyASHCLayer(hidden_dim, original_mlp, lambda_decay=LAMBDA_DECAY, scale_factor=orig_norm)
        # Move the layer to device without changing the dtype of submodules (keeping original_mlp in bfloat16)
        ash_c_layer = ash_c_layer.to(device=DEVICE)
        
        layer.mlp = ash_c_layer
        
        for name, param in layer.mlp.named_parameters():
            if 'original_mlp' not in name:
                if layer_idx < 20:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    gc.collect()
    torch.cuda.empty_cache()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  [PHASE 2 COMPLETE] Symphony ASH-C Memory layers injected. Trainable params: {trainable_params:,}")
    return model

# ============================================================
# DATASET
# ============================================================

class TitanReasoningDataset(Dataset):
    """Load JSONL dataset for causal language modeling."""
    def __init__(self, jsonl_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        log.info(f"  Loading dataset from: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    inp = item.get('input', '')
                    out = item.get('output', '')
                    if inp and out:
                        self.samples.append((inp, out))
                except json.JSONDecodeError:
                    continue
        
        log.info(f"  Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inp, out = self.samples[idx]
        text = f"Question: {inp}\nAnswer: {out}{self.tokenizer.eos_token}"
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

def get_lr(step, total_steps, warmup_steps, max_lr):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train():
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    log.info("=" * 70)
    log.info("TITAN ENGINE - PHASE 4: TRAINING (SYMPHONY ARCHITECTURE)")
    log.info("=" * 70)
    
    print_config()
    
    # ----------------------------------------------------------
    # STEP 1: Load and surgically modify model
    # ----------------------------------------------------------
    log.info("\n[STEP 1] Loading model and executing Symphony surgery...")
    
    model, tokenizer, breakdown = load_and_freeze()
    model = replace_all_mlps(
        model, tokenizer, breakdown['avg_mlp_norms']
    )
    
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        log.info("  Gradient checkpointing: ENABLED")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"  Trainable params: {trainable_params:,} ({trainable_params*2/1e6:.1f} MB)")
    log.info(f"  Frozen params:    {frozen_params:,} ({frozen_params*2/1e6:.1f} MB)")
    
    vram_after_model = torch.cuda.memory_allocated() / 1e6
    log.info(f"  VRAM after model surgery: {vram_after_model:.1f} MB")
    
    # ----------------------------------------------------------
    # STEP 2: Load dataset
    # ----------------------------------------------------------
    log.info("\n[STEP 2] Loading dataset...")
    
    dataset_path = os.path.join(DATASET_DIR, "titan_reasoning.jsonl")
    if not os.path.exists(dataset_path):
        log.error(f"  Dataset not found: {dataset_path}")
        return
    
    dataset = TitanReasoningDataset(dataset_path, tokenizer, max_len=MAX_SEQ_LEN)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    
    steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION
    total_steps = steps_per_epoch * EPOCHS
    
    log.info(f"  Dataset size:     {len(dataset)} samples")
    log.info(f"  Steps/epoch:      {steps_per_epoch} (after grad accum)")
    log.info(f"  Total steps:      {total_steps}")
    
    # ----------------------------------------------------------
    # STEP 3: Setup optimizer
    # ----------------------------------------------------------
    log.info("\n[STEP 3] Setting up optimizer...")
    
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = Adafactor(
        trainable_param_list,
        lr=LEARNING_RATE,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Checkpoint loading with surgical weight migration
    global_step = 0
    start_epoch = 0
    best_loss = float('inf')
    
    if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
        log.info(f"Resuming training from checkpoint: {RESUME_FROM_CHECKPOINT}")
        checkpoint = torch.load(RESUME_FROM_CHECKPOINT, map_location=DEVICE)
        state_dict = checkpoint['memory_state_dict']
        
        model_dict = model.state_dict()
        migrated_dict = {}
        for name, param in state_dict.items():
            # Map checkpoint names containing .fhrr. to current architecture names
            mapped_name = name.replace(".mlp.fhrr.", ".mlp.")
            mapped_name = mapped_name.replace(".mlp.w1.weight", ".mlp.prism_w1.weight")
            mapped_name = mapped_name.replace(".mlp.w2.weight", ".mlp.prism_w2.weight")
            mapped_name = mapped_name.replace(".mlp.w3.weight", ".mlp.prism_w3.weight")
            if mapped_name in model_dict:
                if param.shape == model_dict[mapped_name].shape:
                    migrated_dict[mapped_name] = param
                else:
                    if 'prism_w1.weight' in mapped_name or 'prism_w2.weight' in mapped_name:
                        log.info(f"  Surgically migrating weight for {mapped_name}: {param.shape} -> {model_dict[mapped_name].shape}")
                        new_param = model_dict[mapped_name].clone()
                        H = param.shape[1]
                        new_param[:, :H] = param
                        new_param[:, H:] = 0.0
                        migrated_dict[mapped_name] = new_param
                    else:
                        log.warning(f"  Skipping {mapped_name} due to shape mismatch: {param.shape} vs {model_dict[mapped_name].shape}")
            else:
                log.warning(f"  Checkpoint weight {name} (mapped to {mapped_name}) not found in current model.")
                
        model.load_state_dict(migrated_dict, strict=False)
        
        # Load optimizer state if available to prevent gradient spikes
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log.info("  Optimizer states successfully restored.")
            except Exception as e:
                log.warning(f"  Could not load optimizer states: {e}. Starting with clean optimizer moments.")
                
        global_step = checkpoint.get('step', 0)
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        log.info(f"Surgically loaded checkpoint at step {global_step}, epoch {start_epoch}, best loss {best_loss:.4f}")
    
    # Adafactor uses factorized states, memory is negligible
    optimizer_mem = (trainable_params * 2 / 1536 * 4) / 1e6 # Row/col sums
    log.info(f"  Optimizer memory: ~{optimizer_mem:.1f} MB (Adafactor row/col factorized states)")
    
    # GradScaler for stable FP16/BF16 training if needed
    scaler = torch.amp.GradScaler('cuda') if DTYPE == torch.float16 else None
    
    # ----------------------------------------------------------
    # STEP 4: Training Loop
    # ----------------------------------------------------------
    log.info("\n[STEP 4] Starting training...")
    log.info("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_start = time.time()
    log_interval = 1
    save_interval = 200
    
    loss_history = []
    lr_history = []
    
    model.train()
    
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_tokens = 0
        micro_step = 0
        accumulated_loss = 0.0
        
        log.info(f"\n--- EPOCH {epoch + 1}/{start_epoch + EPOCHS} ---")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # Reset memory states per batch element
            for name, module in model.named_modules():
                if isinstance(module, SymphonyASHCLayer):
                    module.reset_state()
            
            with torch.amp.autocast('cuda', dtype=DTYPE):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                clm_loss = outputs.loss
                
                # Dynamic accumulation of contrastive alignment loss
                contrastive_loss = 0.0
                num_ash_layers = 0
                for name, module in model.named_modules():
                    if isinstance(module, SymphonyASHCLayer) and module.last_contrastive_loss is not None:
                        contrastive_loss += module.last_contrastive_loss
                        num_ash_layers += 1
                        
                if num_ash_layers > 0:
                    contrastive_loss = contrastive_loss / num_ash_layers
                    total_loss = (clm_loss + 0.05 * contrastive_loss) / GRADIENT_ACCUMULATION
                else:
                    total_loss = clm_loss / GRADIENT_ACCUMULATION
            
            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            accumulated_loss += total_loss.item() * GRADIENT_ACCUMULATION
            num_tokens = (labels != -100).sum().item()
            epoch_tokens += num_tokens
            micro_step += 1
            
            if micro_step % GRADIENT_ACCUMULATION == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_param_list, MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_param_list, MAX_GRAD_NORM)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                global_step += 1
                step_loss = accumulated_loss
                epoch_loss += step_loss
                
                loss_history.append(step_loss)
                
                current_lr = get_lr(global_step, total_steps, WARMUP_STEPS, LEARNING_RATE)
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
                lr_history.append(current_lr)
                
                accumulated_loss = 0.0
                
                if global_step % log_interval == 0 or global_step == 1:
                    vram_now = torch.cuda.memory_allocated() / 1e6
                    vram_peak = torch.cuda.max_memory_allocated() / 1e6
                    elapsed = time.time() - train_start
                    tokens_per_sec = epoch_tokens / max(time.time() - epoch_start, 0.1)
                    
                    log.info(
                        f"  Step {global_step:4d}/{total_steps} | "
                        f"Loss: {step_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Grad: {grad_norm:.2f} | "
                        f"VRAM: {vram_now:.0f}/{vram_peak:.0f} MB | "
                        f"Tok/s: {tokens_per_sec:.0f}"
                    )
                
                if global_step % save_interval == 0:
                    save_checkpoint(model, optimizer, global_step, step_loss, epoch)
                
                if step_loss < best_loss:
                    best_loss = step_loss
        
        avg_epoch_loss = epoch_loss / max(steps_per_epoch, 1)
        log.info(f"\n  Epoch {epoch + 1} complete: Avg Loss: {avg_epoch_loss:.4f} | Time: {time.time() - epoch_start:.1f}s")
        save_checkpoint(model, optimizer, global_step, avg_epoch_loss, epoch, is_epoch=True)
    
    # ----------------------------------------------------------
    # STEP 5: Save & Export Trained Weights
    # ----------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("=" * 70)
    
    history_path = os.path.join(OUTPUT_DIR, "training_history_fft.json")
    with open(history_path, 'w') as f:
        json.dump({
            'loss': loss_history,
            'lr': lr_history,
            'total_steps': global_step,
            'best_loss': best_loss,
        }, f, indent=2)
    log.info(f"  History saved to: {history_path}")
    
    export_model(model, tokenizer)
    test_generation(model, tokenizer)
    
    log.info("\n" + "=" * 70)
    log.info("PHASE 4 COMPLETE - Symphony memory layer trained successfully!")
    log.info("=" * 70)

# ============================================================
# UTILITIES
# ============================================================

def save_checkpoint(model, optimizer, step, loss, epoch, is_epoch=False):
    tag = f"epoch{epoch+1}" if is_epoch else f"step{step}"
    ckpt_path = os.path.join(OUTPUT_DIR, f"titan_symphony_{tag}.pt")
    
    memory_state = {
        name: param.data.cpu()
        for name, param in model.named_parameters()
        if (param.requires_grad or ('.mlp.' in name and 'original_mlp' not in name))
    }
    
    torch.save({
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'memory_state_dict': memory_state,
        'optimizer_state_dict': optimizer.state_dict()
    }, ckpt_path)
    
    size_mb = os.path.getsize(ckpt_path) / 1e6
    log.info(f"  [SAVED] {ckpt_path} ({size_mb:.1f} MB, loss={loss:.4f})")

def export_model(model, tokenizer):
    log.info(f"\n  Exporting trained weights to: {EXPORT_DIR}")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    memory_state = {
        name: param.data.to(DTYPE).cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    weights_path = os.path.join(EXPORT_DIR, "titan_symphony_weights.pt")
    torch.save(memory_state, weights_path)
    
    meta = {
        "engine": "Titan Engine - Symphony Architecture",
        "source_model": SOURCE_MODEL,
        "frozen": ["attention", "embeddings", "norms", "lm_head"],
        "trained": ["mlp (SymphonyASHCLayer)"],
        "lambda_decay": LAMBDA_DECAY
    }
    with open(os.path.join(EXPORT_DIR, "titan_symphony_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    log.info(f"  Export complete! Weights saved to: {weights_path}")

def test_generation(model, tokenizer):
    log.info("\n  --- POST-TRAINING GENERATION TEST ---")
    model.eval()
    test_prompts = [
        "Question: What is 15 * 7?\nAnswer:",
        "Question: If A > B and B > C, is A > C?\nAnswer:",
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        for name, module in model.named_modules():
            if isinstance(module, SymphonyASHCLayer):
                module.reset_state()
                
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        log.info(f"\n  Q: {prompt.split(chr(10))[0]}")
        log.info(f"  A: {response[:150]}")
    
    model.train()

if __name__ == "__main__":
    train()
