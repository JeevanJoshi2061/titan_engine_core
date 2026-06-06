"""
TITAN ENGINE - KAGGLE UNIFIED TRAINING SCRIPT (SYMPHONY ARCHITECTURE)
=====================================================================
A self-contained script combining:
  1. Configs
  2. Phase 1 (Freeze)
  3. Phase 2 (Symphony FHRR Memory Layer Replacement)
  4. Phase 4 (Reasoning training loop)

INSTRUCTIONS FOR KAGGLE:
1. Ensure Internet is enabled (to download Qwen2.5-1.5B-Instruct tokenizer/model if not uploaded).
2. Upload your `titan_reasoning.jsonl` dataset to Kaggle.
3. Update `DATASET_PATH` below to point to your uploaded Kaggle dataset path (e.g., /kaggle/input/...).
4. Select a GPU accelerator (L4 or T4x2). Note: if using T4, change DTYPE to torch.float16.
5. Run this script. Outputs will be saved to /kaggle/working/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Adafactor
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
import math
import gc
import logging
import sys

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("titan_train_kaggle")

# ============================================================
# CONFIGURATION
# ============================================================
# Update these paths for Kaggle
SOURCE_MODEL = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/1.5b-instruct/1"  # Change if you uploaded model to /kaggle/input/
DATASET_PATH = "/kaggle/input/datasets/tsfmaster12/70mb-dataset-titan/titan_reasoning.jsonl"  # <-- UPDATE THIS
OUTPUT_DIR = "/kaggle/working/checkpoints"
EXPORT_DIR = "/kaggle/working/titan_model"

# Architecture
ORIGINAL_HIDDEN_DIM = 1536
ORIGINAL_INTERMEDIATE = 8960
NUM_LAYERS = 28

# Training Hyperparameters
BATCH_SIZE = 1                   # Reduced to 1 to guarantee OOM-free training
GRADIENT_ACCUMULATION = 16      # Increased to 16 to keep effective batch size at 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 512
EPOCHS = 2                
WARMUP_STEPS = 300
MAX_GRAD_NORM = 1.0
MAX_TRAIN_SAMPLES = 50000       # Slice dataset to fit inside Kaggle 12-hour limit

# Checkpoint Settings
SAVE_CHECKPOINTS = True          # Toggle: True = save checkpoints, False = only final model
SAVE_INTERVAL = 500              
KEEP_ONLY_LATEST = True          
RESUME_FROM_CHECKPOINT = None          

# Hardware / System
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
USE_GRADIENT_CHECKPOINTING = True
LAMBDA_DECAY = 0.99
USE_SWIGLU = True


def print_config():
    log.info("=" * 70)
    log.info("TITAN ENGINE CONFIGURATION (KAGGLE FFT MATH)")
    log.info("=" * 70)
    log.info(f"  Source Model:       {SOURCE_MODEL}")
    log.info(f"  Dataset Path:       {DATASET_PATH}")
    log.info(f"  Device:             {DEVICE}")
    log.info(f"  Dtype:              {DTYPE}")
    log.info(f"  Original MLP dim:   {ORIGINAL_INTERMEDIATE}")
    log.info(f"  Hidden dim:         {ORIGINAL_HIDDEN_DIM}")
    log.info(f"  Batch size:         {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    log.info(f"  Learning rate:      {LEARNING_RATE}")
    log.info(f"  Grad checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    log.info(f"  SwiGLU Enabled:     {USE_SWIGLU}")
    log.info("=" * 70)


# ============================================================
# PHASE 1: LOAD & FREEZE
# ============================================================
def load_and_freeze():
    log.info("\n[PHASE 1] LOAD & FREEZE")
    log.info("=" * 70)
    
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Load tokenizer
    log.info(f"  Loading tokenizer from: {SOURCE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    
    # Load model
    log.info(f"  Loading model in {DTYPE}...")
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="sdpa"  # Force PyTorch Scaled Dot Product Attention
    )
    
    num_layers = model.config.num_hidden_layers
    
    log.info(f"\n  --- FREEZING ALL PARAMETERS ---")
    frozen_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        param.requires_grad = False
        frozen_count += 1
        total_count += 1
    log.info(f"  Frozen: {frozen_count}/{total_count} parameter tensors")
    
    # Capture Original MLP Output Norms for Phase 2 Calibration
    log.info(f"\n  --- CAPTURING ORIGINAL MLP OUTPUT NORMS ---")
    original_mlp_norms = {}
    hooks = []
    
    def make_norm_hook(layer_idx):
        def hook_fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            norm = o.float().norm(dim=-1).mean().item()
            original_mlp_norms[layer_idx] = norm
        return hook_fn
    
    for i in range(num_layers):
        h = model.model.layers[i].mlp.register_forward_hook(make_norm_hook(i))
        hooks.append(h)
    
    calibration_prompts = [
        "Solve: 3x + 7 = 22",
        "If P implies Q and Q is false, what can we say about P?",
        "What is 15 * 8 + 12?"
    ]
    
    all_layer_norms = {i: [] for i in range(num_layers)}
    
    for prompt in calibration_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            model(**inputs)
        for layer_idx, norm_val in original_mlp_norms.items():
            all_layer_norms[layer_idx].append(norm_val)
    
    for h in hooks:
        h.remove()
    
    avg_mlp_norms = {}
    for layer_idx in range(num_layers):
        norms = all_layer_norms[layer_idx]
        avg_mlp_norms[layer_idx] = sum(norms) / len(norms) if norms else 1.0
    
    log.info(f"  [PHASE 1 COMPLETE] Model loaded and frozen. [OK]")
    
    breakdown = {
        "avg_mlp_norms": avg_mlp_norms,
    }
    return model, tokenizer, breakdown


# ============================================================
# PHASE 2: REPLACE MLP WITH SYMPHONY ARCHITECTURE
# ============================================================
class RMSNorm(nn.Module):
    """Simple RMSNorm implementation for compatibility across all PyTorch versions."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Cast to float32 to prevent x.pow(2) overflow in fp16
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
        # hidden_states: [B, S, D]
        logits = self.salience_proj(hidden_states) # [B, S, 2]
        
        if self.training:
            # hard=True returns discrete one-hot vectors, but gradients bypass the step function
            routing_weights = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        else:
            # Inference: argmax for true discrete choice
            preds = torch.argmax(logits, dim=-1)
            routing_weights = F.one_hot(preds, num_classes=2).to(hidden_states.dtype)
            
        fact_selection = routing_weights[:, :, 1] # [B, S] (binary 0.0 or 1.0)
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
        self.keys = None   # [B, N, D] (padded fact keys)
        self.values = None # [B, N, D] (padded fact values)
        self.masks = None  # [B, N] (boolean mask of valid cache entries)
        self.num_tokens = 0
        
    def add_to_cache(self, new_keys, new_values, fact_selection):
        """
        Dynamically extracts and stores only tokens where fact_selection == 1.
        Saves VRAM by discarding all non-fact tokens before cache expansion.
        """
        B, S, D = new_keys.shape
        device = new_keys.device
        
        # Step 1: Collect valid facts per batch element
        batch_keys = []
        batch_values = []
        max_facts = 0
        
        for b in range(B):
            valid_indices = torch.nonzero(fact_selection[b] > 0.5).squeeze(-1)
            b_keys = new_keys[b, valid_indices]     # [num_facts, D]
            b_values = new_values[b, valid_indices] # [num_facts, D]
            
            batch_keys.append(b_keys)
            batch_values.append(b_values)
            if b_keys.shape[0] > max_facts:
                max_facts = b_keys.shape[0]
                
        # If no facts are found in this batch forward step, return empty signals
        if max_facts == 0:
            return None, None
            
        # Step 2: Pad current batch facts to max_facts to keep computations batched
        padded_keys = torch.zeros(B, max_facts, D, device=device, dtype=new_keys.dtype)
        padded_values = torch.zeros(B, max_facts, D, device=device, dtype=new_values.dtype)
        padded_masks = torch.zeros(B, max_facts, device=device, dtype=torch.bool)
        
        for b in range(B):
            num_f = batch_keys[b].shape[0]
            if num_f > 0:
                padded_keys[b, :num_f] = batch_keys[b]
                padded_values[b, :num_f] = batch_values[b]
                padded_masks[b, :num_f] = True
                
        # Step 3: Append to historical cache
        if self.keys is None:
            self.keys = padded_keys
            self.values = padded_values
            self.masks = padded_masks
        else:
            self.keys = torch.cat([self.keys, padded_keys], dim=1)
            self.values = torch.cat([self.values, padded_values], dim=1)
            self.masks = torch.cat([self.masks, padded_masks], dim=1)
            
        self.num_tokens = self.keys.shape[1]
        
        # Step 4: Pillar 4 (Sleep Cycle - Cache Eviction / Offloading)
        overflow_keys = None
        overflow_values = None
        
        if self.num_tokens > self.max_capacity:
            overflow = self.num_tokens - self.max_capacity
            
            # Retrieve overflow tokens to be bound into FHRR memory
            overflow_keys = self.keys[:, :overflow, :]
            overflow_values = self.values[:, :overflow, :]
            
            # Truncate cache
            self.keys = self.keys[:, overflow:, :]
            self.values = self.values[:, overflow:, :]
            self.masks = self.masks[:, overflow:, :]
            self.num_tokens = self.max_capacity
            
        return overflow_keys, overflow_values

    def exact_attention_retrieval(self, query):
        """Calculates attention only on valid stored facts, masking padded zeros."""
        if self.keys is None or self.keys.shape[1] == 0:
            return torch.zeros_like(query)
            
        # Scaled dot-product scores: [B, S, D] x [B, D, N] -> [B, S, N]
        scores = torch.matmul(query, self.keys.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        
        # Apply attention mask to block padded elements
        mask_expanded = self.masks.unsqueeze(1) # [B, 1, N]
        scores = scores.masked_fill(~mask_expanded, -10000.0)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Softmax over masked entries can yield NaNs if a batch row has absolutely no facts; fix with nan_to_num
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        exact_context = torch.matmul(attn_weights, self.values)
        return exact_context


class SymphonyASHCLayer(nn.Module):
    """
    The True Hardware-Optimized Symphony ASH-C Memory Layer.
    Combines Gumbel-Softmax routing, dynamic memory pruning, exact masked cache,
    FHRR holographic frequency-domain recurrence, and a Contrastive Hopfield clean-up Prism.
    """
    def __init__(self, hidden_dim, original_mlp, lambda_decay=0.99, scale_factor=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.original_mlp = original_mlp # Frozen original MLP
        self.lambda_decay = lambda_decay
        self.scale_factor = scale_factor
        self.freq_dim = hidden_dim // 2 + 1
        
        # Pillar 1: Router
        self.router = ASHCRouter(hidden_dim)
        
        # Pillar 2: Dynamic Cache
        self.phantom_cache = PhantomKVCache(hidden_dim, max_capacity=2048)
        
        # Projections for Keys, Values, and Queries
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Pillar 3: Contrastive Hopfield Prism (Gated cleanup)
        self.prism_w1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.prism_w2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.prism_w3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.norm = RMSNorm(hidden_dim)
        
        self.gate = nn.Parameter(torch.zeros(1))
        self.m_state = None # Recurrent FHRR state
        self.last_contrastive_loss = None

        # Initialize projections with Kaiming normal
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
        
        # Frozen base execution
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
        
        # Normalize Keys in frequency domain to avoid NaN gradients at zero
        K_abs = torch.sqrt(K_freq.real.pow(2) + K_freq.imag.pow(2) + 1e-12)
        K_norm_freq = K_freq / K_abs.to(dtype=torch.complex64)
        
        # Element-wise circular binding
        bound = K_norm_freq * V_freq
        
        # Retrieve or initialize FHRR state
        if self.training or S > 1 or self.m_state is None or self.m_state.shape[0] != B:
            initial_state = torch.zeros(B, self.freq_dim, dtype=torch.complex64, device=device)
        else:
            initial_state = self.m_state
            
        # Calculate decay matrix W
        t_indices = torch.arange(S, device=device).unsqueeze(1)
        i_indices = torch.arange(S, device=device).unsqueeze(0)
        power = torch.clamp(t_indices - i_indices, min=0)
        mask = (t_indices - i_indices >= 0).float()
        W = ((self.lambda_decay ** power) * mask).to(dtype=torch.complex64)
        
        # Vectorized convolution
        scale = math.sqrt(1.0 - self.lambda_decay ** 2)
        outputs_rec_freq = scale * torch.matmul(W, bound)
        
        # Add initial state contributions
        steps = torch.arange(1, S + 1, device=device, dtype=torch.float32)
        decay_factors = (self.lambda_decay ** steps).unsqueeze(0).unsqueeze(2).to(dtype=torch.complex64)
        outputs_rec_freq = outputs_rec_freq + initial_state.unsqueeze(1) * decay_factors
        
        # Pillar 4: Consolidate cache overflow into FHRR (Sleep Cycle)
        if overflow_K is not None:
            O_K_freq = torch.fft.rfft(overflow_K, dim=-1)
            O_V_freq = torch.fft.rfft(overflow_V, dim=-1)
            O_K_abs = torch.sqrt(O_K_freq.real.pow(2) + O_K_freq.imag.pow(2) + 1e-12)
            O_K_norm = O_K_freq / O_K_abs.to(dtype=torch.complex64)
            
            overflow_bound = (O_K_norm * O_V_freq).mean(dim=1)
            outputs_rec_freq = outputs_rec_freq + overflow_bound.unsqueeze(1)
            
        # Save final step state for generation
        if not self.training:
            self.m_state = outputs_rec_freq[:, -1, :].detach()
            
        # Unbind FHRR state using Query vectors
        Q_abs = torch.sqrt(Q_freq.real.pow(2) + Q_freq.imag.pow(2) + 1e-12)
        Q_norm_freq = Q_freq / Q_abs.to(dtype=torch.complex64)
        outputs_rec_freq = outputs_rec_freq * torch.conj(Q_norm_freq)
        
        # Inverse FFT to real domain
        fhrr_rec = torch.fft.irfft(outputs_rec_freq, n=D, dim=-1)
        
        # Retrieve exact context from Phantom Cache
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


def replace_all_mlps(model, tokenizer, avg_mlp_norms):
    log.info("\n[PHASE 2] REPLACE MLP WITH SYMPHONY ARCHITECTURE (PARALLEL HYBRID)")
    log.info("=" * 70)
    
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    
    log.info(f"  Replacing {num_layers} MLPs with SymphonyASHCLayers...")
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        orig_norm = avg_mlp_norms.get(layer_idx, 1.0)
        
        # Keep original MLP and freeze its parameters explicitly
        original_mlp = layer.mlp
        for param in original_mlp.parameters():
            param.requires_grad = False
            
        # Create Symphony ASH-C Layer (custom layers are initialized in float32 by default)
        ash_c_layer = SymphonyASHCLayer(hidden_dim, original_mlp, lambda_decay=LAMBDA_DECAY, scale_factor=orig_norm)
        # Move the layer to device without changing the dtype of submodules (keeping original_mlp in float16)
        ash_c_layer = ash_c_layer.to(device=DEVICE)
        
        # Replace the old MLP directly
        layer.mlp = ash_c_layer
        
        # Ensure only the deep custom layers and gate parameter are trainable
        for name, param in layer.mlp.named_parameters():
            if 'original_mlp' not in name:
                if layer_idx < 20:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    gc.collect()
    torch.cuda.empty_cache()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  [PHASE 2 COMPLETE] FHRR Memory layers injected (Parallel Hybrid). Trainable params: {trainable_params:,}")
    return model


# ============================================================
# DATASET SETUP
# ============================================================
class TitanReasoningDataset(Dataset):
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
                    inp, out = item.get('input', ''), item.get('output', '')
                    if inp and out:
                        self.samples.append((inp, out))
                except json.JSONDecodeError:
                    continue
        # Slice the dataset if it exceeds the max allowed budget
        if MAX_TRAIN_SAMPLES is not None and len(self.samples) > MAX_TRAIN_SAMPLES:
            log.info(f"  Slicing dataset from {len(self.samples)} down to {MAX_TRAIN_SAMPLES} to fit Kaggle time limit")
            self.samples = self.samples[:MAX_TRAIN_SAMPLES]
        log.info(f"  Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inp, out = self.samples[idx]
        text = f"Question: {inp}\nAnswer: {out}{self.tokenizer.eos_token}"
        encoded = self.tokenizer(text, max_length=self.max_len, truncation=True, padding=False, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# ============================================================
# SCHEDULER & UTILS
# ============================================================
def get_lr(step, total_steps, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

def save_checkpoint(model, optimizer, step, loss, epoch, is_epoch=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"epoch{epoch+1}" if is_epoch else f"step{step}"
    ckpt_path = os.path.join(OUTPUT_DIR, f"titan_symphony_{tag}.pt")
    
    if KEEP_ONLY_LATEST:
        for old_file in os.listdir(OUTPUT_DIR):
            if old_file.startswith("titan_symphony_") and old_file.endswith(".pt"):
                old_path = os.path.join(OUTPUT_DIR, old_file)
                try:
                    os.remove(old_path)
                except:
                    pass
    
    # Save all custom Symphony weights (trainable AND frozen ones) so layers 0-19 are preserved exactly
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
    log.info(f"  [SAVED] {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:.1f} MB, loss={loss:.4f})")

def test_generation(model, tokenizer):
    log.info("\n  --- POST-TRAINING GENERATION TEST ---")
    model.eval()
    test_prompts = ["Question: What is 15 * 7?\nAnswer:", "Question: If A > B and B > C, is A > C?\nAnswer:"]
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Reset memory state
        for name, module in model.named_modules():
            if isinstance(module, SymphonyASHCLayer):
                module.reset_state()
                
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        log.info(f"\n  Q: {prompt.split(chr(10))[0]}\n  A: {response[:150]}")
    model.train()


# ============================================================
# TRAINING LOOP
# ============================================================
def train():
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    print_config()
    model, tokenizer, breakdown = load_and_freeze()
    model = replace_all_mlps(model, tokenizer, breakdown['avg_mlp_norms'])
    
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        log.info("  Gradient checkpointing: ENABLED")
        
    if not os.path.exists(DATASET_PATH):
        log.error(f"  Dataset not found at {DATASET_PATH}. Please check your Kaggle input path!")
        return
        
    dataset = TitanReasoningDataset(DATASET_PATH, tokenizer, max_len=MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION
    total_steps = steps_per_epoch * EPOCHS
    
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adafactor(
        trainable_param_list,
        lr=LEARNING_RATE,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Initialize GradScaler for stable float16 training
    scaler = torch.amp.GradScaler('cuda') if DTYPE == torch.float16 else None
    
    log.info("\n[STEP 4] Starting training...")
    log.info("=" * 70)
    
    global_step = 0
    start_epoch = 0
    best_loss = float('inf')
    
    if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
        log.info(f"Resuming training from checkpoint: {RESUME_FROM_CHECKPOINT}")
        checkpoint = torch.load(RESUME_FROM_CHECKPOINT, map_location=DEVICE)
        state_dict = checkpoint['memory_state_dict']
        
        # Surgical weight migration for SwiGLU Prism (shape mismatch: hidden_dim -> hidden_dim * 2)
        model_dict = model.state_dict()
        migrated_dict = {}
        for name, param in state_dict.items():
            mapped_name = name.replace(".mlp.fhrr.", ".mlp.")
            mapped_name = mapped_name.replace(".mlp.w1.weight", ".mlp.prism_w1.weight")
            mapped_name = mapped_name.replace(".mlp.w2.weight", ".mlp.prism_w2.weight")
            mapped_name = mapped_name.replace(".mlp.w3.weight", ".mlp.prism_w3.weight")
            if mapped_name in model_dict:
                if param.shape == model_dict[mapped_name].shape:
                    migrated_dict[mapped_name] = param
                else:
                    # Catch the SwiGLU Prism weights that doubled in size
                    if 'prism_w1.weight' in mapped_name or 'prism_w2.weight' in mapped_name:
                        log.info(f"  Surgically migrating weight for {mapped_name}: {param.shape} -> {model_dict[mapped_name].shape}")
                        new_param = model_dict[mapped_name].clone()
                        H = param.shape[1]
                        # First half (FHRR input branch) gets checkpoint weights
                        new_param[:, :H] = param
                        # Second half (Exact Cache input branch) is initialized to zero to prevent noise
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
    
    train_start = time.time()
    model.train()
    
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        epoch_loss = 0.0
        micro_step = 0
        accumulated_loss = 0.0
        epoch_tokens = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # Reset memory state for each batch to prevent gradient leaks
            for name, module in model.named_modules():
                if isinstance(module, SymphonyASHCLayer):
                    module.reset_state()
            
            with torch.amp.autocast('cuda', dtype=DTYPE):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                clm_loss = outputs.loss
                
                # Retrieve self-supervised contrastive alignment loss
                contrastive_loss = 0.0
                num_ash_layers = 0
                for name, module in model.named_modules():
                    if isinstance(module, SymphonyASHCLayer) and module.last_contrastive_loss is not None:
                        contrastive_loss += module.last_contrastive_loss
                        num_ash_layers += 1
                        
                if num_ash_layers > 0:
                    contrastive_loss = contrastive_loss / num_ash_layers
                    # Mix classification loss and contrastive alignment loss
                    total_loss = clm_loss + 0.05 * contrastive_loss
                else:
                    total_loss = clm_loss
                
                loss = total_loss / GRADIENT_ACCUMULATION
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulated_loss += loss.item()
            num_tokens = (labels != -100).sum().item()
            epoch_tokens += num_tokens
            micro_step += 1
            
            if micro_step % GRADIENT_ACCUMULATION == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_param_list, MAX_GRAD_NORM)
                    
                    current_lr = get_lr(global_step, total_steps, WARMUP_STEPS, LEARNING_RATE)
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr
                        
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_param_list, MAX_GRAD_NORM)
                    
                    current_lr = get_lr(global_step, total_steps, WARMUP_STEPS, LEARNING_RATE)
                    for pg in optimizer.param_groups:
                        pg['lr'] = current_lr
                        
                    optimizer.step()
                    
                optimizer.zero_grad()
                
                global_step += 1
                epoch_loss += accumulated_loss
                
                if global_step % 10 == 0 or global_step == 1:
                    tokens_per_sec = epoch_tokens / max(time.time() - epoch_start, 0.1)
                    log.info(f"  Step {global_step:4d}/{total_steps} | Loss: {accumulated_loss:.4f} | LR: {current_lr:.2e} | Tok/s: {tokens_per_sec:.0f}")
                
                if SAVE_CHECKPOINTS and global_step % SAVE_INTERVAL == 0:
                    save_checkpoint(model, optimizer, global_step, accumulated_loss, epoch)
                
                if accumulated_loss < best_loss:
                    best_loss = accumulated_loss
                    
                accumulated_loss = 0.0
                
        avg_epoch_loss = epoch_loss / max(steps_per_epoch, 1)
        log.info(f"\n  Epoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f} | Time: {time.time() - epoch_start:.1f}s")
        save_checkpoint(model, optimizer, global_step, avg_epoch_loss, epoch, is_epoch=True)
    
    log.info("\n" + "=" * 70)
    log.info("TRAINING COMPLETE - EXPORTING WEIGHTS...")
    log.info("=" * 70)
    
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # Save Symphony weights
    memory_state = {
        name: param.data.to(DTYPE).cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    weights_path = os.path.join(EXPORT_DIR, "titan_symphony_weights.pt")
    torch.save(memory_state, weights_path)
    
    # Save Titan metadata
    meta = {
        "engine": "Titan Engine - Symphony Architecture",
        "source_model": SOURCE_MODEL,
        "frozen": ["attention", "embeddings", "norms", "lm_head"],
        "trained": ["mlp (FHRRMemoryLayer)"],
        "lambda_decay": LAMBDA_DECAY
    }
    with open(os.path.join(EXPORT_DIR, "titan_symphony_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    log.info(f"  Weights exported to: {weights_path}")
    
    test_generation(model, tokenizer)


if __name__ == "__main__":
    train()
