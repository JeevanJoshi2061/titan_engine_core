import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import random
import string
import time
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "../core")))

from titan_ash_c_architecture import SymphonyASHCLayer
from titan_hep_dna import SymphonyHEPDNALayer
from titan_config import SOURCE_MODEL, DEVICE, DTYPE

PHASE4_WEIGHTS = r"E:\titan Engine new\Results\titan_symphony_weights.pt"
PHASE5_WEIGHTS = r"E:\titan Engine new\checkpoints_phase5\titan_hepdna_final.pt"
OUTPUT_DIR = r"E:\titan Engine new\checkpoints_phase6"

BATCH_SIZE = 1
LEARNING_RATE = 5e-4
MAX_STEPS = 1200

SCALE_SCHEDULE = [
    (400,  1024),
    (800,  2048),
    (1200, 4096),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

class LongRangeCopyDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len=1024):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.fillers = [
            "The system architecture requires strict validation protocols for all incoming requests. ",
            "According to the documentation on page 42, the integration setup is complete and verified. ",
            "Network timeout occurred during the standard initialization phase of the service mesh. ",
            "Please ensure that the firewall rules are updated to allow incoming traffic on port 443. ",
            "The user authentication flow uses OAuth 2.0 with JWT tokens for session management. ",
            "Data encryption at rest is handled by AES-256 standard protocols across all storage nodes. ",
            "Memory allocation limits were exceeded during the load testing phase of deployment. ",
            "Kubernetes pod deployment status is currently pending resource allocation from the cluster. ",
            "The database migration script completed successfully with zero schema conflicts detected. ",
            "Automated health checks are running every 30 seconds across all microservice endpoints. ",
            "The CDN cache invalidation propagated to all 47 edge nodes within the expected TTL window. ",
            "Container orchestration logs indicate nominal CPU usage at 23% with 4.2GB memory footprint. ",
            "SSL certificate renewal is scheduled for next month. Current cert expires on 2026-07-15. ",
            "The API gateway rate limiter is configured to allow 1500 requests per minute per client. ",
            "Distributed tracing shows average request latency of 142ms across the service topology. ",
            "The message queue consumer lag is within acceptable thresholds at 12 messages behind head. ",
        ]
        
    def _generate_random_code(self, length=16):
        chars = string.ascii_uppercase + string.digits
        segments = []
        remaining = length
        while remaining > 0:
            seg_len = min(random.choice([3, 4, 5]), remaining)
            segments.append(''.join(random.choice(chars) for _ in range(seg_len)))
            remaining -= seg_len
        return '-'.join(segments)
        
    def __iter__(self):
        while True:
            code_type = random.choice([
                "Verification Code", "API Key", "Transaction ID", 
                "Access Token", "Security Hash", "Session Key",
                "Recovery Code", "License Key"
            ])
            secret_code = self._generate_random_code(random.randint(16, 28))
            
            system_wrap = "<|im_start|>system\nYou are a precise AI assistant with perfect memory.<|im_end|>\n"
            user_start = "<|im_start|>user\nHere is the complete system log:\n\n"
            needle = f"\n[CRITICAL SECURITY ALERT] The {code_type} is: {secret_code}\n"
            user_end = f"\n\nExtract the exact {code_type} from the log above.<|im_end|>\n"
            assistant_response = f"<|im_start|>assistant\nThe exact {code_type} is: {secret_code}<|im_end|>"
            
            overhead_text = system_wrap + user_start + needle + user_end + assistant_response
            overhead_ids = self.tokenizer.encode(overhead_text, add_special_tokens=False)
            
            filler_budget = self.seq_len - len(overhead_ids) - 10
            if filler_budget <= 0:
                filler_budget = 64
            
            prefix_fillers = []
            suffix_fillers = []
            current_filler_tokens = 0
            
            needle_depth = random.uniform(0.3, 0.9)
            prefix_budget = int(filler_budget * needle_depth)
            
            while True:
                sentence = random.choice(self.fillers)
                sent_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
                if current_filler_tokens + len(sent_ids) > prefix_budget:
                    break
                prefix_fillers.append(sentence)
                current_filler_tokens += len(sent_ids)
                
            while True:
                sentence = random.choice(self.fillers)
                sent_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
                if current_filler_tokens + len(sent_ids) > filler_budget:
                    break
                suffix_fillers.append(sentence)
                current_filler_tokens += len(sent_ids)
                
            prefix = "".join(prefix_fillers)
            suffix = "".join(suffix_fillers)
            
            prompt_text = (
                f"{system_wrap}{user_start}"
                f"{prefix}{needle}{suffix}"
                f"{user_end}{assistant_response}"
            )
            
            encoded = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"][0]
            labels = input_ids.clone()
            
            assistant_token = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
            
            start_idx = -1
            for i in range(len(input_ids) - len(assistant_token)):
                if input_ids[i:i+len(assistant_token)].tolist() == assistant_token:
                    start_idx = i + len(assistant_token)
                    break
                    
            if start_idx == -1:
                continue
            else:
                labels[:start_idx] = -100
                labels[input_ids == self.tokenizer.pad_token_id] = -100
                
            yield {
                "input_ids": input_ids,
                "labels": labels
            }

def prepare_model():
    log.info("Loading model...")
    device = torch.device(DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL,
        torch_dtype=DTYPE,
        device_map=DEVICE
    )
    
    hidden_dim = model.config.hidden_size
    
    for idx in range(20, 28):
        original_mlp = model.model.layers[idx].mlp
        symphony_layer = SymphonyASHCLayer(hidden_dim, original_mlp)
        symphony_layer = symphony_layer.to(device=device, dtype=torch.bfloat16)
        model.model.layers[idx].mlp = symphony_layer
        
    log.info(f"Loading Phase 4 weights from {PHASE4_WEIGHTS}...")
    state_dict = torch.load(PHASE4_WEIGHTS, map_location=device, weights_only=True)
    custom_state = {k: v for k, v in state_dict.items() if any(x in k for x in ["mlp.router", "mlp.prism", "mlp.gate", "mlp.key_proj", "mlp.query_proj", "mlp.value_proj"])}
    model.load_state_dict(custom_state, strict=False)
    
    max_training_len = SCALE_SCHEDULE[-1][1]
    hep_dna = SymphonyHEPDNALayer(
        hidden_dim=hidden_dim, 
        vocab_size=model.config.vocab_size,
        max_seq_len=max_training_len,
        use_triton=False
    ).to(device=device, dtype=torch.bfloat16)
    
    log.info(f"Loading Phase 5 pointer checkpoint: {PHASE5_WEIGHTS}")
    pointer_state = torch.load(PHASE5_WEIGHTS, map_location=device, weights_only=True)
    if "pos_embeddings" in pointer_state:
        del pointer_state["pos_embeddings"]
    hep_dna.load_state_dict(pointer_state, strict=False)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in hep_dna.parameters():
        param.requires_grad = True
        
    return model, tokenizer, hep_dna, device

def train():
    model, tokenizer, hep_dna, device = prepare_model()
    optimizer = torch.optim.AdamW(hep_dna.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    model.eval()
    hep_dna.train()
    
    step = 0
    current_scale_idx = 0
    current_seq_len = SCALE_SCHEDULE[0][1]
    
    dataset = LongRangeCopyDataset(tokenizer, seq_len=current_seq_len)
    dataloader = iter(DataLoader(dataset, batch_size=BATCH_SIZE))
    
    log.info(f"Starting Phase 6 long-range calibration...")
    
    start_time = time.time()
    
    while step < MAX_STEPS:
        if current_scale_idx < len(SCALE_SCHEDULE) - 1:
            if step >= SCALE_SCHEDULE[current_scale_idx][0]:
                current_scale_idx += 1
                current_seq_len = SCALE_SCHEDULE[current_scale_idx][1]
                dataset = LongRangeCopyDataset(tokenizer, seq_len=current_seq_len)
                dataloader = iter(DataLoader(dataset, batch_size=BATCH_SIZE))
                log.info(f"Stepping to SEQ_LEN = {current_seq_len}")
        
        try:
            batch = next(dataloader)
        except StopIteration:
            dataloader = iter(DataLoader(dataset, batch_size=BATCH_SIZE))
            batch = next(dataloader)
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        for layer in model.model.layers:
            if hasattr(layer.mlp, "reset_state"):
                layer.mlp.reset_state()
                
        with torch.no_grad():
            outputs = model.model(input_ids)
            hidden_states = outputs.last_hidden_state
            lm_logits = model.lm_head(hidden_states)
            
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_input_ids = input_ids[..., :-1].contiguous()
        shift_lm_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        valid_mask = (shift_labels != -100)
        if not valid_mask.any():
            continue
            
        loss_tensor, gate_tensor = hep_dna(
            hidden_states=shift_hidden,
            input_ids=shift_input_ids,
            lm_logits=shift_lm_logits,
            target_ids=shift_labels
        )
        
        masked_loss = loss_tensor[valid_mask]
        loss = masked_loss.mean()
        mean_gate = gate_tensor.squeeze(-1)[valid_mask].mean().item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hep_dna.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (BATCH_SIZE * current_seq_len * 10) / elapsed
            log.info(f"Step {step:04d}/{MAX_STEPS} | SEQ={current_seq_len} | Loss: {loss.item():.4f} | Gate: {mean_gate:.4f} | Tok/s: {tok_per_sec:.0f}")
            start_time = time.time()
            
        if step % 200 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"titan_hepdna_p6_step{step}.pt")
            torch.save(hep_dna.state_dict(), ckpt_path)
            log.info(f"Saved checkpoint to {ckpt_path}")
            
    final_path = os.path.join(OUTPUT_DIR, "titan_hepdna_longrange_final.pt")
    torch.save(hep_dna.state_dict(), final_path)
    log.info(f"Phase 6 Complete. Final weights: {final_path}")

if __name__ == "__main__":
    train()
