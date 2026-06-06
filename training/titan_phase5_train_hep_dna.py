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
OUTPUT_DIR = r"E:\titan Engine new\checkpoints_phase5"

BATCH_SIZE = 2
SEQ_LEN = 512
LEARNING_RATE = 2e-3
MAX_STEPS = 1500

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

class CopyPasteDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len=512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.fillers = [
            "The system architecture requires strict validation protocols. ",
            "According to the documentation on page 42, the integration setup is complete. ",
            "Network timeout occurred during the standard initialization phase. ",
            "Please ensure that the firewall rules are updated to allow incoming traffic. ",
            "The user authentication flow uses OAuth 2.0 with JWT tokens. ",
            "Data encryption at rest is handled by AES-256 standard protocols. ",
            "Memory allocation limits were exceeded during the stress test. ",
            "Kubernetes pod deployment status is currently pending resource allocation. "
        ]
        
    def _generate_random_code(self, length=12):
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choice(chars) for _ in range(length))
        
    def __iter__(self):
        while True:
            code_type = random.choice(["Verification Code", "API Key", "Transaction ID", "Access Token"])
            secret_code = self._generate_random_code(random.randint(12, 24))
            
            num_fillers = random.randint(3, 10)
            prefix = "".join(random.choices(self.fillers, k=num_fillers))
            middle = f"\n[CRITICAL INFO] The {code_type} is: {secret_code}\n"
            num_fillers_after = random.randint(3, 10)
            suffix = "".join(random.choices(self.fillers, k=num_fillers_after))
            
            prompt_text = f"<|im_start|>system\nYou are a precise AI assistant.<|im_end|>\n<|im_start|>user\nHere is the system log:\n\n{prefix}{middle}{suffix}\n\nWhat is the exact {code_type} from the log?<|im_end|>\n<|im_start|>assistant\nThe exact {code_type} is: {secret_code}<|im_end|>"
            
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
                labels[:] = -100
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
    custom_layer_indices = range(20, 28)
    
    for idx in custom_layer_indices:
        original_mlp = model.model.layers[idx].mlp
        symphony_layer = SymphonyASHCLayer(hidden_dim, original_mlp)
        symphony_layer = symphony_layer.to(device=device, dtype=torch.bfloat16)
        model.model.layers[idx].mlp = symphony_layer
        
    log.info(f"Loading Phase 4 weights from {PHASE4_WEIGHTS}...")
    state_dict = torch.load(PHASE4_WEIGHTS, map_location=device, weights_only=True)
    custom_state = {k: v for k, v in state_dict.items() if any(x in k for x in ["mlp.router", "mlp.prism", "mlp.gate", "mlp.key_proj", "mlp.query_proj", "mlp.value_proj"])}
    
    model.load_state_dict(custom_state, strict=False)
    
    hep_dna = SymphonyHEPDNALayer(
        hidden_dim=hidden_dim, 
        vocab_size=model.config.vocab_size,
        max_seq_len=SEQ_LEN,
        use_triton=True
    ).to(device=device, dtype=torch.bfloat16)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in hep_dna.parameters():
        param.requires_grad = True
        
    return model, tokenizer, hep_dna, device

def train():
    model, tokenizer, hep_dna, device = prepare_model()
    
    dataset = CopyPasteDataset(tokenizer, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    optimizer = torch.optim.AdamW(hep_dna.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    model.eval()
    hep_dna.train()
    
    step = 0
    start_time = time.time()
    
    for batch in dataloader:
        if step >= MAX_STEPS:
            break
            
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
            tok_per_sec = (BATCH_SIZE * SEQ_LEN * 10) / elapsed
            log.info(f"Step {step:04d}/{MAX_STEPS} | Loss: {loss.item():.4f} | Gate Avg: {mean_gate:.4f} | Tok/s: {tok_per_sec:.0f}")
            start_time = time.time()
            
        if step % 500 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"titan_hepdna_step{step}.pt")
            torch.save(hep_dna.state_dict(), ckpt_path)
            log.info(f"Checkpoint saved to {ckpt_path}")
            
    final_path = os.path.join(OUTPUT_DIR, "titan_hepdna_final.pt")
    torch.save(hep_dna.state_dict(), final_path)
    log.info(f"Training Complete. Final weights saved to: {final_path}")

if __name__ == "__main__":
    train()
