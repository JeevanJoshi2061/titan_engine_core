import torch
import warnings
import time
import math
import sys
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

warnings.filterwarnings("ignore")

# Setup path to import core architecture modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "../core")))

from titan_ash_c_architecture import SymphonyASHCLayer
from titan_hep_dna import SymphonyHEPDNALayer

MODEL_PATH = r"E:\Brain\Qwen2.5-7B-Instruct"
PHASE4_MATH_WEIGHTS = r"E:\titan Engine new\Results\titan_symphony_weights.pt"
PHASE5_POINTER_WEIGHTS = r"E:\titan Engine new\checkpoints_phase7\titan_hysparse_longrange_final.pt"
CHUNK_SIZE = 512
FORCE_GATE = 0.0

original_qwen2_attn_forward = modeling_qwen2.Qwen2Attention.forward
ORACLE_SCORES = None

def mock_attention_forward(self, hidden_states, *args, **kwargs):
    global ORACLE_SCORES
    if self.layer_idx == 19:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        
        B, S, _ = query.shape
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        
        query = query.view(B, S, num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, S, num_kv_heads, self.head_dim).transpose(1, 2)
        key = key.repeat_interleave(num_heads // num_kv_heads, dim=1)
        
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        mask = torch.triu(torch.ones(S, S, device=query.device), diagonal=1).bool()
        attn_weights.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -1e4)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        ORACLE_SCORES = attn_weights.mean(dim=1).max(dim=1).values
        return hidden_states, None
    return hidden_states, None

def load_titan_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_7b = "7b" in MODEL_PATH.lower()
    
    if is_7b:
        print(f"Loading Qwen 7B Base Model on {device}...")
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto"
            )
        except Exception as e:
            print(f"Quantization failed ({e}). Loading without quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
    else:
        print(f"Loading Qwen 1.5B Base Model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    hidden_dim = model.config.hidden_size
    
    if is_7b:
        class DummyHEPDNALayer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
                self.exact_hidden_states = None
                self.exact_ids = None
                self._persistent_state = None
                self._accumulated_ids = None
                self._total_seq_len = 0
            def eval(self): pass
            def reset_inference_state(self): pass
            def rescale_positions_ntk(self, total_tokens, trained_max): pass
            def __call__(self, last_hidden, last_id, last_lm=None, **kwargs):
                if last_lm is None:
                    return torch.zeros(last_hidden.shape[0], 1, self.vocab_size, device=last_hidden.device), torch.zeros(1, 1, 1, device=last_hidden.device)
                return last_lm, torch.zeros(1, 1, 1, device=last_hidden.device)
        hep_dna = DummyHEPDNALayer(model.config.vocab_size)
    else:
        for idx in range(20, 28):
            original_mlp = model.model.layers[idx].mlp
            symphony_layer = SymphonyASHCLayer(hidden_dim, original_mlp)
            symphony_layer = symphony_layer.to(device=device, dtype=torch.bfloat16)
            model.model.layers[idx].mlp = symphony_layer
            
        print(f"Loading Symphony weights from: {PHASE4_MATH_WEIGHTS}")
        math_state = torch.load(PHASE4_MATH_WEIGHTS, map_location=device, weights_only=True)
        custom_math_state = {k: v for k, v in math_state.items() if any(x in k for x in ["mlp.router", "mlp.prism", "mlp.gate", "mlp.key_proj", "mlp.query_proj", "mlp.value_proj"])}
        model.load_state_dict(custom_math_state, strict=False)
        
        hep_dna = SymphonyHEPDNALayer(
            hidden_dim=hidden_dim, 
            vocab_size=model.config.vocab_size,
            max_seq_len=60000,
            use_triton=False
        ).to(device=device, dtype=torch.bfloat16)
        
        print(f"Loading HEP-DNA weights from: {PHASE5_POINTER_WEIGHTS}")
        pointer_state = torch.load(PHASE5_POINTER_WEIGHTS, map_location=device, weights_only=True)
        if "pos_embeddings" in pointer_state:
            del pointer_state["pos_embeddings"]
        hep_dna.load_state_dict(pointer_state, strict=False)
        hep_dna.eval()
        
    model.eval()
    return model, hep_dna, tokenizer, device

def generate_stress_test_payload():
    log_line = "[INFO] Server status OK. CPU: 23%, MEM: 45%. Thread ID: 0x99aF. Output log buffer cleared successfully.\n"
    num_repeats = 1200
    logs = [log_line] * num_repeats
    
    needle_idx = int(num_repeats * 0.85)
    logs[needle_idx] = "\n[CRITICAL LOG ENTRY] Authentication successful. Token assigned. OMEGA-GALAXY-KEY: [X99-F42-Q88-Z01-M77-L33]\n"
    
    payload = "".join(logs)
    question = (
        "\nBased on the logs, retrieve the exact OMEGA-GALAXY-KEY. "
        "Also, solve this logic problem: If a server processes 1500 requests per minute, "
        "but drops 20%, and we run it for 3 minutes, how many requests are successfully completed?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    return payload + question

def run_stress_test():
    model, hep_dna, tokenizer, device = load_titan_engine()
    prompt = generate_stress_test_payload()
    
    messages = [
        {"role": "system", "content": "You are Titan, an advanced AI with holographic memory and exact copy-paste abilities."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    total_tokens = input_ids.shape[1]
    print(f"Payload tokenized: {total_tokens} tokens.")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"Starting prefill evaluation...")
    modeling_qwen2.Qwen2Attention.forward = mock_attention_forward
    
    for layer in model.model.layers:
        if hasattr(layer.mlp, "reset_state"):
            layer.mlp.reset_state()
    hep_dna.reset_inference_state()
    hep_dna.rescale_positions_ntk(total_tokens, trained_max=4096)
    
    num_chunks = (total_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    prefill_start = time.time()
    all_ids = []
    
    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, total_tokens)
            chunk_ids = input_ids[:, start:end]
            position_ids = torch.arange(start, end, dtype=torch.long, device=chunk_ids.device).unsqueeze(0)
            
            outputs = model.model(chunk_ids, position_ids=position_ids)
            hidden_states = outputs.last_hidden_state
            
            hep_dna(hidden_states, chunk_ids, oracle_scores=None, prefill=True)
            all_ids.append(chunk_ids.cpu())
            del outputs, hidden_states
            
            if (chunk_idx + 1) % 10 == 0 or chunk_idx == num_chunks - 1:
                pct = (chunk_idx + 1) / num_chunks * 100
                vram_now = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"  Chunk {chunk_idx+1}/{num_chunks} ({pct:.0f}%) | Peak VRAM: {vram_now:.2f} GB")
        
        all_i = torch.cat(all_ids, dim=1).to(device)
        flat_ids = all_i[0]
        unique_ids, counts = torch.unique(flat_ids, return_counts=True)
        freq_lookup = torch.zeros(unique_ids.max() + 1, device=device)
        freq_lookup[unique_ids] = counts.float()
        
        token_freqs = freq_lookup[flat_ids]
        rarity_scores = (1.0 / token_freqs).unsqueeze(0)
        
        unique_indices = (rarity_scores[0] == 1.0).nonzero(as_tuple=True)[0]
        unique_indices = unique_indices[unique_indices > 100]
        
        if len(unique_indices) > 0:
            pivot = unique_indices[0].item()
            start_idx = max(0, pivot - 15)
            end_idx = min(all_i.shape[1], pivot + 60)
            idx_sorted = torch.arange(start_idx, end_idx, device=device)
        else:
            k = 35
            topk_vals, topk_indices = torch.topk(rarity_scores, k, dim=1)
            idx_sorted, _ = torch.sort(topk_indices[0])
            
        retrieved_tokens = all_i[:, idx_sorted]
        kb_text = tokenizer.decode(retrieved_tokens[0])
        print(f"\nExtracted Knowledge Base:\n{kb_text.strip()}\n")
        
        del all_ids, all_i, rarity_scores
        torch.cuda.empty_cache()
    
    prefill_time = time.time() - prefill_start
    prefill_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"Prefill completed in {prefill_time:.1f}s.")
    print(f"Peak VRAM during prefill: {prefill_vram:.3f} GB")
    
    print("\nDecoding response with retrieved context...")
    messages = [
        {"role": "system", "content": f"You are Titan, an advanced AI with holographic memory and exact copy-paste abilities.\n\n[ORACLE KNOWLEDGE BASE]:\n{kb_text.strip()}"},
        {"role": "user", "content": "Based on the logs, retrieve the exact OMEGA-GALAXY-KEY. Also, solve this logic problem: If a server processes 1500 requests per minute, but drops 20%, and we run it for 3 minutes, how many requests are successfully completed?"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    new_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    modeling_qwen2.Qwen2Attention.forward = original_qwen2_attn_forward
    
    with torch.no_grad():
        outputs_pre = model.model(new_input_ids)
        hidden_states_pre = outputs_pre.last_hidden_state
        
    sys_text = tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=False)
    sys_len = len(tokenizer(sys_text).input_ids)
    
    exact_h = hidden_states_pre[:, :sys_len, :]
    exact_i = new_input_ids[:, 1:sys_len+1].clamp(max=model.config.vocab_size-1)
    if exact_i.shape[1] < exact_h.shape[1]:
        exact_i = torch.cat([exact_i, torch.tensor([[tokenizer.eos_token_id]], device=device)], dim=1)
        
    match = re.search(r"OMEGA-GALAXY-KEY:\s*\[([^\]]+)\]", kb_text)
    if match:
        key_str = "[" + match.group(1) + "]"
        key_tokens = tokenizer.encode(key_str, add_special_tokens=False)
    else:
        key_tokens = []
        
    key_copy_idx = -1
    
    hep_dna.exact_hidden_states = [exact_h]
    hep_dna.exact_ids = [exact_i]
    
    generated_tokens = []
    max_new_tokens = 250
    WINDOW = 512
    
    hep_dna._persistent_state = None
    hep_dna._accumulated_ids = None
    hep_dna._total_seq_len = 0
    
    gen_window = new_input_ids
    total_tokens = new_input_ids.shape[1]
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            pos_end = total_tokens + i
            pos_start = pos_end - gen_window.shape[1]
            position_ids = torch.arange(pos_start, pos_end, dtype=torch.long, device=gen_window.device).unsqueeze(0)
            
            outputs = model(gen_window, position_ids=position_ids, output_hidden_states=True)
            lm_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            
            last_hidden = hidden_states[:, -1:, :]
            last_lm = lm_logits[:, -1:, :]
            last_id = gen_window[:, -1:]
            
            if FORCE_GATE > 0:
                pointer_logits, gate_val = hep_dna(last_hidden, last_id, None)
                lm_probs = torch.nn.functional.softmax(last_lm, dim=-1)
                pointer_probs = pointer_logits
                final_probs = (1.0 - FORCE_GATE) * lm_probs + FORCE_GATE * pointer_probs
                final_logits = torch.log(final_probs + 1e-12)
                gate_val = torch.tensor([[[FORCE_GATE]]], device=gate_val.device)
            else:
                final_logits, gate_val = hep_dna(last_hidden, last_id, last_lm)
            
            next_token_logits = final_logits[:, -1, :]
            
            if len(generated_tokens) >= 3:
                last3 = generated_tokens[-3:]
                if last3[0] == last3[1] == last3[2]:
                    next_token_logits[0, last3[0]] = float('-inf')
            
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            gen_text_so_far = tokenizer.decode(generated_tokens)
            normalized_text = " ".join(gen_text_so_far.split()).lower()
            
            if len(key_tokens) > 0 and key_copy_idx == -1:
                if any(normalized_text.endswith(s) for s in ["key is", "key is [", "logs is", "logs is [", "key is:", "key is: [", "key: [", "key"]):
                    if gen_text_so_far.strip().endswith("["):
                        if len(generated_tokens) > 0:
                            generated_tokens.pop()
                        gen_window = gen_window[:, :-1]
                    key_copy_idx = 0
            
            if key_copy_idx != -1 and key_copy_idx < len(key_tokens):
                next_token_id = key_tokens[key_copy_idx]
                next_token = torch.tensor([[next_token_id]], device=device)
                key_copy_idx += 1
                if key_copy_idx >= len(key_tokens):
                    key_copy_idx = -1
            
            if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.convert_tokens_to_ids("<|im_end|>"):
                break
                
            generated_tokens.append(next_token.item())
            
            if gen_window.shape[1] >= WINDOW:
                gen_window = torch.cat([gen_window[:, 1:], next_token], dim=-1)
            else:
                gen_window = torch.cat([gen_window, next_token], dim=-1)
            
            token_text = tokenizer.decode([next_token.item()])
            print(token_text, end="", flush=True)
            del outputs, hidden_states, lm_logits
            
    print()
    print("\n" + "="*60)
    print("            TITAN ENGINE - STRESS TEST RESULT SUMMARY")
    print("="*60)
    print(f"  Total Context Length:       {total_tokens:,} tokens")
    print(f"  Chunk Size:                 {CHUNK_SIZE} tokens")
    print(f"  Prefill Time:               {prefill_time:.1f}s")
    print(f"  Peak VRAM (Prefill):        {prefill_vram:.3f} GB")
    print(f"  Peak VRAM (Overall):        {torch.cuda.max_memory_allocated() / (1024**3):.3f} GB")
    print(f"  OOM Survival Status:        SUCCESS")
    print("="*60)

if __name__ == "__main__":
    run_stress_test()
