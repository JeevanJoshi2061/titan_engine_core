import torch
import time
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

MODEL_PATH = r"E:\Brain\Qwen2.5-7B-Instruct"
CHUNK_SIZE = 512
WINDOW = 512

CONTEXT_LENGTHS = [10000, 25000, 45000]
DEPTHS = [10, 50, 90]

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
        
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(S, S, device=query.device), diagonal=1).bool()
        attn_weights.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -1e4)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        ORACLE_SCORES = attn_weights.mean(dim=1).max(dim=1).values
        return hidden_states, None
    return hidden_states, None

def load_qwen_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading Qwen 7B in 4-bit...")
    
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer, device

def generate_payload(tokenizer, total_tokens_target, depth_pct):
    log_line = "[INFO] Server status OK. CPU: 23%, MEM: 45%. Thread ID: 0x99aF. Output log buffer cleared successfully.\n"
    base_len = len(tokenizer.encode(log_line))
    num_repeats = max(1, total_tokens_target // base_len)
    
    logs = [log_line] * num_repeats
    needle_idx = int(num_repeats * (depth_pct / 100.0))
    needle_text = "\n[CRITICAL LOG ENTRY] Authentication successful. Token assigned. OMEGA-GALAXY-KEY: [X99-F42-Q88-Z01-M77-L33]\n"
    logs[needle_idx] = needle_text
    
    payload = "".join(logs)
    input_ids = tokenizer.encode(payload)
    if len(input_ids) > total_tokens_target:
        input_ids = input_ids[:total_tokens_target]
    
    return tokenizer.decode(input_ids)

def run_trial(model, tokenizer, device, context_length, depth):
    print(f"\nRunning Trial: Length={context_length} tokens, Depth={depth}%")
    
    prompt = generate_payload(tokenizer, context_length, depth)
    messages = [
        {"role": "system", "content": "You are Titan, an advanced AI with holographic memory and exact copy-paste abilities."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    total_tokens = input_ids.shape[1]
    
    modeling_qwen2.Qwen2Attention.forward = mock_attention_forward
    
    num_chunks = (total_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    all_ids = []
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, total_tokens)
            chunk_ids = input_ids[:, start:end]
            position_ids = torch.arange(start, end, dtype=torch.long, device=chunk_ids.device).unsqueeze(0)
            
            outputs = model.model(chunk_ids, position_ids=position_ids)
            all_ids.append(chunk_ids.cpu())
            del outputs
            
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
        
    prefill_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    modeling_qwen2.Qwen2Attention.forward = original_qwen2_attn_forward
    
    match = re.search(r"OMEGA-GALAXY-KEY:\s*\[([^\]]+)\]", kb_text)
    success = False
    if match and match.group(1) == "X99-F42-Q88-Z01-M77-L33":
        success = True
        
    print(f"Trial Result: Success={success}, Time={prefill_time:.1f}s, Peak VRAM={peak_vram:.3f} GB")
    return success, prefill_time, peak_vram

def main():
    model, tokenizer, device = load_qwen_model()
    
    results = {}
    for length in CONTEXT_LENGTHS:
        for depth in DEPTHS:
            success, duration, vram = run_trial(model, tokenizer, device, length, depth)
            results[(length, depth)] = {
                "success": success,
                "time": duration,
                "vram": vram
            }
            
    print("\n" + "="*50)
    print("           NIH EVALUATION RESULTS SUMMARY")
    print("="*50)
    for length in CONTEXT_LENGTHS:
        for depth in DEPTHS:
            res = results[(length, depth)]
            status = "SUCCESS" if res["success"] else "FAILED"
            print(f"Length: {length:5d} tok | Depth: {depth:2d}% | Status: {status} | Time: {res['time']:.1f}s | VRAM: {res['vram']:.2f} GB")
            
    print("\nGenerating visual heatmap plot...")
    x = np.arange(len(DEPTHS))
    y = np.arange(len(CONTEXT_LENGTHS))
    grid_success = np.zeros((len(CONTEXT_LENGTHS), len(DEPTHS)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, l in enumerate(CONTEXT_LENGTHS):
        for j, d in enumerate(DEPTHS):
            res = results[(l, d)]
            grid_success[i, j] = 1.0 if res["success"] else 0.0
            
            status_txt = "SUCCESS" if res["success"] else "FAILED"
            label = f"{status_txt}\nTime: {res['time']:.1f}s\nVRAM: {res['vram']:.2f} GB"
            
            cell_color = '#d4edda' if res["success"] else '#f8d7da'
            edge_color = '#c3e6cb' if res["success"] else '#f5c6cb'
            
            rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, 
                                 facecolor=cell_color, edgecolor=edge_color, 
                                 linewidth=1.5)
            ax.add_patch(rect)
            
            ax.text(j, i, label, ha='center', va='center', 
                    color='#155724' if res["success"] else '#721c24',
                    fontweight='bold', fontsize=10)
            
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}% Depth" for d in DEPTHS], fontsize=11, fontweight='semibold')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l:,}\nTokens" for l in CONTEXT_LENGTHS], fontsize=11, fontweight='semibold')
    
    ax.set_xlim(-0.5, len(DEPTHS) - 0.5)
    ax.set_ylim(-0.5, len(CONTEXT_LENGTHS) - 0.5)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.set_title("Symphony Engine: Needle-in-a-Haystack (NIH) Grid\n(Base Model: Qwen2.5-7B-Instruct 4-bit on RTX 4060 8GB)", 
                 fontsize=13, fontweight='bold', pad=20, color='#2c3e50')
    
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath(os.path.join(script_dir, "../images/nih_grid.png"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visual grid saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
