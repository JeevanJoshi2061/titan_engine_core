import torch
import time
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = r"E:\Brain\Qwen2.5-7B-Instruct"
EVAL_TOKENS_LIMIT = 8192

# Resolve data path relative to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.abspath(os.path.join(script_dir, "../../genesis_distilled_FINAL.txt"))

def load_qwen_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_7b = "7b" in MODEL_PATH.lower()
    
    print(f"Loading model from {MODEL_PATH}...")
    if is_7b:
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer, device

def evaluate_ppl(model, encodings, device, max_length, stride=512):
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / seq_len)
    
    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    return ppl.item(), peak_vram, elapsed

def main():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found!")
        return
        
    model, tokenizer, device = load_qwen_model()
    
    print(f"Reading and tokenizing {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
        
    encodings = tokenizer(text, return_tensors="pt")
    
    if encodings.input_ids.size(1) > EVAL_TOKENS_LIMIT:
        encodings.input_ids = encodings.input_ids[:, :EVAL_TOKENS_LIMIT]
        encodings.attention_mask = encodings.attention_mask[:, :EVAL_TOKENS_LIMIT]
        
    actual_tokens = encodings.input_ids.size(1)
    print(f"Tokenized sequence length: {actual_tokens} tokens.")
    
    print("\nRunning Perplexity Benchmarks...")
    
    print("\nEvaluating Full Context Baseline (Window = 2048)...")
    ppl_full, vram_full, time_full = evaluate_ppl(model, encodings, device, max_length=2048, stride=512)
    print(f"PPL: {ppl_full:.4f} | Peak VRAM: {vram_full:.3f} GB | Time: {time_full:.1f}s")
    
    print("\nEvaluating Constrained Local Window (Window = 512)...")
    ppl_local, vram_local, time_local = evaluate_ppl(model, encodings, device, max_length=512, stride=256)
    print(f"PPL: {ppl_local:.4f} | Peak VRAM: {vram_local:.3f} GB | Time: {time_local:.1f}s")
    
    print("\n" + "="*60)
    print("             TITAN ENGINE - PERPLEXITY BENCHMARK REPORT")
    print("="*60)
    print(f"  Model Evaluated:        {MODEL_PATH}")
    print(f"  Evaluation Dataset:     {TEST_DATA_PATH}")
    print(f"  Tokens Evaluated:       {actual_tokens}")
    print("-"*60)
    print(f"  Full Context (Window=2048):")
    print(f"    - Perplexity:         {ppl_full:.4f}")
    print(f"    - Peak VRAM:          {vram_full:.3f} GB")
    print(f"    - Evaluation Time:    {time_full:.1f} seconds")
    print(f"  Constrained (Window=512):")
    print(f"    - Perplexity:         {ppl_local:.4f}")
    print(f"    - Peak VRAM:          {vram_local:.3f} GB")
    print(f"    - Evaluation Time:    {time_local:.1f} seconds")
    print("-"*60)
    
    ppl_diff = ((ppl_local - ppl_full) / ppl_full) * 100
    vram_saved = vram_full - vram_local
    print(f"  Perplexity Difference:  +{ppl_diff:.2f}%")
    print(f"  VRAM Saved in Local:    {vram_saved:.3f} GB")
    print("="*60)

if __name__ == "__main__":
    main()
