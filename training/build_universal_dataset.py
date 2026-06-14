import json
import os
import sys
import random

output_file = "universal_titan_reasoning.jsonl"

try:
    from datasets import load_dataset
except ImportError:
    print("error: datasets library not installed. run: pip install datasets tqdm apache-beam")
    sys.exit(1)

from tqdm import tqdm

def clean_and_split(text, max_in=1500, max_out=750):
    if len(text) < (max_in + 150):
        return None
        
    split_idx = text.find(".", max_in)
    if split_idx == -1 or split_idx > (max_in + 300):
        split_idx = max_in
    else:
        split_idx += 1
        
    inp_text = text[:split_idx].strip()
    remaining = text[split_idx:].strip()
    
    if not inp_text or not remaining:
        return None
        
    out_end = remaining.find(".", max_out)
    if out_end == -1 or out_end > (max_out + 200):
        out_text = remaining[:max_out].strip()
    else:
        out_text = remaining[:out_end + 1].strip()
        
    if len(out_text) < 100:
        return None
        
    return {
        "input": f"Context: {inp_text}\n\nTask: Continue writing the next technical section or summary of the above context.",
        "output": out_text
    }

def main():
    print("starting universal dataset creation...")
    samples = []
    
    print("fetching aya multilingual...")
    try:
        aya = load_dataset("CohereForAI/aya_dataset", split="train", streaming=True)
        count = 0
        pbar = tqdm(total=30000)
        for item in aya:
            inp = item.get("inputs", "").strip()
            out = item.get("targets", "").strip()
            if 150 <= len(inp) <= 1600 and 100 <= len(out) <= 800:
                samples.append({"input": inp, "output": out})
                count += 1
                pbar.update(1)
                if count >= 30000:
                    break
        pbar.close()
    except Exception as e:
        print("error loading aya:", e)

    print("fetching code instructions (alpaca 120k)...")
    try:
        code = load_dataset("iamtarun/code_instructions_120k_alpaca", split="train", streaming=True)
        count = 0
        pbar = tqdm(total=30000)
        for item in code:
            instruction = item.get("instruction", "").strip()
            input_val = item.get("input", "").strip()
            out = item.get("output", "").strip()
            
            if input_val:
                inp = f"{instruction}\nInput Context:\n{input_val}"
            else:
                inp = instruction
                
            if 150 <= len(inp) <= 1600 and 100 <= len(out) <= 800:
                samples.append({"input": inp, "output": out})
                count += 1
                pbar.update(1)
                if count >= 30000:
                    break
        pbar.close()
    except Exception as e:
        print("error loading code instructions:", e)

    print("fetching arxiv...")
    try:
        arxiv = load_dataset("ccdv/arxiv-summarization", "document", split="train", streaming=True)
        count = 0
        pbar = tqdm(total=20000)
        for item in arxiv:
            doc = item.get("article", "")
            res = clean_and_split(doc)
            if res:
                samples.append(res)
                count += 1
                pbar.update(1)
                if count >= 20000:
                    break
        pbar.close()
    except Exception as e:
        print("error loading arxiv:", e)

    print("fetching wikipedia...")
    wiki_langs = [("en", 8000), ("hi", 4000), ("es", 2700), ("fr", 2700), ("de", 2600)]
    pbar = tqdm(total=20000)
    for lang, limit in wiki_langs:
        try:
            wiki = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
            count = 0
            for item in wiki:
                text = item.get("text", "")
                res = clean_and_split(text)
                if res:
                    samples.append(res)
                    count += 1
                    pbar.update(1)
                    if count >= limit:
                        break
        except Exception as e:
            print(f"error loading wiki {lang}: {e}")
    pbar.close()

    print(f"shuffling and saving {len(samples)} samples...")
    random.shuffle(samples)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
            
    size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"done! file size: {size:.2f} MB")

if __name__ == "__main__":
    main()
