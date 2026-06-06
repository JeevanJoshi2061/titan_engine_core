import json
import os
import random
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "../core")))

try:
    from titan_config import DATASET_DIR
except ImportError:
    DATASET_DIR = "./datasets"

TASK_LIMITS = {
    'futoshiki': 200,
    'kakurasu': 200,
    'tsumego': 200,
    'knight_swap': 200,
    'word_ladder': 200,
    'sudoku': 200,
    'survo': 200,
    'cryptarithm': 200,
    'zebra_puzzles': 200,
    'sokoban': 200,
    'rubiks_cube': 500,
    'rush_hour': 500,
    'maze': 2000,
    'tower_of_hanoi': 2000,
    'game_of_life': 2000,
    'game_of_life_halting': 2000,
    'n_queens': 2000,
    'shortest_path': 2000,
    'course_schedule': 2000,
    'family_relationships': 2000
}

def download_hf_logic():
    try:
        from datasets import load_dataset
        ds = load_dataset("hivaze/LOGIC-701", "en", split="train")
        return [{"input": i.get("problem_statement", "").strip(), "output": f"{i.get('solution', '').strip()} End.", "domain": "hf_logic"} for i in ds if i.get("problem_statement") and i.get("solution")]
    except Exception as e:
        print(f"HF Download failed or library missing. Installing datasets...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            from datasets import load_dataset
            ds = load_dataset("hivaze/LOGIC-701", "en", split="train")
            return [{"input": i.get("problem_statement", "").strip(), "output": f"{i.get('solution', '').strip()} End.", "domain": "hf_logic"} for i in ds if i.get("problem_statement") and i.get("solution")]
        except Exception as err:
            print(f"HuggingFace dataset download unavailable: {err}. Falling back to synthetic logic data.")
            return []

def ensure_reasoning_gym():
    try:
        import reasoning_gym
    except ImportError:
        print("reasoning-gym is not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "reasoning-gym"])
        except Exception as e:
            print(f"Failed to install reasoning-gym: {e}.")
            sys.exit(1)

def build_production_dataset(total=1000000):
    print("Generating Titan logic dataset...")
    ensure_reasoning_gym()
    
    from reasoning_gym.factory import DATASETS
    import reasoning_gym
    
    all_tasks = sorted([t for t in DATASETS.keys() if t != 'composite'])
    print(f"Loaded {len(all_tasks)} logical and algorithmic reasoning tasks from reasoning-gym.")
    
    hf_data = download_hf_logic()
    print(f"Loaded {len(hf_data)} HF Logic Samples.")
    
    remaining_total = max(total - len(hf_data), total // 2)
    preset_sum = sum(TASK_LIMITS.get(t, 0) for t in all_tasks if t in TASK_LIMITS)
    unrestricted_tasks = [t for t in all_tasks if t not in TASK_LIMITS]
    
    unrestricted_total = max(remaining_total - preset_sum, 0)
    samples_per_unrestricted_task = unrestricted_total // len(unrestricted_tasks) if unrestricted_tasks else 0
    
    dataset = []
    dataset.extend(hf_data)
    
    print(f"Generating procedural data from {len(all_tasks)} tasks...")
    for idx, task in enumerate(all_tasks):
        try:
            size = TASK_LIMITS.get(task, samples_per_unrestricted_task)
            if size <= 0:
                continue
                
            print(f"Generating {size} samples for '{task}'...")
            data = reasoning_gym.create_dataset(task, size=size, seed=random.randint(1, 1000000))
            for item in data:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if q and a:
                    dataset.append({
                        "input": q,
                        "output": f"{a} End.",
                        "domain": task
                    })
        except Exception as e:
            print(f"Failed to generate task '{task}': {e}")
            
    random.shuffle(dataset)
    dataset = dataset[:total]
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    path = os.path.join(DATASET_DIR, "titan_reasoning.jsonl")
    
    print(f"Writing {len(dataset)} samples to {path}...")
    with open(path, 'w', encoding='utf-8') as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(dataset)} samples to {path} ({os.path.getsize(path)/1e6:.2f} MB)")

if __name__ == "__main__":
    build_production_dataset(1000000)
