import matplotlib.pyplot as plt
import os

def create_vram_comparison_plot(output_dir):
    labels = ['Symphony (7B)', 'Standard Qwen (7B)']
    vram_usage = [5.49, 48.0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, vram_usage, color=['#2ecc71', '#e74c3c'], width=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        label_text = f'{yval:.2f} GB' if yval < 48 else 'OOM (>48 GB)'
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, label_text, ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Peak VRAM (GB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Footprint: 43k Token Context Prefill\n(Lower is Better)', fontsize=14, fontweight='bold', pad=20)
    
    ax.axhline(y=8.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1.4, 8.5, 'RTX 4060 Limit (8GB)', color='black', fontsize=10, fontstyle='italic')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 55)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "vram_benchmark.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved VRAM benchmark to: {output_path}")

def create_ppl_comparison_plot(output_dir):
    labels = ['Full Context\n(Window=2048)', 'Symphony Local\n(Window=512)']
    ppl_scores = [3.96, 4.58]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(labels, ppl_scores, color=['#3498db', '#f39c12'], width=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Perplexity (PPL)', fontsize=12, fontweight='bold')
    ax.set_title('Language Modeling Quality\n(Lower is Better)', fontsize=14, fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 6)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "ppl_benchmark.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved PPL benchmark to: {output_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.abspath(os.path.join(script_dir, "../images"))
    os.makedirs(images_dir, exist_ok=True)
    
    create_vram_comparison_plot(images_dir)
    create_ppl_comparison_plot(images_dir)
