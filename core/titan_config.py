import torch

model_path = r"E:\Brain\Qwen2.5-1.5B-Instruct"
out_dir = r"E:\titan Engine new\checkpoints"
data_dir = r"E:\titan Engine new\datasets"
export_dir = r"E:\titan Engine new\titan_model"

hidden_size = 1536
mlp_size = 8960
n_layers = 28
new_mlp = 4096

use_swiglu = True

bs = 1
grad_acc = 16
lr = 1e-5
wd = 0.01
seq_len = 512
epochs = 1
warmup = 300
clip_norm = 1.0

dev = "cuda" if torch.cuda.is_available() else "cpu"
fp = torch.bfloat16
use_grad_ckpt = True

cal_samples = 20
init_scale = 1.0

SOURCE_MODEL = model_path
DEVICE = dev
DTYPE = fp


def show_config():
    print("=" * 70)
    print("TITAN ENGINE CONFIGURATION")
    print("=" * 70)
    print(f"  model:         {model_path}")
    print(f"  device:        {dev}")
    print(f"  dtype:         {fp}")
    print(f"  mlp dim:       {mlp_size}")
    print(f"  new mlp dim:   {new_mlp}")
    print(f"  mlp ratio:     {new_mlp/mlp_size*100:.1f}%")
    print(f"  batch:         {bs} x {grad_acc} = {bs * grad_acc}")
    print(f"  lr:            {lr}")
    print(f"  grad ckpt:     {use_grad_ckpt}")
    print("=" * 70)


if __name__ == "__main__":
    show_config()
