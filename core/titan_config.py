import torch

# Paths
SOURCE_MODEL = r"E:\Brain\Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = r"E:\titan Engine new\checkpoints"
DATASET_DIR = r"E:\titan Engine new\datasets"
EXPORT_DIR = r"E:\titan Engine new\titan_model"

# Model Dimensions
ORIGINAL_HIDDEN_DIM = 1536
ORIGINAL_INTERMEDIATE = 8960
NUM_LAYERS = 28
NEW_INTERMEDIATE = 4096

# Architecture Flags
USE_SWIGLU = True

# Training Hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 512
EPOCHS = 1
WARMUP_STEPS = 300
MAX_GRAD_NORM = 1.0

# Hardware Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
USE_GRADIENT_CHECKPOINTING = True

# Calibration Settings
CALIBRATION_SAMPLES = 20
INIT_SCALE_FACTOR = 1.0

def print_config():
    print("=" * 70)
    print("TITAN ENGINE CONFIGURATION")
    print("=" * 70)
    print(f"  Source Model:       {SOURCE_MODEL}")
    print(f"  Device:             {DEVICE}")
    print(f"  Dtype:              {DTYPE}")
    print(f"  Original MLP dim:   {ORIGINAL_INTERMEDIATE}")
    print(f"  New MLP dim:        {NEW_INTERMEDIATE}")
    print(f"  MLP size ratio:     {NEW_INTERMEDIATE/ORIGINAL_INTERMEDIATE*100:.1f}%")
    print(f"  Batch size:         {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate:      {LEARNING_RATE}")
    print(f"  Grad checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
