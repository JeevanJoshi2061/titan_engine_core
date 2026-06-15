# Titan Engine: Symphony Architecture

📄 **[Read the Full Paper (PDF)](Latest_symphony_paper.pdf)**

An experimental sequence modeling engine that runs on constant memory.

Symphony replaces the standard Transformer KV-Cache. Instead of eating up more VRAM as your context grows, it uses a fixed, constant amount of memory.

---

## Core Modules

* **ASH-C (Memory Core):** Compresses historical context into a fixed-size matrix. VRAM stays flat no matter how long the input text is.
* **HEP-DNA (Pointer Core):** Tracks token positions using a simple list of integer IDs instead of huge vectors. This allows 100% exact word-for-word copy-paste from past context.

---

## Key Features

* **Plug and Play:** Drops directly into the MLP layers of standard models. Currently tested and working on Qwen-1.5B and 7B.
* **No Full Retraining:** The base model stays completely frozen. We only train the newly injected memory layers.
* **Infinite Context:** Trained at 43k tokens to handle noise. Thanks to NTK rescaling, it can test at 100k+ tokens during inference without extra fine-tuning.

---

## Setup & How to Run

```bash
pip install torch transformers accelerate bitsandbytes matplotlib numpy
```

### Training Steps

1. **Generate Data:** Create the long-range reasoning tasks.
```bash
python training/build_universal_dataset.py
```

2. **Phase 4 (Alignment):** Train on short context (512 tokens) so the model learns the basic memory math.
```bash
python training/titan_phase4_kaggle_train_fft_math.py
```

3. **Phase 7 (Long Range):** Stretch the context to 43k+ tokens to fine-tune the pointer network.
```bash
python training/titan_phase7_hysparse_training.py
```

---

## Benchmarks & Tests

Look inside the `evaluation/` folder to verify the model's performance:

* `titan_stress_test_oom_survival.py`: Checks VRAM stability over escalating lengths.
* `plot_nih_grid.py`: Tests the ability to find specific passcodes at 43k context length.
* `titan_eval_ppl.py`: Measures standard language accuracy vs speed.
* `titan_birthday_test.py`: Ensures the frozen base model hasn't forgotten its original facts.

---

## Info & Citation

Based on concepts from FHRR/VSAs, Pointer Networks, and State Space Models.

📄 **[Read the Latest Technical Paper (PDF)](Latest_symphony_paper.pdf)**  
*(Updated with learned routing and dual-stream FHRR mechanics)*

---

**Previous Draft (Archived):**

Jeevan Joshi (2026). *Symphony: Constant-Memory Sequence Modeling via Holographic Recurrence and Coordinate Pointer Networks.*  
DOI: [10.5281/zenodo.20566771](https://doi.org/10.5281/zenodo.20566771)

```bibtex
@misc{joshi2026symphony,
  author       = {Joshi, Jeevan},
  title        = {Symphony: Constant-Memory Sequence Modeling via Holographic Recurrence and Coordinate Pointer Networks},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20566771}
}
```

## License

MIT License.
