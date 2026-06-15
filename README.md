# Titan Engine: Symphony Architecture

An experimental, constant-memory sequence modeling engine. 

Symphony completely bypasses the traditional linear Key-Value Cache. It delivers a true constant memory footprint ($O(1)$) during inference, keeping VRAM stable regardless of context length.

---

## 🚀 The Dual-System Core

* **Symphony ASH-C:** Active Selective Holographic-Compression.
  * Folds historical tokens into a fixed-size complex matrix.
  * Uses FHRR (Fractional Holographic Reduced Representation) frequency domain math.
* **Symphony HEP-DNA:** Holographic Exact Pointer. 
  * Eliminates floating-point vector caching. 
  * Tracks positions via a lightweight integer queue of token IDs.
  * Executes 100% exact retrieval via direct vocabulary `scatter_add_` injection.

---

## ⚙️ How It Works (The Mechanics)

* **FHRR Superposition:** History is maintained as a `[Batch, Hidden_Dim]` complex tensor. New tokens are superimposed. No linear VRAM growth.
* **Coordinate Extraction:** Conjugated query is multiplied by the FHRR state. Inverse FFT (`irfft`) extracts the physical token position.
* **Ditto Copy Protocol:** PointerNet grabs the integer ID from the queue. It projects the probability directly into the final vocabulary tensor. 100% exact retrieval. Zero semantic loss.

---

## 🔧 Base Model Integration

Symphony is a plug-and-play architectural graft.

* **Compatibility:** Validated on Qwen-1.5B/7B. Drops cleanly into the MLP layer of standard decoder-only Transformers.
* **No Pre-training:** Base model attention and MLPs remain heavily frozen. We only train the injected `MemLayer` and `PointerNet`.
* **Zero-Shot Extrapolation:** Trained on 43k tokens to resolve geometric superposition noise. NTK (Neural Tangent Kernel) positional rescaling handles 100k+ tokens out-of-the-box. No retraining needed.

---

## 🏃‍♂️ Execution Pipeline

**Setup:**
```bash
pip install torch transformers accelerate bitsandbytes matplotlib numpy
```

**Training Phases:**
The engine is calibrated to prevent catastrophic forgetting (lobotomization) of the frozen base model.

1. **Build Dataset:** Generates structured needle-in-a-haystack reasoning data.
   ```bash
   python training/build_universal_dataset.py
   ```
2. **Phase 4 (Alignment):** Short-context (512 tokens). Freezes base weights, injects MemLayers. Teaches MLPs the FFT memory math via contrastive loss.
   ```bash
   python training/titan_phase4_kaggle_train_fft_math.py
   ```
3. **Phase 7 (HySparse):** Long-context (43,000+ tokens). Trains the PointerNet to resolve dense geometric noise.
   ```bash
   python training/titan_phase7_hysparse_training.py
   ```

---

## 📊 Benchmarks & Proof

Rigorous evaluations to prove the base model is not lobotomized (`evaluation/`):

* **OOM Survival Test:** Flat VRAM usage curve over escalating lengths. Empirically proves $O(1)$ scaling. (`titan_stress_test_oom_survival.py`)
* **Needle-in-a-Haystack (NIAH):** 100% exact retrieval of passcodes across 43k+ tokens. (`plot_nih_grid.py`)
* **Perplexity Preservation:** FHRR compression causes an acceptable ~15% PPL hit, but grants a 4.6x speedup. (`titan_eval_ppl.py`)
* **Knowledge Integrity:** Base model retains pre-trained facts and coding syntax. (`titan_birthday_test.py`)

---

## 📚 Origins & Academic Citation

Built upon foundational sequence modeling concepts:
* **FHRR / VSAs:** Plate (1995), Edward Raff.
* **Pointer Networks:** Vinyals et al. (2015).
* **Position Extrapolation:** YaRN / NTK-Aware RoPE scaling.
* **Bounded Memory:** State Space Models (Mamba).

**Cite our Zenodo Publication:**
> Jeevan Joshi (2026). *Symphony: Constant-Memory Sequence Modeling via Holographic Recurrence and Coordinate Pointer Networks.*  
> DOI: [10.5281/zenodo.20566771](https://doi.org/10.5281/zenodo.20566771)

```bibtex
@misc{joshi2026symphony,
  author       = {Joshi, Jeevan},
  title        = {Symphony: Constant-Memory Sequence Modeling via Holographic Recurrence and Coordinate Pointer Networks},
  month        = jun,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20566771},
  url          = {https://doi.org/10.5281/zenodo.20566771}
}
```

## License
MIT License.
