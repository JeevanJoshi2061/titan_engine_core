# Titan Engine: Symphony Architecture

**Symphony** is an experimental, constant-memory $O(1)$ sequence modeling engine. It replaces the traditional linear-scaling Key-Value (KV) Cache with a dual-system architecture: a continuous **Fractional Holographic Reduced Representation (FHRR)** for dense history compression, and a discrete **Coordinate-based Pointer Network (HEP-DNA)** for exact token retrieval.

This repository contains the official PyTorch implementation, training pipelines, and evaluation benchmarks for the Symphony architecture.

---

## 1. Introduction & Motivation
The primary bottleneck of modern Large Language Models is the quadratic attention mechanism and the linearly scaling KV Cache, which makes processing infinite-length contexts physically impossible on standard hardware. While State Space Models (SSMs) like Mamba achieve $O(1)$ memory, they suffer from "lossy compression," failing at exact retrieval tasks like "Needle in a Haystack" (NIAH).

**The Symphony Solution:** 
Symphony bridges this gap by splitting memory into two pathways:
1. **Symphony ASH-C (Active Selective Holographic-Compression):** Compresses historical context into fixed-size circular convolution matrices via FHRR. The memory state remains mathematically bounded (constant VRAM) regardless of sequence length.
2. **Symphony HEP-DNA (Holographic Exact Pointer):** A recurrent pointer network that avoids vector storage entirely. It maintains a lightweight queue of integer token IDs and uses the FHRR frequency domain to calculate positional probabilities. Using a zero-shot NTK extrapolation mechanism and a `scatter_add_` vocabulary injection, it executes 100% exact copy-paste retrievals over massive contexts.

---

## 2. Base Model Compatibility & Retraining Requirements
*(Addressing integration capabilities)*

Symphony is designed as a **plug-and-play architectural injection** rather than a standalone foundation model.
- **Compatibility:** It can theoretically be injected into the MLP layers of any standard decoder-only Transformer. The current implementation is validated on **Qwen-1.5B/7B**.
- **Retraining Requirements:** You **do not** need to pre-train a model from scratch. The base model's self-attention and MLP weights are heavily frozen. The training process only updates the newly injected `MemLayer` and `PointerNet`.
- **Inference Scaling:** Because the FHRR state is $O(1)$ and the PointerNet utilizes NTK (Neural Tangent Kernel) positional rescaling, the model achieves **Zero-Shot Length Extrapolation**. Once trained at a context length of 43k tokens to resolve geometric superposition noise, the architecture can theoretically process 100k+ tokens during inference without requiring additional fine-tuning.

---

## 3. Technical Implementation Details
To implement or reproduce this architecture confidently, note the following core mechanics found in `core/titan_hep_dna.py` and `core/titan_ash_c_architecture.py`:

* **FHRR Superposition:** History is maintained as a complex tensor `[Batch, Hidden_Dim]`. New tokens are bound to their positional coordinates using complex multiplication and superimposed (added) into the fixed-size state.
* **Coordinate Extraction:** During generation, the current query is conjugated and multiplied against the FHRR state. An Inverse FFT (`irfft`) extracts the positional distribution map.
* **The "Ditto Copy" Protocol:** Instead of generating floating-point vectors for output, the PointerNet calculates the exact historical position index, retrieves the original token integer ID, and projects the probability directly into the final vocabulary distribution tensor using PyTorch's `scatter_add_`. This completely prevents semantic degradation.

---

## 4. Pipeline & Reproduction Steps

### Environment Setup
```bash
pip install torch transformers accelerate bitsandbytes matplotlib numpy
```

### Multi-Phase Training
The Symphony engine is calibrated through targeted phases to prevent catastrophic forgetting (lobotomization) of the base model. The training logic is consolidated into the following primary scripts:

1. **Dataset Generation:** Generates the structured reasoning and needle-in-a-haystack data.
   ```bash
   python training/build_universal_dataset.py
   ```
2. **Phase 4 (Architectural Alignment):** A monolithic script that loads the base model, freezes pre-trained weights to preserve world knowledge, injects the Symphony MemLayers, and performs short-context (512 tokens) alignment training to teach the MLP layers how to utilize the FFT memory math via contrastive loss.
   ```bash
   python training/titan_phase4_kaggle_train_fft_math.py
   ```
3. **Phase 7 (HySparse Long-Range):** Stretches the context to 43,000+ tokens to train the PointerNet to resolve dense geometric noise.
   ```bash
   python training/titan_phase7_hysparse_training.py
   ```

---

## 5. Methodical Validation & Experiments
To ensure the base model is not lobotomized by the memory injection, we run multiple rigorous evaluations (`evaluation/` directory):

1. **Needle-in-a-Haystack (NIAH) Grid (`plot_nih_grid.py`):** Proves 100% exact retrieval of API keys and passcodes over 43,000+ tokens.
2. **Perplexity (PPL) Preservation (`titan_eval_ppl.py`):** Validates that general linguistic capabilities remain intact. The FHRR compression introduces an acceptable ~15% PPL penalty while delivering a 4.6x speedup.
3. **OOM Survival Stress Test (`titan_stress_test_oom_survival.py`):** Empirically proves the $O(1)$ scaling by profiling VRAM usage over escalating sequence lengths.
4. **General Knowledge Integrity (`titan_birthday_test.py`):** Verifies that the frozen base model retains its pre-trained factual knowledge (e.g., historical dates, coding syntax).

---

## 6. Related Work & Acknowledgements
This architecture builds upon and synthesizes several foundational concepts in sequence modeling:
- **Vector Symbolic Architectures (VSAs) & FHRR:** Plate (1995) introduced Holographic Reduced Representations. Recent work by researchers like Edward Raff has explored applying these concepts to modern deep learning.
- **Pointer Networks:** Vinyals et al. (2015) introduced the concept of pointing to input sequences rather than predicting from a fixed vocabulary.
- **Position Extrapolation:** YaRN (Peng et al., 2023) and NTK-Aware RoPE scaling heavily influenced Symphony's zero-shot coordinate scaling capabilities.
- **State Space Models:** Mamba (Gu & Dao, 2023) provided the inspiration for bounded memory footprint sequence processing.

---

## 7. Academic Citation
For full mathematical and architectural proofs, please refer to our publication:
> **Symphony: Constant-Memory Sequence Modeling via Holographic Recurrence and Coordinate Pointer Networks**  
> Jeevan Joshi (2026). Zenodo.  
> DOI: [10.5281/zenodo.20566771](https://doi.org/10.5281/zenodo.20566771)

### BibTeX
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
Open-source under the MIT License.
