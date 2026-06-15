import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Adafactor
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
import math
import gc
import random
import sys

print_fn = print

model_path = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/1.5b-instruct/1"
data_path = "/kaggle/input/datasets/tsfmaster/142mb-universal-datasets/universal_titan_reasoning.jsonl"
ckpt_dir = "/kaggle/working/checkpoints"
export_dir = "/kaggle/working/titan_model"

hidden_size = 1536
mlp_size = 8960
n_layers = 28

bs = 1
grad_acc = 16
lr = 1e-4
wd = 0.01
seq_len = 512
epochs = 1
warmup = 300
clip_norm = 1.0
max_samples = None

save_ckpts = True
save_every = 500
keep_latest = True
resume_path = None

dev = "cuda" if torch.cuda.is_available() else "cpu"
fp = torch.float16
use_grad_ckpt = True
decay = 0.99

def show_config():
    print_fn(f"model: {model_path}")
    print_fn(f"data: {data_path}")
    print_fn(f"device: {dev}, dtype: {fp}")
    print_fn(f"hidden: {hidden_size}, mlp: {mlp_size}")
    print_fn(f"batch: {bs} x {grad_acc} = {bs * grad_acc}")
    print_fn(f"lr: {lr}, grad_ckpt: {use_grad_ckpt}")


def load_and_freeze():
    print_fn("\n--- phase 1: load and freeze ---")

    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print_fn(f"loading tokenizer from {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)

    print_fn(f"loading model in {fp}")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=fp, device_map=dev, attn_implementation="sdpa"
    )

    nl = mdl.config.num_hidden_layers

    ct = 0
    for n, p in mdl.named_parameters():
        p.requires_grad = False
        ct += 1
    print_fn(f"frozen {ct}/{ct} params")

    print_fn("capturing mlp norms")
    raw_norms = {}
    hooks = []

    def make_hook(idx):
        def fn(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            val = o.float().norm(dim=-1).mean().item()
            if idx not in raw_norms:
                raw_norms[idx] = []
            raw_norms[idx].append(val)
        return fn

    for i in range(nl):
        h = mdl.model.layers[i].mlp.register_forward_hook(make_hook(i))
        hooks.append(h)

    cal_texts = []
    if os.path.exists(data_path):
        try:
            pool = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    inp, out = item.get('input', ''), item.get('output', '')
                    if inp and out:
                        pool.append(f"Question: {inp}\nAnswer: {out}")
                        if len(pool) >= 1000:
                            break
            if pool:
                cal_texts = random.sample(pool, min(50, len(pool)))
            print_fn(f"loaded {len(cal_texts)} calibration samples")
        except Exception as e:
            print_fn(f"calibration load failed: {e}")

    if not cal_texts:
        fallbacks = [
            "How can we optimize this convolutional neural network architecture to reduce latency by 30% without sacrificing accuracy? Consider quantization, pruning, and architectural changes.",
            "Solve the Diophantine equation 7x + 11y = 35",
            "Analyze the eigenvalues and eigenvectors of this 3x3 covariance matrix: [[1, 0.5, 0], [0.5, 2, 0.3], [0, 0.3, 3]]",
            "Derive the Van der Waals equation of state from first principles using statistical mechanics and virial expansion.",
            "Implement a parallel FFT algorithm for a dataset of 1 million floating-point numbers using CUDA.",
            "What are the constraints on the parameters of a stationary Gaussian process, and how does this relate to kernel selection?",
            "If a Markov chain has transition matrix P, what is its steady-state distribution, and how would you compute it using eigenvalue decomposition?",
            "Prove or disprove: the product of two positive definite matrices is positive definite.",
            "Write a vectorized custom CUDA kernel for matrix multiplication in C++.",
            "Analyze the following server logs and find the memory leak: Traceback (most recent call last)...",
            "If matrix A has dimension 4x4 and rank 2, what is the dimension of its null space?",
            "Explain Albert Einstein's mass-energy equivalence principle from first principles."
        ]
        for p in fallbacks:
            txt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            cal_texts.append(txt)
        print_fn(f"using {len(cal_texts)} fallback prompts")

    with torch.no_grad():
        for txt in cal_texts:
            inputs = tok([txt], return_tensors="pt", truncation=True, max_length=512).to(dev)
            mdl(**inputs)

    for h in hooks:
        h.remove()

    avg_norms = {}
    for idx in range(nl):
        vals = raw_norms.get(idx, [1.0])
        avg_norms[idx] = sum(vals) / len(vals) if vals else 1.0

    print_fn("phase 1 done")
    return mdl, tok, avg_norms


class Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + self.eps)).to(x.dtype) * self.w


class Router(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def forward(self, x, temp=1.0, hard=True):
        logits = self.proj(x)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=temp, hard=hard)
        else:
            preds = torch.argmax(logits, dim=-1)
            weights = F.one_hot(preds, num_classes=2).to(x.dtype)
        return weights[:, :, 1]


class Cache(nn.Module):
    def __init__(self, dim, cap=2048):
        super().__init__()
        self.cap = cap
        self.dim = dim
        self.clear()

    def clear(self):
        self.keys = None
        self.values = None
        self.masks = None
        self.count = 0

    def add(self, new_k, new_v, selection):
        B, S, D = new_k.shape
        device = new_k.device

        bk = []
        bv = []
        mf = 0

        for b in range(B):
            idx = torch.nonzero(selection[b] > 0.5).squeeze(-1)
            k = new_k[b, idx]
            v = new_v[b, idx]
            bk.append(k)
            bv.append(v)
            if k.shape[0] > mf:
                mf = k.shape[0]

        if mf == 0:
            return None, None

        pk = torch.zeros(B, mf, D, device=device, dtype=new_k.dtype)
        pv = torch.zeros(B, mf, D, device=device, dtype=new_v.dtype)
        pm = torch.zeros(B, mf, device=device, dtype=torch.bool)

        for b in range(B):
            n = bk[b].shape[0]
            if n > 0:
                pk[b, :n] = bk[b]
                pv[b, :n] = bv[b]
                pm[b, :n] = True

        if self.keys is None:
            self.keys = pk
            self.values = pv
            self.masks = pm
        else:
            self.keys = torch.cat([self.keys, pk], dim=1)
            self.values = torch.cat([self.values, pv], dim=1)
            self.masks = torch.cat([self.masks, pm], dim=1)

        self.count = self.keys.shape[1]

        if self.count > self.cap:
            ov = self.count - self.cap
            self.keys = self.keys[:, ov:, :]
            self.values = self.values[:, ov:, :]
            self.masks = self.masks[:, ov:]
            self.count = self.cap

    def retrieve(self, query):
        if self.keys is None or self.keys.shape[1] == 0:
            return torch.zeros_like(query)

        scores = torch.matmul(query, self.keys.transpose(-2, -1)) / math.sqrt(self.dim)
        me = self.masks.unsqueeze(1)
        scores = scores.masked_fill(~me, -10000.0)
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.matmul(attn, self.values)


class MemLayer(nn.Module):
    def __init__(self, dim, orig_mlp, lam=0.99, scale=1.0):
        super().__init__()
        self.dim = dim
        self.orig_mlp = orig_mlp
        self.lam = lam
        self.scale = scale
        self.fdim = dim // 2 + 1

        self.router = Router(dim)
        self.cache = Cache(dim, cap=128)

        self.kp = nn.Linear(dim, dim, bias=False)
        self.vp = nn.Linear(dim, dim, bias=False)
        self.qp = nn.Linear(dim, dim, bias=False)

        self.pw1 = nn.Linear(dim * 2, dim, bias=False)
        self.pw2 = nn.Linear(dim * 2, dim, bias=False)
        self.pw3 = nn.Linear(dim, dim, bias=False)
        self.act = nn.SiLU()
        self.norm = Norm(dim)

        self.gate = nn.Parameter(torch.tensor([-10.0]))
        self.mem = None
        self.cl = None

        nn.init.kaiming_normal_(self.kp.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.vp.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.qp.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.pw1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.pw2.weight, nonlinearity='linear')
        nn.init.xavier_normal_(self.pw3.weight)

    def reset_state(self):
        self.mem = None
        self.cache.clear()
        self.cl = None

    def forward(self, x):
        if self.training:
            self.reset_state()

        B, S, D = x.shape
        device = x.device
        dtype = x.dtype

        base = self.orig_mlp(x)

        sel = self.router(x)

        k = self.kp(x)
        v = self.vp(x)
        q = self.qp(x)

        kf = k.float()
        vf = v.float()
        qf = q.float()

        self.cache.add(kf, vf, sel)

        kfr = torch.fft.rfft(kf, dim=-1)
        vfr = torch.fft.rfft(vf, dim=-1)
        qfr = torch.fft.rfft(qf, dim=-1)

        ka = torch.sqrt(kfr.real.pow(2) + kfr.imag.pow(2) + 1e-12)
        kn = kfr / ka.to(dtype=torch.complex64)

        bound = kn * vfr

        if self.training or S > 1 or self.mem is None or self.mem.shape[0] != B:
            state0 = torch.zeros(B, self.fdim, dtype=torch.complex64, device=device)
        else:
            state0 = self.mem

        ti = torch.arange(S, device=device).unsqueeze(1)
        ii = torch.arange(S, device=device).unsqueeze(0)
        pw = torch.clamp(ti - ii, min=0)
        msk = (ti - ii >= 0).float()
        W = ((self.lam ** pw) * msk).to(dtype=torch.complex64)

        sc = math.sqrt(1.0 - self.lam ** 2)
        rec = sc * torch.matmul(W, bound)

        steps = torch.arange(1, S + 1, device=device, dtype=torch.float32)
        df = (self.lam ** steps).unsqueeze(0).unsqueeze(2).to(dtype=torch.complex64)
        rec = rec + state0.unsqueeze(1) * df

        if not self.training:
            self.mem = rec[:, -1, :].detach()

        qa = torch.sqrt(qfr.real.pow(2) + qfr.imag.pow(2) + 1e-12)
        qn = qfr / qa.to(dtype=torch.complex64)
        rec = rec * torch.conj(qn)

        fhrr_out = torch.fft.irfft(rec, n=D, dim=-1)
        fhrr_out = torch.clamp(fhrr_out, -65000.0, 65000.0)

        exact_out = self.cache.retrieve(qf)

        combined = torch.cat([fhrr_out, exact_out], dim=-1).to(dtype)
        gated = self.pw1(combined) * self.act(self.pw2(combined))
        proj = self.pw3(gated)

        clean = self.norm(proj.float()) * (self.scale / math.sqrt(self.dim))

        if self.training:
            clean_f = clean.float()
            exact_f = exact_out.float()
            fhrr_f = fhrr_out.float()
            pd = F.pairwise_distance(clean_f, exact_f, p=2, eps=1e-6)
            nd = F.pairwise_distance(clean_f, fhrr_f, p=2, eps=1e-6)
            self.cl = torch.clamp(1.0 + pd - nd, min=0.0).mean()
        else:
            self.cl = None

        clean = clean.to(dtype)
        g = torch.sigmoid(self.gate).to(dtype)
        return base + g * clean


def inject_layers(mdl, tok, norms):
    print_fn("\n--- phase 2: inject memory layers ---")

    nl = mdl.config.num_hidden_layers
    dim = mdl.config.hidden_size

    print_fn(f"replacing {nl} mlps")

    for i in range(nl):
        layer = mdl.model.layers[i]
        orig = layer.mlp
        for p in orig.parameters():
            p.requires_grad = False

        mem = MemLayer(dim, orig, lam=decay, scale=norms.get(i, 1.0))
        mem = mem.to(device=dev)
        layer.mlp = mem

        for n, p in layer.mlp.named_parameters():
            if 'orig_mlp' not in n:
                p.requires_grad = True

    gc.collect()
    torch.cuda.empty_cache()

    tp = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print_fn(f"phase 2 done, trainable: {tp:,}")
    return mdl


class TrainData(Dataset):
    def __init__(self, path, tok, ml=512):
        self.tok = tok
        self.ml = ml
        self.data = []

        print_fn(f"loading data from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    inp, out = item.get('input', ''), item.get('output', '')
                    if inp and out:
                        self.data.append((inp, out))
                except json.JSONDecodeError:
                    continue
        if max_samples is not None and len(self.data) > max_samples:
            print_fn(f"slicing {len(self.data)} to {max_samples}")
            self.data = self.data[:max_samples]
        print_fn(f"loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, out = self.data[idx]
        text = f"Question: {inp}\nAnswer: {out}{self.tok.eos_token}"
        enc = self.tok(text, max_length=self.ml, truncation=True, padding=False, return_tensors='pt')
        ids = enc['input_ids'].squeeze(0)
        mask = enc['attention_mask'].squeeze(0)
        labels = ids.clone()
        labels[mask == 0] = -100
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def cosine_lr(step, total, wu, max_lr):
    if step < wu:
        return max_lr * step / max(wu, 1)
    prog = (step - wu) / max(total - wu, 1)
    return max_lr * 0.5 * (1 + math.cos(math.pi * prog))


def save_ckpt(mdl, opt, step, loss, epoch, is_epoch=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"epoch{epoch+1}" if is_epoch else f"step{step}"
    path = os.path.join(ckpt_dir, f"titan_symphony_{tag}.pt")

    if keep_latest:
        for f in os.listdir(ckpt_dir):
            if f.startswith("titan_symphony_") and f.endswith(".pt"):
                try:
                    os.remove(os.path.join(ckpt_dir, f))
                except:
                    pass

    weights = {
        n: p.data.cpu()
        for n, p in mdl.named_parameters()
        if (p.requires_grad or ('.mlp.' in n and 'orig_mlp' not in n))
    }

    torch.save({
        'step': step, 'epoch': epoch, 'loss': loss,
        'memory_state_dict': weights,
        'optimizer_state_dict': opt.state_dict()
    }, path)
    print_fn(f"saved {path} ({os.path.getsize(path)/1e6:.1f} MB, loss={loss:.4f})")


def test_gen(mdl, tok):
    print_fn("\n--- generation test ---")
    mdl.eval()
    prompts = ["Question: What is 15 * 7?\nAnswer:", "Question: If A > B and B > C, is A > C?\nAnswer:"]
    for p in prompts:
        inputs = tok(p, return_tensors="pt").to(dev)
        for n, m in mdl.named_modules():
            if isinstance(m, MemLayer):
                m.reset_state()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=fp):
                out = mdl.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True, pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print_fn(f"Q: {p.split(chr(10))[0]}")
        print_fn(f"A: {resp[:150]}")
    mdl.train()


def train():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    show_config()
    mdl, tok, norms = load_and_freeze()
    mdl = inject_layers(mdl, tok, norms)

    if use_grad_ckpt:
        mdl.gradient_checkpointing_enable()
        print_fn("gradient checkpointing on")

    if not os.path.exists(data_path):
        print_fn(f"dataset not found at {data_path}")
        return

    ds = TrainData(data_path, tok, ml=seq_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    spe = len(dl) // grad_acc
    total = spe * epochs

    params = [p for p in mdl.parameters() if p.requires_grad]
    opt = Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=wd)

    scaler = torch.amp.GradScaler('cuda') if fp == torch.float16 else None

    print_fn(f"\nstarting training, {total} steps total")

    gstep = 0
    start_ep = 0
    best = float('inf')

    cf = resume_path
    if not cf and os.path.exists(ckpt_dir):
        files = [f for f in os.listdir(ckpt_dir) if f.startswith("titan_symphony_") and f.endswith(".pt")]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
            cf = os.path.join(ckpt_dir, files[-1])

    if cf and os.path.exists(cf):
        print_fn(f"resuming from {cf}")
        ckpt = torch.load(cf, map_location=dev)
        sd = ckpt['memory_state_dict']

        md = mdl.state_dict()
        migrated = {}
        for name, param in sd.items():
            mn = name.replace(".mlp.fhrr.", ".mlp.")
            mn = mn.replace(".mlp.w1.weight", ".mlp.pw1.weight")
            mn = mn.replace(".mlp.w2.weight", ".mlp.pw2.weight")
            mn = mn.replace(".mlp.w3.weight", ".mlp.pw3.weight")
            mn = mn.replace("key_proj", "kp")
            mn = mn.replace("value_proj", "vp")
            mn = mn.replace("query_proj", "qp")
            mn = mn.replace("prism_w1", "pw1")
            mn = mn.replace("prism_w2", "pw2")
            mn = mn.replace("prism_w3", "pw3")
            mn = mn.replace("salience_proj", "proj")
            mn = mn.replace("phantom_cache", "cache")
            mn = mn.replace("m_state", "mem")
            mn = mn.replace("last_contrastive_loss", "cl")
            if mn in md:
                if param.shape == md[mn].shape:
                    migrated[mn] = param
                else:
                    if 'pw1.weight' in mn or 'pw2.weight' in mn:
                        print_fn(f"migrating {mn}: {param.shape} -> {md[mn].shape}")
                        np = md[mn].clone()
                        H = param.shape[1]
                        np[:, :H] = param
                        np[:, H:] = 0.0
                        migrated[mn] = np
                    else:
                        print_fn(f"skipping {mn}, shape mismatch")
            else:
                print_fn(f"key {name} not found in model")

        mdl.load_state_dict(migrated, strict=False)

        if 'optimizer_state_dict' in ckpt:
            try:
                opt.load_state_dict(ckpt['optimizer_state_dict'])
                print_fn("optimizer restored")
            except Exception as e:
                print_fn(f"optimizer restore failed: {e}")

        gstep = ckpt.get('step', 0)
        start_ep = ckpt.get('epoch', 0)
        best = ckpt.get('loss', float('inf'))
        print_fn(f"resumed at step {gstep}, epoch {start_ep}, loss {best:.4f}")

    t0 = time.time()
    mdl.train()

    for ep in range(start_ep, epochs):
        ep_start = time.time()
        ep_loss = 0.0
        ms = 0
        acc_loss = 0.0
        ep_toks = 0

        if ep == start_ep and gstep > 0:
            skip_n = (gstep % spe) * grad_acc
            if skip_n > 0:
                print_fn(f"fast-forwarding {skip_n} micro-steps")
        else:
            skip_n = 0

        for batch in dl:
            if ms < skip_n:
                ms += 1
                continue

            ids = batch['input_ids'].to(dev)
            mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            for n, m in mdl.named_modules():
                if isinstance(m, MemLayer):
                    m.reset_state()

            with torch.amp.autocast('cuda', dtype=fp):
                out = mdl(input_ids=ids, attention_mask=mask, labels=labels)
                clm = out.loss

                cl_total = 0.0
                cl_count = 0
                for n, m in mdl.named_modules():
                    if isinstance(m, MemLayer) and m.cl is not None:
                        cl_total += m.cl
                        cl_count += 1

                if cl_count > 0:
                    total_loss = clm + 0.05 * (cl_total / cl_count)
                else:
                    total_loss = clm

                loss = total_loss / grad_acc

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            lv = loss.item()
            if math.isnan(lv) or math.isinf(lv):
                opt.zero_grad()
                acc_loss = 0.0
                ms += 1
                continue

            acc_loss += lv
            ntok = (labels != -100).sum().item()
            ep_toks += ntok
            ms += 1

            if ms % grad_acc == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, clip_norm)
                    cur_lr = cosine_lr(gstep, total, warmup, lr)
                    for pg in opt.param_groups:
                        pg['lr'] = cur_lr
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, clip_norm)
                    cur_lr = cosine_lr(gstep, total, warmup, lr)
                    for pg in opt.param_groups:
                        pg['lr'] = cur_lr
                    opt.step()

                opt.zero_grad()
                gstep += 1
                ep_loss += acc_loss

                if gstep % 10 == 0 or gstep == 1:
                    tps = ep_toks / max(time.time() - ep_start, 0.1)
                    print_fn(f"step {gstep:4d}/{total} | loss: {acc_loss:.4f} | lr: {cur_lr:.2e} | tok/s: {tps:.0f}")

                if save_ckpts and gstep % save_every == 0:
                    save_ckpt(mdl, opt, gstep, acc_loss, ep)

                if acc_loss < best:
                    best = acc_loss

                acc_loss = 0.0

        avg = ep_loss / max(spe, 1)
        print_fn(f"\nepoch {ep+1} done | avg loss: {avg:.4f} | time: {time.time()-ep_start:.1f}s")
        save_ckpt(mdl, opt, gstep, avg, ep, is_epoch=True)

    print_fn("\ntraining complete, exporting weights")

    os.makedirs(export_dir, exist_ok=True)

    weights = {
        n: p.data.to(fp).cpu()
        for n, p in mdl.named_parameters()
        if p.requires_grad
    }
    wp = os.path.join(export_dir, "titan_symphony_weights.pt")
    torch.save(weights, wp)

    meta = {
        "engine": "Titan Engine - Symphony Architecture",
        "source_model": model_path,
        "frozen": ["attention", "embeddings", "norms", "lm_head"],
        "trained": ["mlp (FHRRMemoryLayer)"],
        "lambda_decay": decay
    }
    with open(os.path.join(export_dir, "titan_symphony_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print_fn(f"weights saved to {wp}")
    test_gen(mdl, tok)


if __name__ == "__main__":
    train()
