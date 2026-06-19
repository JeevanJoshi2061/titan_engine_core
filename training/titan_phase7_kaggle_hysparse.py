import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

        ovk = None
        ovv = None

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
    def __init__(self, dim, orig_mlp, lam=0.99):
        super().__init__()
        self.dim = dim
        self.orig_mlp = orig_mlp
        self.lam = lam
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
        self.norm = nn.RMSNorm(dim)

        self.gate = nn.Parameter(torch.tensor([-10.0]))
        self.mem = None

    def reset_state(self):
        self.mem = None
        self.cache.clear()

    def forward(self, x):
        B, S, D = x.shape
        device = x.device

        base = self.orig_mlp(x)
        sel = self.router(x)

        k = self.kp(x).float()
        v = self.vp(x).float()
        q = self.qp(x).float()

        self.cache.add(k, v, sel)

        kfr = torch.fft.rfft(k, dim=-1)
        vfr = torch.fft.rfft(v, dim=-1)
        qfr = torch.fft.rfft(q, dim=-1)

        ka = torch.sqrt(kfr.real.pow(2) + kfr.imag.pow(2) + 1e-12)
        kn = kfr / ka.to(dtype=torch.complex64)

        bound = kn * vfr

        if self.training or self.mem is None or self.mem.shape[0] != B:
            state0 = torch.zeros(B, self.fdim, dtype=torch.complex64, device=device)
        else:
            state0 = self.mem

        sc = math.sqrt(1.0 - self.lam ** 2)

        if S <= 8192:
            ti = torch.arange(S, device=device).unsqueeze(1)
            ii = torch.arange(S, device=device).unsqueeze(0)
            pw = torch.clamp(ti - ii, min=0)
            msk = (ti - ii >= 0).float()
            W = ((self.lam ** pw) * msk).to(dtype=torch.complex64)

            rec = sc * torch.matmul(W, bound)

            steps = torch.arange(1, S + 1, device=device, dtype=torch.float32)
            df = (self.lam ** steps).unsqueeze(0).unsqueeze(2).to(dtype=torch.complex64)
            rec = rec + state0.unsqueeze(1) * df
        else:
            rec = torch.empty_like(bound)
            curr = state0.clone()
            for t in range(S):
                curr.mul_(self.lam).add_(bound[:, t], alpha=sc)
                rec[:, t] = curr

        if not self.training:
            self.mem = rec[:, -1, :].detach()

        qa = torch.sqrt(qfr.real.pow(2) + qfr.imag.pow(2) + 1e-12)
        qn = qfr / qa.to(dtype=torch.complex64)
        rec = rec * torch.conj(qn)

        fhrr_out = torch.fft.irfft(rec, n=D, dim=-1)
        fhrr_out = torch.clamp(fhrr_out, -65000.0, 65000.0).to(x.dtype)
        exact_out = self.cache.retrieve(q).to(x.dtype)

        combined = torch.cat([fhrr_out, exact_out], dim=-1)
        gated = self.pw1(combined) * self.act(self.pw2(combined))
        clean = self.norm(self.pw3(gated))

        g = torch.sigmoid(self.gate)
        return base + g * clean


def contrastive_loss(clean, target, distractor, margin=1.0):
    pd = F.pairwise_distance(clean.float(), target.float(), p=2, eps=1e-6)
    nd = F.pairwise_distance(clean.float(), distractor.float(), p=2, eps=1e-6)
    return torch.clamp(margin + pd - nd, min=0.0).mean()


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def fhrr_scan_kernel(
        kr_ptr, ki_ptr, vr_ptr, vi_ptr,
        or_ptr, oi_ptr,
        ir_ptr, ii_ptr,
        decay, scale,
        b_stride, s_stride, f_stride,
        B, S, F_dim,
        BLOCK: tl.constexpr
    ):
        pb = tl.program_id(0)
        pf = tl.program_id(1)

        offs = pf * BLOCK + tl.arange(0, BLOCK)
        mask = offs < F_dim

        init_off = pb * F_dim + offs
        ar = tl.load(ir_ptr + init_off, mask=mask, other=0.0)
        ai = tl.load(ii_ptr + init_off, mask=mask, other=0.0)

        for s in range(0, S):
            base = pb * b_stride + s * s_stride + offs * f_stride

            kr = tl.load(kr_ptr + base, mask=mask, other=0.0)
            ki = tl.load(ki_ptr + base, mask=mask, other=0.0)
            vr = tl.load(vr_ptr + base, mask=mask, other=0.0)
            vi = tl.load(vi_ptr + base, mask=mask, other=0.0)

            mag = tl.sqrt(kr * kr + ki * ki + 1e-6)
            kn_r = kr / mag
            kn_i = ki / mag

            br = kn_r * vr - kn_i * vi
            bi = kn_r * vi + kn_i * vr

            ar = decay * ar + scale * br
            ai = decay * ai + scale * bi

            tl.store(or_ptr + base, ar, mask=mask)
            tl.store(oi_ptr + base, ai, mask=mask)


def triton_fhrr_scan(keys, values, init=None, lam=0.99):
    B, S, D = keys.shape
    device = keys.device

    kf = torch.fft.rfft(keys, dim=-1)
    vf = torch.fft.rfft(values, dim=-1)

    kr = kf.real.contiguous()
    ki = kf.imag.contiguous()
    vr = vf.real.contiguous()
    vi = vf.imag.contiguous()
    out_r = torch.empty_like(kr)
    out_i = torch.empty_like(ki)

    if init is not None:
        ir = init.real.contiguous()
        ii = init.imag.contiguous()
    else:
        ir = torch.zeros(B, kr.shape[2], device=device, dtype=kr.dtype)
        ii = torch.zeros(B, kr.shape[2], device=device, dtype=kr.dtype)

    bs, ss, fs = kr.stride()
    fd = kr.shape[2]
    BLK = min(512, triton.next_power_of_2(fd))
    sc = math.sqrt(1.0 - lam**2)

    grid = (B, triton.cdiv(fd, BLK))
    fhrr_scan_kernel[grid](
        kr, ki, vr, vi,
        out_r, out_i,
        ir, ii,
        lam, sc,
        bs, ss, fs,
        B, S, fd,
        BLOCK=BLK
    )

    return torch.complex(out_r, out_i)


class PointerNet(nn.Module):
    def __init__(self, dim, vocab, scales=[0.5, 0.9, 0.999], max_len=None, lam=None, use_triton=None):
        super().__init__()
        self.dim = dim
        self.vocab = vocab
        self.scales = scales
        
        self.kp = nn.Linear(dim, dim, bias=False)
        self.vp = nn.Linear(dim, dim, bias=False)
        self.qp = nn.Linear(dim, dim, bias=False)
        
        self.out_proj1 = nn.Linear(dim * len(scales), dim * 2, bias=False)
        self.out_proj2 = nn.Linear(dim * 2, dim, bias=False)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(dim)
        
        self.gp = nn.Linear(dim, 1)
        nn.init.normal_(self.gp.weight, std=0.01)
        nn.init.constant_(self.gp.bias, -2.0)

        fdim = dim // 2 + 1
        self.register_buffer("freqs", self._make_freqs(fdim))
        self.reset_inference_state()

    def _make_freqs(self, fdim, base=10000.0):
        idx = torch.arange(fdim, dtype=torch.float32)
        freqs = 1.0 / (base ** (idx / fdim))
        return freqs

    def reset_inference_state(self):
        self._states = None
        self._t = 0
        
    def rescale_positions_ntk(self, total_len, trained_max=512):
        # No-op: True O(1) memory uses relative RoPE, not absolute bounded encodings!
        pass

    def _norm_fft(self, x):
        return x / (torch.abs(x) + 1e-6)

    def forward(self, hidden, ids=None, lm_logits=None, target_ids=None, oracle_scores=None, prefill=False, lm_head_weight=None):
        B, S, D = hidden.shape
        device = hidden.device
        dtype = hidden.dtype
        fdim = D // 2 + 1
        
        keys = self.kp(hidden).float()
        vals = self.vp(hidden).float()
        queries = self.qp(hidden).float()
        
        kf = torch.fft.rfft(keys, dim=-1)
        vf = torch.fft.rfft(vals, dim=-1)
        qf = torch.fft.rfft(queries, dim=-1)
        
        pos = torch.arange(self._t, self._t + S, device=device, dtype=torch.float32).unsqueeze(1)
        angles = pos * self.freqs.unsqueeze(0)
        angles = angles.unsqueeze(0).expand(B, -1, -1)
        rot = torch.polar(torch.ones_like(angles), angles)
        
        kf_rot = kf * rot
        qf_rot = qf * rot
        
        kn = self._norm_fft(kf_rot)
        qn = self._norm_fft(qf_rot)
        bound = kn * vf
        
        retrieved_scales = []
        
        if self.training or S > 1:
            ti = torch.arange(S, device=device).unsqueeze(1)
            ii = torch.arange(S, device=device).unsqueeze(0)
            pw = torch.clamp(ti - ii, min=0)
            msk = (ti - ii >= 0).float()
            
            new_states = []
            for i, lam in enumerate(self.scales):
                sc = math.sqrt(1.0 - lam**2)
                W = ((lam ** pw) * msk).to(dtype=torch.complex64)
                rec = sc * torch.einsum("st,btd->bsd", W, bound)
                
                if self._states is not None:
                    state0 = self._states[i]
                    steps = torch.arange(1, S + 1, device=device, dtype=torch.float32)
                    df = (lam ** steps).unsqueeze(0).unsqueeze(2).to(dtype=torch.complex64)
                    rec = rec + state0.unsqueeze(1) * df
                
                new_states.append(rec[:, -1, :].detach())
                uf = rec * torch.conj(qn)
                rp = torch.fft.irfft(uf, n=D, dim=-1)
                rp = torch.clamp(rp, -65000.0, 65000.0).to(dtype)
                retrieved_scales.append(rp)
            self._states = new_states
        else:
            new_states = []
            for i, lam in enumerate(self.scales):
                sc = math.sqrt(1.0 - lam**2)
                if self._states is None:
                    state0 = torch.zeros(B, fdim, dtype=torch.complex64, device=device)
                else:
                    state0 = self._states[i]
                rec = lam * state0 + sc * bound[:, 0, :]
                new_states.append(rec.detach())
                uf = rec.unsqueeze(1) * torch.conj(qn)
                rp = torch.fft.irfft(uf, n=D, dim=-1)
                rp = torch.clamp(rp, -65000.0, 65000.0).to(dtype)
                retrieved_scales.append(rp)
            self._states = new_states
            
        self._t += S
        
        combined = torch.cat(retrieved_scales, dim=-1)
        h_out = self.out_proj2(self.act(self.out_proj1(combined)))
        h_out = self.norm(h_out)
        
        gate = torch.sigmoid(self.gp(hidden))
        
        if lm_head_weight is not None:
            if target_ids is not None:
                if lm_logits is not None:
                    safe_t = target_ids.clone()
                    safe_t[safe_t == -100] = 0 
                    
                    CHUNK = 512
                    ptr_tgt_chunks = []
                    ptr_log_z_chunks = []
                    
                    for start in range(0, S, CHUNK):
                        end = min(start + CHUNK, S)
                        chunk = F.linear(h_out[:, start:end].float(), lm_head_weight.float())
                        
                        safe_t_chunk = safe_t[:, start:end].unsqueeze(-1)
                        ptr_tgt_chunks.append(torch.gather(chunk, -1, safe_t_chunk).squeeze(-1))
                        ptr_log_z_chunks.append(torch.logsumexp(chunk, dim=-1))
                        del chunk
                        
                    del h_out
                    ptr_tgt = torch.cat(ptr_tgt_chunks, dim=1)
                    ptr_log_z = torch.cat(ptr_log_z_chunks, dim=1)
                    
                    lm_log_z = torch.logsumexp(lm_logits, dim=-1)     # [B, S]
                    lm_tgt  = torch.gather(lm_logits,  -1, safe_t.unsqueeze(-1)).squeeze(-1)  # [B, S]
                    
                    lm_lp  = lm_tgt  - lm_log_z   # [B, S]
                    ptr_lp  = ptr_tgt - ptr_log_z  # [B, S]
                    
                    gate_sq  = gate.squeeze(-1)                              # [B, S]
                    log_1mg  = torch.log((1.0 - gate_sq).clamp(min=1e-7))
                    log_g    = torch.log(gate_sq.clamp(min=1e-7))
                    
                    mixed_lp = torch.logaddexp(lm_lp + log_1mg, ptr_lp + log_g)  # [B, S]
                    
                    # Step 5: Loss
                    loss = -mixed_lp
                    loss = loss * (target_ids != -100).float()  # zero out padding
                else:
                    ptr_logits = F.linear(h_out, lm_head_weight)
                    del h_out
                    loss = F.cross_entropy(ptr_logits.view(-1, self.vocab), target_ids.view(-1), ignore_index=-100, reduction='none')
                return loss.view(B, S), gate
            else:
                ptr_logits = F.linear(h_out, lm_head_weight)
                del h_out
                ptr_probs = F.softmax(ptr_logits, dim=-1)
                if lm_logits is not None:
                    lm_probs = F.softmax(lm_logits, dim=-1)
                    mixed_probs = (1.0 - gate) * lm_probs + gate * ptr_probs
                    return torch.log(mixed_probs + 1e-12), gate
                return torch.log(ptr_probs + 1e-12), gate
                
        return torch.zeros(B, S, device=device), gate

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import random
import string
import time
import sys



PHASE4_WEIGHTS = r"/kaggle/input/datasets/jeevanjoshi1/titan-symphony-weights/titan_symphony_weights.pt"
OUTPUT_DIR = r"/kaggle/working/checkpoints"

BATCH_SIZE = 1
LEARNING_RATE = 2e-3
MAX_STEPS = 20000

RESUME_CHECKPOINT = ""
START_STEP = 0

SCALE_SCHEDULE = [
    (300,  512),
    (600,  1024),
    (800,  2048),
    (1000, 4096),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

class LongRangeCopyDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len=1024):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.fillers = [
            "The system architecture requires strict validation protocols for all incoming requests. ",
            "According to the documentation on page 42, the integration setup is complete and verified. ",
            "Network timeout occurred during the standard initialization phase of the service mesh. ",
            "Please ensure that the firewall rules are updated to allow incoming traffic on port 443. ",
            "The user authentication flow uses OAuth 2.0 with JWT tokens for session management. ",
            "Data encryption at rest is handled by AES-256 standard protocols across all storage nodes. ",
            "Memory allocation limits were exceeded during the load testing phase of deployment. ",
            "Kubernetes pod deployment status is currently pending resource allocation from the cluster. ",
            "The database migration script completed successfully with zero schema conflicts detected. ",
            "Automated health checks are running every 30 seconds across all microservice endpoints. ",
            "The CDN cache invalidation propagated to all 47 edge nodes within the expected TTL window. ",
            "Container orchestration logs indicate nominal CPU usage at 23% with 4.2GB memory footprint. ",
            "SSL certificate renewal is scheduled for next month. Current cert expires on 2026-07-15. ",
            "The API gateway rate limiter is configured to allow 1500 requests per minute per client. ",
            "Distributed tracing shows average request latency of 142ms across the service topology. ",
            "The message queue consumer lag is within acceptable thresholds at 12 messages behind head. ",
        ]
        
    def _generate_random_code(self, length=16):
        chars = string.ascii_uppercase + string.digits
        segments = []
        remaining = length
        while remaining > 0:
            seg_len = min(random.choice([3, 4, 5]), remaining)
            segments.append(''.join(random.choice(chars) for _ in range(seg_len)))
            remaining -= seg_len
        return '-'.join(segments)
        
    def __iter__(self):
        while True:
            code_type = random.choice([
                "Verification Code", "API Key", "Transaction ID", 
                "Access Token", "Security Hash", "Session Key",
                "Recovery Code", "License Key"
            ])
            secret_code = self._generate_random_code(random.randint(16, 28))
            
            system_wrap = "<|im_start|>system\nYou are a precise AI assistant with perfect memory.<|im_end|>\n"
            user_start = "<|im_start|>user\nHere is the complete system log:\n\n"
            needle = f"\n[CRITICAL SECURITY ALERT] The {code_type} is: {secret_code}\n"
            user_end = f"\n\nExtract the exact {code_type} from the log above.<|im_end|>\n"
            assistant_response = f"<|im_start|>assistant\nThe exact {code_type} is: {secret_code}<|im_end|>"
            
            overhead_text = system_wrap + user_start + needle + user_end + assistant_response
            overhead_ids = self.tokenizer.encode(overhead_text, add_special_tokens=False)
            
            filler_budget = self.seq_len - len(overhead_ids) - 10
            if filler_budget <= 0:
                filler_budget = 64
            
            prefix_fillers = []
            suffix_fillers = []
            current_filler_tokens = 0
            
            needle_depth = random.uniform(0.3, 0.9)
            prefix_budget = int(filler_budget * needle_depth)
            
            while True:
                sentence = random.choice(self.fillers)
                sent_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
                if current_filler_tokens + len(sent_ids) > prefix_budget:
                    break
                prefix_fillers.append(sentence)
                current_filler_tokens += len(sent_ids)
                
            while True:
                sentence = random.choice(self.fillers)
                sent_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
                if current_filler_tokens + len(sent_ids) > filler_budget:
                    break
                suffix_fillers.append(sentence)
                current_filler_tokens += len(sent_ids)
                
            prefix = "".join(prefix_fillers)
            suffix = "".join(suffix_fillers)
            
            prompt_text = (
                f"{system_wrap}{user_start}"
                f"{prefix}{needle}{suffix}"
                f"{user_end}{assistant_response}"
            )
            
            encoded = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"][0]
            
            code_ids = self.tokenizer.encode(secret_code, add_special_tokens=False)
            
            code_start_idx = -1
            L = len(code_ids)
            for i in range(len(input_ids) - L, -1, -1):
                if input_ids[i:i+L].tolist() == code_ids:
                    code_start_idx = i
                    break
                    
            if code_start_idx == -1:
                continue
                
            code_end_idx = code_start_idx + L
            
            labels = torch.full_like(input_ids, -100)
            labels[code_start_idx:code_end_idx] = input_ids[code_start_idx:code_end_idx]
            
            yield {
                "input_ids": input_ids,
                "labels": labels
            }

def prepare_model():
    log.info("Loading model...")
    device = torch.device("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/models/qwen-lm/qwen2.5/transformers/1.5b-instruct/1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "/kaggle/input/models/qwen-lm/qwen2.5/transformers/1.5b-instruct/1",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    hidden_dim = model.config.hidden_size
    
    for idx in range(20, 28):
        original_mlp = model.model.layers[idx].mlp
        symphony_layer = MemLayer(hidden_dim, original_mlp)
        symphony_layer = symphony_layer.to(device=device, dtype=torch.bfloat16)
        model.model.layers[idx].mlp = symphony_layer
        
    log.info(f"Loading Phase 4 weights from {PHASE4_WEIGHTS}...")
    state_dict = torch.load(PHASE4_WEIGHTS, map_location=device, weights_only=True)
    custom_state = {k: v for k, v in state_dict.items() if any(x in k for x in ["mlp.router", "mlp.pw", "mlp.gate", "mlp.kp", "mlp.qp", "mlp.vp"])}
    model.load_state_dict(custom_state, strict=False)
    
    hep_dna = PointerNet(
        dim=hidden_dim, 
        vocab=model.config.vocab_size,
        scales=[0.999, 0.9998, 0.99995]  # Built for 4096+ tokens (survival at 4k: 1.7%, 44%, 81%)
    ).to(device=device, dtype=torch.bfloat16)
    
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        log.info(f"Resuming HEP-DNA pointer from {RESUME_CHECKPOINT}...")
        hep_state = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=True)
        
        hep_dna.rescale_positions_ntk(SCALE_SCHEDULE[-1][1])
        
        hep_dna.load_state_dict(hep_state)
    else:
        log.info("Initializing HEP-DNA pointer from scratch for unified training...")
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in hep_dna.parameters():
        param.requires_grad = True
        
    return model, tokenizer, hep_dna, device

def train():
    model, tokenizer, hep_dna, device = prepare_model()
    optimizer = torch.optim.AdamW(hep_dna.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    model.eval()
    hep_dna.train()
    
    step = START_STEP
    current_scale_idx = 0
    
    while current_scale_idx < len(SCALE_SCHEDULE) - 1 and step >= SCALE_SCHEDULE[current_scale_idx][0]:
        current_scale_idx += 1
        
    current_seq_len = SCALE_SCHEDULE[current_scale_idx][1]
    
    dataset = LongRangeCopyDataset(tokenizer, seq_len=current_seq_len)
    dataloader = iter(DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, prefetch_factor=2))
    hep_dna.rescale_positions_ntk(current_seq_len)
    
    log.info(f"Resuming Phase 7 long-range calibration from step {step} at SEQ_LEN={current_seq_len}...")
    
    start_time = time.time()
    
    while step < MAX_STEPS:
        if current_scale_idx < len(SCALE_SCHEDULE) - 1:
            if step >= SCALE_SCHEDULE[current_scale_idx][0]:
                current_scale_idx += 1
                current_seq_len = SCALE_SCHEDULE[current_scale_idx][1]
                dataset = LongRangeCopyDataset(tokenizer, seq_len=current_seq_len)
                dataloader = iter(DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, prefetch_factor=2))
                hep_dna.rescale_positions_ntk(current_seq_len)
                log.info(f"Stepping to SEQ_LEN = {current_seq_len}")
        
        try:
            batch = next(dataloader)
        except StopIteration:
            dataloader = iter(DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, prefetch_factor=2))
            batch = next(dataloader)
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        for layer in model.model.layers:
            if hasattr(layer.mlp, "reset_state"):
                layer.mlp.reset_state()
        hep_dna.reset_inference_state()
        
        S = input_ids.shape[1]
        mask_2d = torch.full((S, S), float("-inf"), device=device, dtype=torch.bfloat16)
        mask_2d.fill_diagonal_(0.0)
        mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model.model(input_ids, attention_mask=mask_4d)
            hidden_states = outputs.last_hidden_state
            lm_logits = model.lm_head(hidden_states)
            
        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_input_ids = input_ids[..., :-1].contiguous()
        shift_lm_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        del lm_logits
        del hidden_states
        torch.cuda.empty_cache()
        
        valid_mask = (shift_labels != -100)
        if not valid_mask.any():
            continue
            
        optimizer.zero_grad()
        
        loss_tensor, gate_tensor = hep_dna(
            hidden=shift_hidden,
            ids=shift_input_ids,
            lm_logits=shift_lm_logits,
            target_ids=shift_labels,
            lm_head_weight=model.lm_head.weight
        )
        
        loss = loss_tensor[valid_mask].mean()
        loss.backward()
        
        gnorm = torch.nn.utils.clip_grad_norm_(hep_dna.parameters(), 1.0)
        
        if step == START_STEP:
            log.info(f"DEBUG: gnorm = {gnorm.item()}")
            g_norm = hep_dna.gp.weight.grad.norm().item() if hep_dna.gp.weight.grad is not None else None
            log.info(f"DEBUG: gp weight grad = {g_norm}")
        optimizer.step()
        
        loss_val = loss.item()
        mean_gate = gate_tensor.squeeze(-1)[valid_mask].mean().item()
        if step == START_STEP:
            log.info(f"DEBUG GATE TENSOR AT VALID MASK: {gate_tensor.squeeze(-1)[valid_mask].tolist()}")
        
        step += 1
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (BATCH_SIZE * current_seq_len * 10) / elapsed
            log.info(f"Step {step:04d}/{MAX_STEPS} | SEQ={current_seq_len} | Loss: {loss_val:.4f} | Gate: {mean_gate:.4f} | Tok/s: {tok_per_sec:.0f}")
            start_time = time.time()
            
        if step % 1000 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"titan_hysparse_p7_step{step}.pt")
            torch.save(hep_dna.state_dict(), ckpt_path)
            log.info(f"Saved checkpoint to {ckpt_path}")
            
    final_path = os.path.join(OUTPUT_DIR, "titan_hysparse_longrange_final.pt")
    torch.save(hep_dna.state_dict(), final_path)
    log.info(f"Phase 7 Complete. Final weights: {final_path}")

if __name__ == "__main__":
    train()
