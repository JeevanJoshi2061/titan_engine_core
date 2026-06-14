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

        fhrr_out = torch.fft.irfft(rec, n=D, dim=-1).to(x.dtype)
        exact_out = self.cache.retrieve(q).to(x.dtype)

        combined = torch.cat([fhrr_out, exact_out], dim=-1)
        gated = self.pw1(combined) * self.act(self.pw2(combined))
        clean = self.norm(self.pw3(gated))

        g = torch.sigmoid(self.gate)
        return base + g * clean


def contrastive_loss(clean, target, distractor, margin=1.0):
    pd = F.pairwise_distance(clean, target, p=2)
    nd = F.pairwise_distance(clean, distractor, p=2)
    return torch.clamp(margin + pd - nd, min=0.0).mean()
