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
    def __init__(self, dim, vocab, max_len=2048, lam=0.999, use_triton=True):
        super().__init__()
        self.dim = dim
        self.vocab = vocab
        self.max_len = max_len
        self.lam = lam
        self.use_triton = use_triton and HAS_TRITON

        self.kp = nn.Linear(dim, dim, bias=False)
        self.qp = nn.Linear(dim, dim, bias=False)
        self.gp = nn.Linear(dim, 1)
        nn.init.normal_(self.gp.weight, std=0.01)
        nn.init.constant_(self.gp.bias, -10.0)

        self.register_buffer("pos_emb", self._make_pos(max_len, dim))
        self.reset_inference_state()

    def reset_inference_state(self):
        self._state = None
        self._ids = None
        self._seq_len = 0
        self.exact_h = []
        self.exact_i = []
        self._prompt_len = 0

    def rescale_positions_ntk(self, total_len, trained_max=512):
        if total_len <= trained_max:
            return
        alpha = total_len / trained_max
        d = self.dim
        new_base = 10000.0 * (alpha ** (d / (d - 2)))
        device = self.pos_emb.device
        padded = total_len + 256
        self.pos_emb = self._make_pos(padded, d, base=new_base).to(device)

    def _make_pos(self, length, dim, base=10000.0):
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        di = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
        rates = 1.0 / torch.pow(base, (2.0 * (di // 2)) / dim)
        angles = pos * rates
        emb = torch.zeros(length, dim)
        emb[:, 0::2] = torch.sin(angles[:, 0::2])
        emb[:, 1::2] = torch.cos(angles[:, 1::2])
        return emb

    def _norm_fft(self, x):
        return x / (torch.abs(x) + 1e-6)

    def forward(self, hidden, ids, lm_logits=None, target_ids=None, oracle_scores=None, prefill=False):
        B, S, D = hidden.shape
        device = hidden.device

        if prefill:
            if self._seq_len == 0:
                self.reset_inference_state()
            if oracle_scores is not None:
                k = max(1, int(S * 0.20))
                topk_v, topk_i = torch.topk(oracle_scores, k, dim=1)

                gi = topk_i.unsqueeze(-1).expand(-1, -1, D)
                eh = torch.gather(hidden, 1, gi).detach()

                ni = (topk_i + 1).clamp(max=S-1)
                ei = torch.gather(ids, 1, ni).detach()

                self.exact_h.append(eh)
                self.exact_i.append(ei)

        keys = self.kp(hidden).float()
        queries = self.qp(hidden).float()
        off = self._seq_len
        vals = self.pos_emb[off:off+S].unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=torch.float32)

        if self._ids is not None:
            self._ids = torch.cat([self._ids, ids], dim=1)
        else:
            self._ids = ids.clone()

        C = 512
        kf = torch.fft.rfft(keys, dim=-1)
        qf = torch.fft.rfft(queries, dim=-1)
        vf = torch.fft.rfft(vals, dim=-1)

        kn = self._norm_fft(kf)
        qn = self._norm_fft(qf)
        bound = kn * vf

        if S > 1:
            K = (S + C - 1) // C
            pad = K * C - S
            if pad > 0:
                bp = F.pad(bound, (0, 0, 0, pad))
            else:
                bp = bound

            br = bp.view(B, K, C, -1)
            cs = torch.cumsum(br, dim=2)
            fs = cs[:, :, -1, :]

            if prefill:
                if S <= C:
                    self._done_chunks = None
                    self._cur_chunk = cs[:, 0, S-1, :].detach()
                    self._cur_len = S
                else:
                    if S % C == 0:
                        self._done_chunks = fs.detach()
                        self._cur_chunk = None
                        self._cur_len = 0
                    else:
                        self._done_chunks = fs[:, :K-1, :].detach()
                        self._cur_chunk = cs[:, -1, (S % C) - 1, :].detach()
                        self._cur_len = S % C

            ti = torch.arange(K * C, device=device)
            ci = ti // C
            ii = ti % C
            ji = torch.arange(K, device=device)

            cg = ci.unsqueeze(1)
            jg = ji.unsqueeze(0)

            is_past = (jg < cg)
            is_active = (jg == cg)

            past_s = fs.unsqueeze(1).expand(-1, K * C, -1, -1)
            active_s = cs[:, ci, ii, :]

            hist = torch.zeros(B, K * C, K, kf.shape[-1], dtype=bound.dtype, device=device)
            hist = torch.where(is_past.unsqueeze(0).unsqueeze(-1), past_s, hist)
            hist = torch.where(is_active.unsqueeze(0).unsqueeze(-1), active_s.unsqueeze(2), hist)
            hist = hist[:, :S, :, :]

        else:
            bt = bound.squeeze(1)
            if self._cur_chunk is None:
                self._cur_chunk = bt
            else:
                self._cur_chunk = self._cur_chunk + bt
            self._cur_len += 1

            if self._cur_len == C:
                if self._done_chunks is None:
                    self._done_chunks = self._cur_chunk.unsqueeze(1)
                else:
                    self._done_chunks = torch.cat([self._done_chunks, self._cur_chunk.unsqueeze(1)], dim=1)
                self._cur_chunk = torch.zeros_like(self._cur_chunk)
                self._cur_len = 0

            if self._done_chunks is not None:
                if self._cur_len > 0:
                    hist = torch.cat([self._done_chunks, self._cur_chunk.unsqueeze(1)], dim=1)
                else:
                    hist = self._done_chunks
            else:
                hist = self._cur_chunk.unsqueeze(1)

            K = hist.shape[1]

        self._seq_len += S

        gate = torch.sigmoid(self.gp(hidden))

        if target_ids is not None:
            uf = hist * torch.conj(qn).unsqueeze(2)
            rp = torch.fft.irfft(uf, n=D, dim=-1)
            rp = torch.clamp(rp, -65000.0, 65000.0).to(hidden.dtype)

            pe_pad = F.pad(self.pos_emb, (0, 0, 0, K * C - self.pos_emb.shape[0]))
            pe_r = pe_pad[:K*C].view(K, C, D).to(device=device, dtype=hidden.dtype)

            pl = torch.einsum("bskd,kcd->bskc", rp, pe_r)
            pl = pl.view(B, S, K * C)
            pl = pl[:, :, :S] / math.sqrt(self.dim)

            ti_seq = torch.arange(S, device=device).unsqueeze(0)
            qi_seq = torch.arange(S, device=device).unsqueeze(1)
            cm = (ti_seq > qi_seq).unsqueeze(0)
            pl.masked_fill_(cm, -1e4)

            pp = F.softmax(pl, dim=-1)
            ps = torch.zeros_like(pp)
            ps[:, :, 1:] = pp[:, :, :-1]

            mm = (self._ids.unsqueeze(1) == target_ids.unsqueeze(2)).to(hidden.dtype)

            active = (target_ids != -100)
            if active.any():
                first = active.nonzero(as_tuple=True)[1].min().item()
                mm[:, :, first:] = 0.0

            ptp = torch.sum(ps * mm, dim=-1)

            if lm_logits is not None:
                lm_lp = F.log_softmax(lm_logits, dim=-1)
                safe_t = torch.where(target_ids == -100, torch.zeros_like(target_ids), target_ids)
                lm_tp = torch.exp(torch.gather(lm_lp, dim=-1, index=safe_t.unsqueeze(-1)).squeeze(-1))
                ftp = (1.0 - gate.squeeze(-1)) * lm_tp + gate.squeeze(-1) * ptp
            else:
                ftp = ptp

            gl = -torch.log(ftp + 1e-12)
            ptl = -torch.log(ptp + 1e-12)

            copyable = (mm * (~cm).to(hidden.dtype)).any(dim=-1)
            ptl = torch.where(copyable, ptl, torch.zeros_like(ptl))

            loss = gl + 0.1 * ptl
            return loss, gate

        elif prefill:
            self._prompt_len = self._seq_len
            return None, None

        else:
            uf = hist * torch.conj(qn)
            rp = torch.fft.irfft(uf, n=D, dim=-1)
            rp = torch.clamp(rp, -65000.0, 65000.0).to(hidden.dtype)

            total = self._seq_len
            pe_h = self.pos_emb[:total]

            pad = K * C - total
            if pad > 0:
                pe_pad = F.pad(pe_h, (0, 0, 0, pad))
            else:
                pe_pad = pe_h

            pe_r = pe_pad[:K*C].view(K, C, D).to(device=device, dtype=hidden.dtype)

            pl = torch.einsum("bkd,kcd->bkc", rp, pe_r)
            pl = pl.view(B, 1, K * C)
            pl = pl[:, :, :total] / math.sqrt(self.dim)

            pp = F.softmax(pl, dim=-1)
            ps = torch.zeros_like(pp)
            ps[:, :, 1:] = pp[:, :, :-1]

            if hasattr(self, "_prompt_len") and self._prompt_len > 0:
                ps[:, :, self._prompt_len:] = 0.0
                ps = ps / (ps.sum(dim=-1, keepdim=True) + 1e-12)

            pv = torch.zeros(B, 1, self.vocab, device=device, dtype=pp.dtype)
            pv.scatter_add_(dim=2, index=self._ids.unsqueeze(1), src=ps)

            gv = torch.sigmoid(self.gp(hidden[:, -1:, :]))

            print(f"\n[DEBUG Step] gate: {gv.item():.4f}")
            top_v, top_i = torch.topk(ps[0, 0], 5)
            print("  Top 5 Pointer Positions:")
            for v, i in zip(top_v, top_i):
                tid = self._ids[0, i].item()
                print(f"    Pos {i.item()}: ID {tid} | Prob: {v.item():.4f}")

            if lm_logits is not None:
                lm_last = lm_logits[:, -1:, :]
                lm_p = F.softmax(lm_last, dim=-1)
                fp = (1.0 - gv) * lm_p + gv * pv

                top_lv, top_li = torch.topk(lm_p[0, 0], 3)
                print("  Top 3 LM Tokens:")
                for v, i in zip(top_lv, top_li):
                    print(f"    ID {i.item()}: Prob {v.item():.4f}")

                top_fv, top_fi = torch.topk(fp[0, 0], 3)
                print("  Top 3 Final Tokens:")
                for v, i in zip(top_fv, top_fi):
                    print(f"    ID {i.item()}: Prob {v.item():.4f}")

                return torch.log(fp + 1e-12), gv
            else:
                return torch.log(pv + 1e-12), gv