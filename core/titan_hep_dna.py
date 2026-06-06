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
    def fhrr_recurrent_scan_kernel(
        keys_ptr, values_ptr, output_ptr,
        decay, scale_factor,
        batch_stride, seq_stride, feat_stride,
        batch_size, seq_len, feat_dim,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Fused Triton GPU kernel for FHRR recurrent decay scan.
        Performs normalization, complex holographic binding, and decay accumulation.
        """
        pid_batch = tl.program_id(0)
        pid_feat = tl.program_id(1)
        
        feat_offsets = tl.arange(0, BLOCK_SIZE)
        feat_mask = feat_offsets < feat_dim
        
        decay_val = decay
        scale_val = scale_factor
        
        accum_real = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        accum_imag = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        for s in range(0, seq_len):
            base_offset = (
                pid_batch * batch_stride +
                s * seq_stride +
                feat_offsets * feat_stride
            )
            
            k_real = tl.load(keys_ptr + base_offset, mask=feat_mask, other=0.0)
            k_imag = tl.load(keys_ptr + base_offset + 1, mask=feat_mask, other=0.0)
            
            v_real = tl.load(values_ptr + base_offset, mask=feat_mask, other=0.0)
            v_imag = tl.load(values_ptr + base_offset + 1, mask=feat_mask, other=0.0)
            
            # Key normalization
            mag = tl.sqrt(k_real * k_real + k_imag * k_imag + 1e-6)
            kn_real = k_real / mag
            kn_imag = k_imag / mag
            
            # Circular convolution complex binding
            bound_real = kn_real * v_real - kn_imag * v_imag
            bound_imag = kn_real * v_imag + kn_imag * v_real
            
            # Decay accumulation
            accum_real = decay_val * accum_real + scale_val * bound_real
            accum_imag = decay_val * accum_imag + scale_val * bound_imag
            
            out_offset = (
                pid_batch * batch_stride +
                s * seq_stride +
                feat_offsets * feat_stride
            )
            tl.store(output_ptr + out_offset, accum_real, mask=feat_mask)
            tl.store(output_ptr + out_offset + 1, accum_imag, mask=feat_mask)


def run_triton_fhrr_scan(keys, values, decay=0.99):
    B, S, D = keys.shape
    device = keys.device
    
    keys_freq = torch.fft.rfft(keys, dim=-1)
    values_freq = torch.fft.rfft(values, dim=-1)
    
    keys_packed = torch.view_as_real(keys_freq).contiguous()
    values_packed = torch.view_as_real(values_freq).contiguous()
    out_packed = torch.empty_like(keys_packed)
    
    batch_stride, seq_stride, feat_stride, _ = keys_packed.stride()
    feat_dim = keys_packed.shape[2]
    BLOCK_SIZE = triton.next_power_of_2(feat_dim)
    scale_factor = math.sqrt(1.0 - decay**2)
    
    grid = (B, 1)
    fhrr_recurrent_scan_kernel[grid](
        keys_packed, values_packed, out_packed,
        decay, scale_factor,
        batch_stride, seq_stride, feat_stride,
        B, S, feat_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    out_complex = torch.view_as_complex(out_packed)
    return out_complex


class SymphonyHEPDNALayer(nn.Module):
    """
    Symphony HEP-DNA Memory Layer.
    Positions are encoded in frequency space and mapped to keys for exact coordinate lookup.
    """
    def __init__(self, hidden_dim, vocab_size, max_seq_len=2048, decay=0.99, use_triton=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.decay = decay
        self.use_triton = use_triton and HAS_TRITON
        
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, 1)
        
        self.register_buffer("pos_embeddings", self._create_sinusoidal_embeddings(max_seq_len, hidden_dim))
        
        self.reset_inference_state()
        
    def reset_inference_state(self):
        self._persistent_state = None
        self._accumulated_ids = None
        self._total_seq_len = 0
        self.exact_hidden_states = []
        self.exact_ids = []

    def rescale_positions_ntk(self, total_len, trained_max=512):
        """NTK-aware frequency scaling for extrapolation beyond max training context."""
        if total_len <= trained_max:
            return
        
        alpha = total_len / trained_max
        dim = self.hidden_dim
        new_base = 10000.0 * (alpha ** (dim / (dim - 2)))
        
        device = self.pos_embeddings.device
        padded_len = total_len + 256
        new_emb = self._create_sinusoidal_embeddings(padded_len, dim, base=new_base)
        self.pos_embeddings = new_emb.to(device)

    def _create_sinusoidal_embeddings(self, seq_len, dim, base=10000.0):
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        dim_idx = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
        angle_rates = 1.0 / torch.pow(base, (2.0 * (dim_idx // 2)) / dim)
        angles = pos * angle_rates
        
        embeddings = torch.zeros(seq_len, dim)
        embeddings[:, 0::2] = torch.sin(angles[:, 0::2])
        embeddings[:, 1::2] = torch.cos(angles[:, 1::2])
        return embeddings

    def _fft_normalize(self, x):
        magnitude = torch.abs(x) + 1e-6
        return x / magnitude

    def forward(self, hidden_states, input_ids, lm_logits=None, target_ids=None, oracle_scores=None, prefill=False):
        B, S, D = hidden_states.shape
        device = hidden_states.device
        
        if prefill and oracle_scores is not None:
            k = max(1, int(S * 0.20))
            topk_vals, topk_indices = torch.topk(oracle_scores, k, dim=1)
            
            for b in range(B):
                idx = topk_indices[b]
                exact_h = hidden_states[b:b+1, idx, :]
                next_idx = (idx + 1).clamp(max=S-1)
                exact_i = input_ids[b:b+1, next_idx]
                self.exact_hidden_states.append(exact_h)
                self.exact_ids.append(exact_i)
        
        keys = self.key_proj(hidden_states).float()
        queries = self.query_proj(hidden_states).float()
        
        offset = self._total_seq_len if not self.training else 0
        values = self.pos_embeddings[offset:offset+S].unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=torch.float32)
        
        if self.use_triton:
            m_state = run_triton_fhrr_scan(keys, values, decay=self.decay)
            queries_freq = torch.fft.rfft(queries, dim=-1)
            queries_norm = self._fft_normalize(queries_freq)
        else:
            keys_freq = torch.fft.rfft(keys, dim=-1)
            queries_freq = torch.fft.rfft(queries, dim=-1)
            values_freq = torch.fft.rfft(values, dim=-1)
            
            keys_norm = self._fft_normalize(keys_freq)
            queries_norm = self._fft_normalize(queries_freq)
            bound = keys_norm * values_freq
            scale = math.sqrt(1.0 - self.decay**2)
            
            if self.training:
                t_indices = torch.arange(S, device=device).unsqueeze(1)
                i_indices = torch.arange(S, device=device).unsqueeze(0)
                power = torch.clamp(t_indices - i_indices, min=0)
                mask = (t_indices - i_indices >= 0).float()
                W = ((self.decay ** power) * mask).to(dtype=torch.complex64)
                m_state = scale * torch.matmul(W, bound)
            else:
                if self._persistent_state is not None:
                    curr = self._persistent_state.clone()
                else:
                    curr = torch.zeros_like(bound[:, 0])
                
                m_state = torch.empty_like(bound)
                for t in range(S):
                    curr.mul_(self.decay).add_(bound[:, t], alpha=scale)
                    m_state[:, t] = curr
                
                self._persistent_state = curr.detach()
                
        if not self.training:
            if self._accumulated_ids is not None:
                self._accumulated_ids = torch.cat([self._accumulated_ids, input_ids], dim=1)
            else:
                self._accumulated_ids = input_ids.clone()
            self._total_seq_len += S
        
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        
        if target_ids is not None:
            keys_f = F.normalize(self.key_proj(hidden_states), dim=-1)
            queries_f = F.normalize(self.query_proj(hidden_states), dim=-1)
            pointer_logits = torch.matmul(queries_f, keys_f.transpose(1, 2)) * 20.0
            
            mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
            pointer_logits.masked_fill_(mask.unsqueeze(0), -1e4)
            pointer_probs = F.softmax(pointer_logits, dim=-1)
            
            match_mask = (input_ids.unsqueeze(1) == target_ids.unsqueeze(2)).to(hidden_states.dtype)
            pointer_target_prob = torch.sum(pointer_probs * match_mask, dim=-1)
            
            if lm_logits is not None:
                lm_log_probs = F.log_softmax(lm_logits, dim=-1)
                safe_target_ids = torch.where(target_ids == -100, torch.zeros_like(target_ids), target_ids)
                lm_target_prob = torch.exp(torch.gather(lm_log_probs, dim=-1, index=safe_target_ids.unsqueeze(-1)).squeeze(-1))
                
                final_target_prob = (1.0 - gate.squeeze(-1)) * lm_target_prob + gate.squeeze(-1) * pointer_target_prob
            else:
                final_target_prob = pointer_target_prob
                
            loss = -torch.log(final_target_prob + 1e-12)
            return loss, gate
        
        elif prefill:
            return None, None
            
        else:
            if len(self.exact_hidden_states) > 0:
                exact_h = torch.cat(self.exact_hidden_states, dim=1)
                exact_i = torch.cat(self.exact_ids, dim=1)
                
                exact_k = F.normalize(self.key_proj(exact_h).float(), dim=-1)
                query_last = F.normalize(self.query_proj(hidden_states[:, -1:, :]).float(), dim=-1)
                
                exact_logits = torch.bmm(query_last, exact_k.transpose(1, 2)) * 20.0
                exact_probs = F.softmax(exact_logits, dim=-1)
                
                pointer_vocab_probs_last = torch.zeros(B, 1, self.vocab_size, device=device, dtype=exact_probs.dtype)
                pointer_vocab_probs_last.scatter_add_(dim=2, index=exact_i.unsqueeze(1), src=exact_probs)
            else:
                pointer_vocab_probs_last = torch.zeros(B, 1, self.vocab_size, device=device, dtype=torch.float32)
            
            if lm_logits is not None:
                lm_logits_last = lm_logits[:, -1:, :]
                final_logits_last = lm_logits_last.clone()
                
                if not self.training and len(self.exact_hidden_states) > 0:
                    max_score = exact_logits.max(-1).values.item()
                    if max_score > 12.0:
                        max_pos = exact_logits.argmax(-1)
                        target_token = exact_i.gather(1, max_pos)
                        final_logits_last.scatter_(dim=2, index=target_token.unsqueeze(1), value=100.0)
            else:
                final_logits_last = pointer_vocab_probs_last
                
            return final_logits_last, gate[:, -1:, :]
