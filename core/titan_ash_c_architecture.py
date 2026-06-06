import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ASHCRouter(nn.Module):
    """
    Gumbel-Softmax Router for selective token routing.
    Outputs a discrete [0.0, 1.0] choice using straight-through estimators.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.salience_proj = nn.Linear(hidden_dim, 2)
        
    def forward(self, hidden_states, temperature=1.0, hard=True):
        logits = self.salience_proj(hidden_states)
        
        if self.training:
            routing_weights = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        else:
            preds = torch.argmax(logits, dim=-1)
            routing_weights = F.one_hot(preds, num_classes=2).to(hidden_states.dtype)
            
        fact_selection = routing_weights[:, :, 1]
        return fact_selection


class PhantomKVCache(nn.Module):
    """
    Dynamic sparse cache that stores only key tokens tagged as facts.
    """
    def __init__(self, hidden_dim, max_capacity=2048):
        super().__init__()
        self.max_capacity = max_capacity
        self.hidden_dim = hidden_dim
        self.reset_cache()
        
    def reset_cache(self):
        self.keys = None
        self.values = None
        self.masks = None
        self.num_tokens = 0
        
    def add_to_cache(self, new_keys, new_values, fact_selection):
        B, S, D = new_keys.shape
        device = new_keys.device
        
        batch_keys = []
        batch_values = []
        max_facts = 0
        
        for b in range(B):
            valid_indices = torch.nonzero(fact_selection[b] > 0.5).squeeze(-1)
            b_keys = new_keys[b, valid_indices]
            b_values = new_values[b, valid_indices]
            
            batch_keys.append(b_keys)
            batch_values.append(b_values)
            if b_keys.shape[0] > max_facts:
                max_facts = b_keys.shape[0]
                
        if max_facts == 0:
            return None, None
            
        padded_keys = torch.zeros(B, max_facts, D, device=device, dtype=new_keys.dtype)
        padded_values = torch.zeros(B, max_facts, D, device=device, dtype=new_values.dtype)
        padded_masks = torch.zeros(B, max_facts, device=device, dtype=torch.bool)
        
        for b in range(B):
            num_f = batch_keys[b].shape[0]
            if num_f > 0:
                padded_keys[b, :num_f] = batch_keys[b]
                padded_values[b, :num_f] = batch_values[b]
                padded_masks[b, :num_f] = True
                
        if self.keys is None:
            self.keys = padded_keys
            self.values = padded_values
            self.masks = padded_masks
        else:
            self.keys = torch.cat([self.keys, padded_keys], dim=1)
            self.values = torch.cat([self.values, padded_values], dim=1)
            self.masks = torch.cat([self.masks, padded_masks], dim=1)
            
        self.num_tokens = self.keys.shape[1]
        
        overflow_keys = None
        overflow_values = None
        
        if self.num_tokens > self.max_capacity:
            overflow = self.num_tokens - self.max_capacity
            overflow_keys = self.keys[:, :overflow, :]
            overflow_values = self.values[:, :overflow, :]
            
            self.keys = self.keys[:, overflow:, :]
            self.values = self.values[:, overflow:, :]
            self.masks = self.masks[:, overflow:]
            self.num_tokens = self.max_capacity
            
        return overflow_keys, overflow_values

    def exact_attention_retrieval(self, query):
        if self.keys is None or self.keys.shape[1] == 0:
            return torch.zeros_like(query)
            
        scores = torch.matmul(query, self.keys.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        mask_expanded = self.masks.unsqueeze(1)
        scores = scores.masked_fill(~mask_expanded, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        exact_context = torch.matmul(attn_weights, self.values)
        return exact_context


class SymphonyASHCLayer(nn.Module):
    """
    Symphony ASH-C Memory Layer.
    Combines routing, dynamic cache, FHRR recurrence, and Hopfield cleaning.
    """
    def __init__(self, hidden_dim, original_mlp, lambda_decay=0.99):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.original_mlp = original_mlp
        self.lambda_decay = lambda_decay
        self.freq_dim = hidden_dim // 2 + 1
        
        self.router = ASHCRouter(hidden_dim)
        self.phantom_cache = PhantomKVCache(hidden_dim, max_capacity=2048)
        
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.prism_w1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.prism_w2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.prism_w3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(hidden_dim)
        
        self.gate = nn.Parameter(torch.zeros(1))
        self.m_state = None

    def reset_state(self):
        self.m_state = None
        self.phantom_cache.reset_cache()

    def forward(self, x):
        B, S, D = x.shape
        device = x.device
        
        mlp_out = self.original_mlp(x)
        fact_selection = self.router(x)
        
        K = self.key_proj(x).float()
        V = self.value_proj(x).float()
        Q = self.query_proj(x).float()
        
        overflow_K, overflow_V = self.phantom_cache.add_to_cache(K, V, fact_selection)
        
        K_freq = torch.fft.rfft(K, dim=-1)
        V_freq = torch.fft.rfft(V, dim=-1)
        Q_freq = torch.fft.rfft(Q, dim=-1)
        
        K_abs = torch.sqrt(K_freq.real.pow(2) + K_freq.imag.pow(2) + 1e-12)
        K_norm_freq = K_freq / K_abs.to(dtype=torch.complex64)
        
        bound = K_norm_freq * V_freq
        
        if self.training or self.m_state is None or self.m_state.shape[0] != B:
            initial_state = torch.zeros(B, self.freq_dim, dtype=torch.complex64, device=device)
        else:
            initial_state = self.m_state
            
        if overflow_K is not None:
            O_K_freq = torch.fft.rfft(overflow_K, dim=-1)
            O_V_freq = torch.fft.rfft(overflow_V, dim=-1)
            O_K_abs = torch.sqrt(O_K_freq.real.pow(2) + O_K_freq.imag.pow(2) + 1e-12)
            O_K_norm = O_K_freq / O_K_abs.to(dtype=torch.complex64)
            overflow_bound = (O_K_norm * O_V_freq).mean(dim=1)
            initial_state = initial_state + overflow_bound
            
        scale = math.sqrt(1.0 - self.lambda_decay ** 2)
        
        if S <= 8192:
            t_indices = torch.arange(S, device=device).unsqueeze(1)
            i_indices = torch.arange(S, device=device).unsqueeze(0)
            power = torch.clamp(t_indices - i_indices, min=0)
            mask = (t_indices - i_indices >= 0).float()
            W = ((self.lambda_decay ** power) * mask).to(dtype=torch.complex64)
            
            outputs_rec_freq = scale * torch.matmul(W, bound)
            
            steps = torch.arange(1, S + 1, device=device, dtype=torch.float32)
            decay_factors = (self.lambda_decay ** steps).unsqueeze(0).unsqueeze(2).to(dtype=torch.complex64)
            outputs_rec_freq = outputs_rec_freq + initial_state.unsqueeze(1) * decay_factors
        else:
            outputs_rec_freq = torch.empty_like(bound)
            curr = initial_state.clone()
            for t in range(S):
                curr.mul_(self.lambda_decay).add_(bound[:, t], alpha=scale)
                outputs_rec_freq[:, t] = curr
            
        if not self.training:
            self.m_state = outputs_rec_freq[:, -1, :].detach()
            
        Q_abs = torch.sqrt(Q_freq.real.pow(2) + Q_freq.imag.pow(2) + 1e-12)
        Q_norm_freq = Q_freq / Q_abs.to(dtype=torch.complex64)
        outputs_rec_freq = outputs_rec_freq * torch.conj(Q_norm_freq)
        
        fhrr_rec = torch.fft.irfft(outputs_rec_freq, n=D, dim=-1).to(x.dtype)
        exact_rec = self.phantom_cache.exact_attention_retrieval(Q).to(x.dtype)
        
        combined_signal = torch.cat([fhrr_rec, exact_rec], dim=-1)
        gated = self.prism_w1(combined_signal) * self.act(self.prism_w2(combined_signal))
        clean_out = self.norm(self.prism_w3(gated))
        
        gate_val = torch.sigmoid(self.gate)
        output = mlp_out + gate_val * clean_out
        
        return output

def compute_contrastive_loss(clean_outputs, target_facts, distractor_facts, margin=1.0):
    pos_dist = F.pairwise_distance(clean_outputs, target_facts, p=2)
    neg_dist = F.pairwise_distance(clean_outputs, distractor_facts, p=2)
    loss = torch.clamp(margin + pos_dist - neg_dist, min=0.0).mean()
    return loss
