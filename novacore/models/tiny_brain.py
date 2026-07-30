"""
TinyBrain — Self-contained implementation.
BUGFIX: W_in and W_out now identity init to prevent signal collapse.
Previously: random init causing 1600x norm collapse (1.28 → 0.0008)
Now: identity init preserves hidden state norm through cells.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


@dataclass
class TinyBrainConfig:
    vocab_size: int = 152064
    hidden_size: int = 1024
    memory_slots: int = 64
    num_cells: int = 6
    num_think_heads: int = 8
    max_think_steps: int = 32
    min_think_steps: int = 1
    confidence_threshold: float = 0.85
    memory_compress_rate: float = 0.01
    correction_steps: int = 1
    output_mlp_hidden: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    confidence_penalty_weight: float = 0.01
    step_penalty_weight: float = 0.1
    memory_l2_weight: float = 0.001


class ThinkingStep(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.ln = RMSNorm(d, config.layer_norm_eps)
        self.W_o = nn.Linear(d, d, bias=False)
        self.W_c = nn.Linear(d, d, bias=True)
        self.W_e1 = nn.Linear(d, d, bias=False)
        self.W_e2 = nn.Linear(d, d, bias=False)
        self.do = nn.Dropout(config.dropout)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        O = self.W_o(self.ln(x))
        C = torch.tanh(self.W_c(O))
        E = F.silu(self.W_e1(O)) * self.W_e2(O)
        delta = torch.tanh(C * E)
        return x + torch.tanh(self.gamma) * self.do(delta)


class LearnedMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, m = config.hidden_size, config.memory_slots
        self.ln = RMSNorm(d, config.layer_norm_eps)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_w = nn.Linear(d, m, bias=True)
        self.W_e = nn.Linear(d, m, bias=True)
        self.W_i = nn.Linear(d, 1, bias=True)
        self.M0 = nn.Parameter(torch.randn(m, d) * config.initializer_range)
        self.eta = config.memory_compress_rate
        self.out_gate = nn.Parameter(torch.zeros(1))
    def forward(self, x, mem=None):
        B, S, d = x.shape
        m = self.M0.shape[0]
        if mem is None:
            mem = self.M0.unsqueeze(0).expand(B, -1, -1)
        q = self.W_q(self.ln(x))
        a = F.softmax(torch.bmm(q, mem.transpose(1,2)) / math.sqrt(d), dim=-1)
        r = torch.bmm(a, mem)
        v = self.W_v(x)
        gw = torch.sigmoid(self.W_w(x)).mean(dim=1)
        ge = torch.sigmoid(self.W_e(x)).mean(dim=1)
        vs = v.mean(dim=1)
        mem = mem * (1 - ge.unsqueeze(-1)) + gw.unsqueeze(-1) * vs.unsqueeze(1)
        I = torch.sigmoid(self.W_i(mem))
        if self.training:
            mem = mem * (1 - (1 - I) * self.eta)
        else:
            mem = mem * (I > 0.1).float()
        r_gated = torch.tanh(self.out_gate) * r
        return r_gated, mem


class ConfidenceGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.step_emb = nn.Embedding(config.max_think_steps + 1, 2*d)
        self.ln = RMSNorm(2*d, config.layer_norm_eps)
        self.proj = nn.Linear(2*d, 1, bias=True)
        self.thresh = config.confidence_threshold
    def forward(self, x, x0, step):
        B, S, d = x.shape
        sidx = torch.full((B,S), min(step, self.step_emb.num_embeddings-1), device=x.device, dtype=torch.long)
        e = self.step_emb(sidx)
        inp = torch.cat([x, x - x0], dim=-1) + e
        inp = self.ln(inp)
        logit = self.proj(inp)
        c = torch.sigmoid(logit)
        if self.training:
            noise = -torch.log(-torch.log(torch.rand_like(logit).clamp(1e-10, 1-1e-10)))
            h = torch.sigmoid((logit + noise) / 1.0)
        else:
            h = (c >= self.thresh).float()
        return c, h


class AdaptiveThinkingCell(nn.Module):
    def __init__(self, config, idx=0):
        super().__init__()
        d = config.hidden_size
        self.idx = idx
        self.max_s = config.max_think_steps
        self.min_s = config.min_think_steps
        # BUGFIX: Identity init prevents 1600x signal collapse
        self.W_in = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W_in.weight)
        self.W_in.skip_init = True
        self.think = ThinkingStep(config)
        self.mem = LearnedMemory(config)
        self.conf = ConfidenceGate(config)
        self.W_out = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W_out.weight)
        self.W_out.skip_init = True
        self.lc = config.confidence_penalty_weight
        self.ls = config.step_penalty_weight
    def forward(self, x, memory=None):
        x = self.W_in(x)  # Now identity, no collapse
        x0 = x
        steps, csum = 0, 0.0
        for t in range(self.max_s):
            x = self.think(x)
            r, memory = self.mem(x, memory)
            x = x + torch.tanh(r)
            steps += 1
            c, h = self.conf(x, x0, t)
            csum += c.mean().item()
            if t >= self.min_s - 1:
                if not self.training and h.mean() > 0.5: break
                if self.training and h.mean() > 0.8: break
        out = self.W_out(x)  # Now identity, no collapse
        aux = {}
        if self.training:
            aux["conf"] = self.lc * (1 - csum / max(steps, 1))
            aux["step"] = self.ls * (steps / self.max_s) ** 2
        return out, memory, aux


class SelfCorrection(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.verify = nn.Sequential(RMSNorm(d, config.layer_norm_eps), nn.Linear(d, 1))
        self.refine = nn.Sequential(RMSNorm(d, config.layer_norm_eps), nn.Linear(d, d), nn.SiLU())
        self.gate = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        self.do = nn.Dropout(config.dropout)
    def forward(self, x):
        for _ in range(self.correction_steps if hasattr(self, 'correction_steps') else 1):
            v = torch.sigmoid(self.verify(x))
            r = self.refine(x)
            g = self.gate(x) * (1 - v)
            x = x + self.do(r * g)
        return x, v


class TinyBrainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.cells = nn.ModuleList([AdaptiveThinkingCell(config, i) for i in range(config.num_cells)])
        self.sc = SelfCorrection(config)
        self.out_mlp = nn.Sequential(
            RMSNorm(config.hidden_size, config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.output_mlp_hidden),
            nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.output_mlp_hidden, config.hidden_size),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight
        self.lm = config.memory_l2_weight
        self._init()
    def _init(self):
        s = self.config.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, 'skip_init') and m.skip_init:
                    continue
                m.weight.data.normal_(0, s)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, s)
    def forward(self, input_ids, labels=None, memory_states=None):
        K = self.config.num_cells
        x = self.embed(input_ids)
        if memory_states is None:
            memory_states = [None] * K
        aux = {}
        new_mems = []
        for i, cell in enumerate(self.cells):
            x, mem, a = cell(x, memory_states[i])
            new_mems.append(mem)
            for k, v in a.items(): aux[f"c{i}_{k}"] = v
        x, vs = self.sc(x)
        x = self.out_mlp(x)
        logits = self.lm_head(x)
        out = {"logits": logits, "memory_states": new_mems, "verification_scores": vs}
        if labels is not None:
            shift = logits[..., :-1, :].contiguous()
            target = labels[..., 1:].contiguous()
            L_lm = F.cross_entropy(shift.view(-1, self.config.vocab_size), target.view(-1), ignore_index=-100)
            L_mem = sum(m.pow(2).mean() for m in new_mems if m is not None) * self.lm
            L_aux = sum(aux.values()) if aux else 0
            out["loss"] = L_lm + L_mem + L_aux
        return out