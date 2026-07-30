"""
Emergency rebuild script - recreates entire project.
Uses 'novacore' (lowercase) for Windows compatibility.
"""
import os, shutil

BASE = 'C:/project3/novacore'

def write(path, content):
    full = os.path.join(BASE, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(content)
    return len(content)

print("Rebuilding novacore project...")

# ── __init__.py ──
write('__init__.py', '"""novacore - LLM Training Framework"""\n__version__ = "0.1.0"\n')

# ── requirements.txt ──
write('requirements.txt', 'torch>=2.0.0\ntransformers>=4.30.0\ndatasets>=2.12.0\nsafetensors>=0.3.0\ntokenizers>=0.13.0\nnumpy>=1.24.0\ntqdm>=4.64.0\ntensorboard>=2.13.0\n')

# ── setup.py ──
write('setup.py', 'from setuptools import setup, find_packages\nsetup(name="novacore", version="0.1.0", packages=find_packages(), python_requires=">=3.9")\n')

# ── README.md ──
write('README.md', '# novacore\n')

# ═══════════════════════════════════════════════════
# CORE MODULE
# ═══════════════════════════════════════════════════

write('core/__init__.py', 'from .config import NovaConfig\nfrom .model import NovaModel, RMSNorm\nfrom .optimizer import create_optimizer\n')

# config.py
write('core/config.py', '''
from dataclasses import dataclass
from typing import Optional

@dataclass
class NovaConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: Optional[int] = None
    max_seq_length: int = 2048
    intermediate_size: Optional[int] = None
    use_bias: bool = False
    initializer_range: float = 0.02
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
        if self.intermediate_size is None:
            self.intermediate_size = int(8 * self.hidden_size / 3)
            self.intermediate_size = ((self.intermediate_size + 63) // 64) * 64
''')

# model.py - simplified RMSNorm only
write('core/model.py', '''
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class NovaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        from .config import NovaConfig
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight
    def forward(self, input_ids, labels=None, **kw):
        x = self.embed(input_ids)
        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if labels is not None:
            shift = logits[..., :-1, :].contiguous()
            target = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(shift.view(-1, self.config.vocab_size), target.view(-1), ignore_index=-100)
            out["loss"] = loss
        return out
''')

# optimizer.py
write('core/optimizer.py', '''
import torch
def create_optimizer(model, lr=3e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr)
''')

# ═══════════════════════════════════════════════════
# MODELS - TinyBrain
# ═══════════════════════════════════════════════════

write('models/__init__.py', 'from .tiny_brain import TinyBrainConfig, TinyBrainModel, ThinkingStep, LearnedMemory, ConfidenceGate, AdaptiveThinkingCell, SelfCorrection\n')

# tiny_brain.py - SELF-CONTAINED (no novacore imports)
write('models/tiny_brain.py', '''
"""
TinyBrain — 0.5B Adaptive Thinking Model
Complete implementation with no external package dependencies.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── RMSNorm (inlined to avoid cross-package imports) ──
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
    """§2 — x_{{t+1}} = x_t + tanh(W_c·O + b_c) ⊙ SiLU(W_e1·O)⊙W_e2·O,  O = W_o·LN(x)"""
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.obs_norm = RMSNorm(d, config.layer_norm_eps)
        self.W_o = nn.Linear(d, d, bias=False)
        self.W_c = nn.Linear(d, d, bias=True)
        self.W_e1 = nn.Linear(d, d, bias=False)
        self.W_e2 = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x_t):
        O = self.W_o(self.obs_norm(x_t))
        C = torch.tanh(self.W_c(O))
        E = F.silu(self.W_e1(O)) * self.W_e2(O)
        return x_t + self.dropout(C * E)

class LearnedMemory(nn.Module):
    """§3 — Read: α·M, Write: gated erase-then-add, Compress: importance decay"""
    def __init__(self, config):
        super().__init__()
        d, m = config.hidden_size, config.memory_slots
        self.read_norm = RMSNorm(d, config.layer_norm_eps)
        self.W_read = nn.Linear(d, d, bias=False)
        self.W_value = nn.Linear(d, d, bias=False)
        self.W_write = nn.Linear(d, m, bias=True)
        self.W_erase = nn.Linear(d, m, bias=True)
        self.W_imp = nn.Linear(d, 1, bias=True)
        self.M_init = nn.Parameter(torch.randn(m, d) * config.initializer_range)
        self.compress_rate = config.memory_compress_rate
    def forward(self, x, memory=None):
        batch, seq, d = x.shape
        m = self.M_init.shape[0]
        if memory is None:
            memory = self.M_init.unsqueeze(0).expand(batch, -1, -1)
        q = self.W_read(self.read_norm(x))
        alpha = F.softmax(torch.bmm(q, memory.transpose(1,2)) / math.sqrt(d), dim=-1)
        r = torch.bmm(alpha, memory)
        v = self.W_value(x)
        gw = torch.sigmoid(self.W_write(x)).mean(dim=1)
        ge = torch.sigmoid(self.W_erase(x)).mean(dim=1)
        vs = v.mean(dim=1)
        memory = memory * (1 - ge.unsqueeze(-1)) + gw.unsqueeze(-1) * vs.unsqueeze(1)
        I = torch.sigmoid(self.W_imp(memory))
        if self.training:
            memory = memory * (1 - (1 - I) * self.compress_rate)
        else:
            memory = memory * (I > 0.1).float()
        return r, memory

class ConfidenceGate(nn.Module):
    """§4 — c_t = σ(W_conf·[x_t; x_t-x_0] + step_embed + b_conf)"""
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.step_embed = nn.Embedding(config.max_think_steps + 1, d)
        self.conf_norm = RMSNorm(2*d, config.layer_norm_eps)
        self.W_conf = nn.Linear(2*d, 1, bias=True)
        self.threshold = config.confidence_threshold
    def forward(self, x_t, x_0, step):
        batch, seq, d = x_t.shape
        step_idx = torch.full((batch, seq), min(step, self.step_embed.num_embeddings-1), device=x_t.device, dtype=torch.long)
        e = self.step_embed(step_idx)
        inp = torch.cat([x_t, x_t - x_0], dim=-1) + e
        inp = self.conf_norm(inp)
        logit = self.W_conf(inp)
        c = torch.sigmoid(logit)
        if self.training:
            noise = -torch.log(-torch.log(torch.rand_like(logit).clamp(1e-10, 1-1e-10)))
            h = torch.sigmoid((logit + noise) / 1.0)
        else:
            h = (c >= self.threshold).float()
        return c, h

class AdaptiveThinkingCell(nn.Module):
    """§5 — AdaptiveThinkingCell: Think → Memory → Confidence → Halt/Continue"""
    def __init__(self, config, cell_idx=0):
        super().__init__()
        d = config.hidden_size
        self.cell_idx = cell_idx
        self.max_steps = config.max_think_steps
        self.min_steps = config.min_think_steps
        self.W_in = nn.Linear(d, d, bias=False)
        self.think = ThinkingStep(config)
        self.memory = LearnedMemory(config)
        self.confidence = ConfidenceGate(config)
        self.W_out = nn.Linear(d, d, bias=False)
        self.lambda_c = config.confidence_penalty_weight
        self.lambda_s = config.step_penalty_weight
    def forward(self, x, memory=None):
        x = self.W_in(x)
        x0 = x
        steps, conf_sum = 0, 0.0
        for step in range(self.max_steps):
            x = self.think(x)
            r, memory = self.memory(x, memory)
            x = x + r
            steps += 1
            c, h = self.confidence(x, x0, step)
            conf_sum += c.mean().item()
            if step >= self.min_steps - 1:
                if not self.training and h.mean() > 0.5:
                    break
                if self.training and h.mean() > 0.8:
                    break
        out = self.W_out(x)
        aux = {}
        if self.training:
            aux["confidence_penalty"] = self.lambda_c * (1 - conf_sum / max(steps,1))
            aux["step_penalty"] = self.lambda_s * (steps / self.max_steps) ** 2
        return out, memory, aux

class SelfCorrection(nn.Module):
    """§6 — Verify + refine output"""
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.K = config.correction_steps
        self.verify = nn.Sequential(RMSNorm(d, config.layer_norm_eps), nn.Linear(d, 1))
        self.refine = nn.Sequential(RMSNorm(d, config.layer_norm_eps), nn.Linear(d, d), nn.SiLU())
        self.gate = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        for _ in range(self.K):
            v = torch.sigmoid(self.verify(x))
            r = self.refine(x)
            g = self.gate(x) * (1 - v)
            x = x + self.dropout(r * g)
        return x, v

class TinyBrainModel(nn.Module):
    """§10 — Complete TinyBrain Model"""
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
        self.lambda_m = config.memory_l2_weight
        self._init_weights()
    def _init_weights(self):
        std = self.config.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, std)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(0, std)
    def forward(self, input_ids, labels=None, memory_states=None):
        batch, seq = input_ids.shape
        K = self.config.num_cells
        x = self.embed(input_ids)
        if memory_states is None:
            memory_states = [None] * K
        aux = {}
        logs = []
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
            L_mem = sum(m.pow(2).mean() for m in new_mems if m is not None) * self.lambda_m
            L_aux = sum(aux.values()) if aux else 0
            out["loss"] = L_lm + L_mem + L_aux
            out["lm_loss"] = L_lm
        return out
    @torch.no_grad()
    def generate(self, input_ids, max_new=100, temp=0.8, top_k=50, top_p=0.95, eos=None):
        self.eval()
        gen = input_ids.clone()
        mem = None
        for _ in range(max_new):
            o = self(gen, memory_states=mem)
            logits = o["logits"][:, -1, :] / temp
            mem = o["memory_states"]
            if top_k > 0:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, -1:]] = float('-inf')
            if top_p < 1.0:
                s, si = torch.sort(logits, descending=True, dim=-1)
                cp = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
                mask = cp > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                logits = logits.masked_fill(mask.scatter(-1, si, mask), float('-inf'))
            p = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(p, 1)
            gen = torch.cat([gen, nxt], dim=-1)
            if eos is not None and (nxt == eos).any(): break
        self.train()
        return gen
''')

# ═══════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════

write('experiments/__init__.py', '"""novacore experiments"""\n')

write('experiments/phase3_verification.py', '''
"""Phase 3 — 15 Mathematical Diagnostics"""
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from ..models.tiny_brain import TinyBrainConfig, ThinkingStep, LearnedMemory, ConfidenceGate, AdaptiveThinkingCell, TinyBrainModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cfg(d=64):
    return TinyBrainConfig(hidden_size=d, num_think_heads=4, memory_slots=8, confidence_hidden=max(16,d//4), max_think_steps=32, min_think_steps=1, output_mlp_hidden=d*2)

# 1. Fixed-point convergence
def test_convergence():
    step = ThinkingStep(cfg()).to(DEVICE)
    x = torch.randn(4, 16, 64, device=DEVICE)
    deltas = []
    for _ in range(100):
        xn = step(x); d_ = (xn - x).norm(dim=-1).mean().item(); deltas.append(d_); x = xn
    ratio = deltas[-1] / max(deltas[0], 1e-10)
    return {"id":"brain_1","name":"Fixed-point convergence","status":"PASS" if ratio<0.01 else "FAIL","final/initial":round(ratio,6),"final_delta":round(deltas[-1],6)}

# 2. Spectral radius
def test_spectral():
    step = ThinkingStep(cfg()).to(DEVICE)
    x = torch.randn(1, 1, 64, device=DEVICE)
    v = torch.randn_like(x)
    for _ in range(10):
        xi = x.detach().clone().requires_grad_()
        xo = step(xi)
        Jv = torch.autograd.grad(xo, xi, v, retain_graph=True)[0]
        sn = Jv.norm().item(); v = Jv / (sn + 1e-10)
    return {"id":"brain_2","name":"Spectral radius","status":"PASS" if sn<1.5 else "FAIL","est_spectral_radius":round(sn,4)}

# 3. Jacobian norm
def test_jacobian():
    step = ThinkingStep(cfg()).to(DEVICE)
    x = torch.randn(2, 8, 64, device=DEVICE)
    norms = []
    for _ in range(30):
        x.requires_grad_(True); xn = step(x)
        g = torch.autograd.grad(xn.norm(), x, retain_graph=False)[0]
        norms.append(g.norm().item()); x = xn.detach()
    return {"id":"brain_3","name":"Jacobian norm","status":"PASS" if max(norms)<10 else "FAIL","max_jacobian":round(max(norms),4),"final_jacobian":round(norms[-1],4)}

# 4. Gradient flow
def test_gradient():
    cell = AdaptiveThinkingCell(cfg()).to(DEVICE); cell.train()
    x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
    out, _, _ = cell(x)
    loss = out.pow(2).mean(); loss.backward()
    gn = x.grad.norm().item()
    ok = 1e-8 < gn < 1e6
    return {"id":"brain_4","name":"Gradient flow","status":"PASS" if ok else "FAIL","input_grad_norm":round(gn,6)}

# 5. Hidden state variance
def test_variance():
    step = ThinkingStep(cfg()).to(DEVICE)
    x = torch.randn(4, 16, 64, device=DEVICE)
    vars_ = []
    for _ in range(50): x = step(x); vars_.append(x.var().item())
    r = vars_[-1] / max(vars_[0], 1e-10)
    return {"id":"brain_5","name":"Hidden variance","status":"PASS" if 0.01<r<100 else "FAIL","var_ratio":round(r,4)}

# 6. Write collisions
def test_write_collisions():
    mem = LearnedMemory(cfg()).to(DEVICE)
    pats = []; ms = None
    for i in range(20):
        x = torch.randn(1, 8, 64, device=DEVICE) * (i+1)
        r, ms = mem(x, ms)
        with torch.no_grad(): gw = torch.sigmoid(mem.W_write(x.mean(dim=1,keepdim=True)))
        pats.append(gw.squeeze().cpu().numpy())
    P = np.array(pats)
    if P.ndim==2 and P.shape[0]>1:
        c = np.corrcoef(P); ac = (c.sum()-c.trace())/(c.size-c.shape[0])
    else: ac=0.0
    return {"id":"memory_6","name":"Write collisions","status":"PASS" if ac<0.5 else "FAIL","avg_corr":round(float(ac),4)}

# 7. Slot utilization
def test_utilization():
    mem = LearnedMemory(cfg()).to(DEVICE); ms=None
    for _ in range(50): x = torch.randn(1,16,64,device=DEVICE); r,ms = mem(x,ms)
    active = (ms.norm(dim=-1).squeeze() > 0.01).float().mean().item()
    return {"id":"memory_7","name":"Slot utilization","status":"PASS" if active>0.2 else "FAIL","active_ratio":round(active,4)}

# 8. Retention
def test_retention():
    mem = LearnedMemory(cfg()).to(DEVICE)
    ms = None; target = torch.randn(1,1,64,device=DEVICE)
    r, ms = mem(target, ms)
    for _ in range(50): x = torch.randn(1,16,64,device=DEVICE); r,ms = mem(x,ms)
    rf,_ = mem(target, ms)
    err = (rf - target).norm().item()
    return {"id":"memory_8","name":"Retention (50 steps)","status":"PASS" if err<10 else "FAIL","error":round(err,4)}

# 10. Compression ratio
def test_compression():
    mem = LearnedMemory(cfg()).to(DEVICE); ms=None
    for _ in range(100): x = torch.randn(1,64,64,device=DEVICE); r,ms = mem(x,ms)
    active = (ms.norm(dim=-1).squeeze() > 0.01).sum().item()
    ratio = (64*64) / max(active*64,1)
    return {"id":"memory_10","name":"Compression ratio","status":"PASS" if ratio>1 else "FAIL","ratio":round(ratio,2),"active_slots":active}

# 11. ECE
def test_ece():
    gate = ConfidenceGate(cfg()).to(DEVICE); gate.eval()
    x0 = torch.randn(4,16,64,device=DEVICE)
    confs, deltas_ = [], []
    for t in range(20):
        xt = x0 + torch.randn_like(x0)*0.1*(t+1)
        c,_ = gate(xt, x0, t); confs.append(c.mean().item())
        deltas_.append((xt-x0).norm().item())
    conf = np.array(confs); dA = np.array(deltas_)
    dn = dA / max(dA); ece = np.abs(conf - (1-dn)).mean()
    return {"id":"conf_11","name":"ECE","status":"PASS" if ece<0.3 else "FAIL","ece":round(float(ece),4)}

# 12. Brier
def test_brier():
    gate = ConfidenceGate(cfg()).to(DEVICE); gate.eval()
    x0 = torch.randn(4,16,64,device=DEVICE); scores=[]
    for t in range(20):
        xt = x0 + torch.randn_like(x0)*0.1*(t+1)
        c,_ = gate(xt, x0, t)
        delta = (xt-x0).norm(dim=-1)
        outcome = (delta < delta.median()).float()
        s = ((c.squeeze(-1)-outcome)**2).mean().item(); scores.append(s)
    return {"id":"conf_12","name":"Brier Score","status":"PASS" if sum(scores)/len(scores)<0.25 else "FAIL","brier":round(sum(scores)/len(scores),4)}

# 13. Correlation
def test_corr():
    gate = ConfidenceGate(cfg()).to(DEVICE); gate.eval()
    x0 = torch.randn(2,8,64,device=DEVICE)
    cL,dL=[],[]
    for t in range(30):
        xt = x0 + torch.randn_like(x0)*0.05*(t+1)
        c,_ = gate(xt, x0, t)
        cL.extend(c.squeeze(-1).flatten().tolist())
        dL.extend((xt-x0).norm(dim=-1).flatten().tolist())
    corr = np.corrcoef(cL, dL)[0,1]
    return {"id":"conf_13","name":"Conf vs Δ corr","status":"PASS" if corr>0.3 else "FAIL","corr":round(float(corr),4)}

# 14. Early stop accuracy
def test_early_stop():
    cell = AdaptiveThinkingCell(cfg()).to(DEVICE); cell.eval()
    x = torch.randn(4,16,64,device=DEVICE)
    xf,_,_ = cell(x)
    res = []
    for th in [0.5,0.7,0.85,0.95]:
        cell.confidence.threshold = th
        xe,_,log = cell(x)
        err = (xe-xf).norm().item()
        res.append({"threshold":th,"error":round(err,4),"steps":log["steps"]})
    return {"id":"conf_14","name":"Early-stop accuracy","status":"INFO","results":res}

# 15. Compute profile
def test_compute():
    c = TinyBrainConfig(hidden_size=512, num_cells=4, memory_slots=32, max_think_steps=16)
    model = TinyBrainModel(c).to(DEVICE); model.eval()
    inp = torch.randint(0, 1000, (2, 128), device=DEVICE)
    for _ in range(3): model(inp)
    import time
    t0 = time.perf_counter()
    for _ in range(10): model(inp)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    lat = (t1-t0)/10; tok_s = (2*128)/lat
    params = sum(p.numel() for p in model.parameters())
    vram = params*4/(1024*1024)
    return {"id":"compute_15","name":"Compute profile","status":"INFO","latency_s":round(lat,4),"tokens_per_sec":round(tok_s,0),"vram_mb":round(vram,2),"params":params}

def run_all():
    tests = [
        ("Brain", test_convergence),("Brain", test_spectral),("Brain", test_jacobian),
        ("Brain", test_gradient),("Brain", test_variance),
        ("Memory", test_write_collisions),("Memory", test_utilization),
        ("Memory", test_retention),("Memory", test_compression),
        ("Conf", test_ece),("Conf", test_brier),("Conf", test_corr),("Conf", test_early_stop),
        ("Compute", test_compute),
    ]
    results = {}
    passed = 0
    for cat, fn in tests:
        r = fn(); s = r["status"]
        results.setdefault(cat,[]).append(r)
        if s=="PASS": passed+=1
        print(f"  [{s}] {r.get('name','?')}: {r.get('status','?')}")
    print(f"\\n{'='*60}\\nPhase 3: {passed}/{len(tests)} passed\\n{'='*60}")
    import json
    out = Path(__file__).parent / "results" / f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out,"w") as f: json.dump(results,f,indent=2,default=str)
    print(f"Saved: {out}")
    return results

if __name__=="__main__": run_all()
''')

# ── experiment_log.md ──
write('experiments/results/experiment_log.md', '''# Experiment Log

---

## Experiment #1 — Phase 3: Mathematical Verification

**Date**: 2026-07-29
**Status**: ⏳ Running

### Results

| # | Metric | Status | Value |
|---|--------|--------|-------|
| 1 | Fixed-point convergence | ? | ? |
| 2 | Spectral radius | ? | ? |
| 3 | Jacobian norm | ? | ? |
| 4 | Gradient flow | ? | ? |
| 5 | Hidden variance | ? | ? |
| 6 | Write collisions | ? | ? |
| 7 | Slot utilization | ? | ? |
| 8 | Retention | ? | ? |
| 9 | Compression ratio | ? | ? |
| 10 | ECE | ? | ? |
| 11 | Brier | ? | ? |
| 12 | Conf vs Δ corr | ? | ? |
| 13 | Early-stop | ? | ? |
| 14 | Compute profile | ? | ? |

### Conclusion
*PENDING*

### Next Action
*PENDING*
''')

# ── run_experiments.py ──
write('../run_experiments.py', '''"""Run novacore experiments"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from novacore.experiments.phase3_verification import run_all as run_p3

if __name__ == "__main__":
    print("Running Phase 3: 14 diagnostics")
    run_p3()
    print("\\nDone. Check novacore/experiments/results/")
''')

print("\\n✅ Rebuild complete! All files created.")
print(f"Total: {sum(len(open(os.path.join(dp,f)).read()) for dp,dn,fn in os.walk(BASE) for f in fn if f.endswith('.py') or f.endswith('.md'))} bytes")