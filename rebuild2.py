# -*- coding: utf-8 -*-
"""Simple rebuild script."""
import os

BASE = 'C:/project3/novacore'

def ensure(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

def w(relpath, content):
    path = os.path.join(BASE, relpath)
    ensure(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# __init__.py files
w('__init__.py', '# novacore package\n__version__ = "0.1.0"\n')

w('core/__init__.py', 'from .config import NovaConfig\nfrom .model import NovaModel, RMSNorm\n')

w('core/config.py', '''
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

w('core/model.py', '''
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

w('models/__init__.py', 'from .tiny_brain import TinyBrainConfig, TinyBrainModel\n')

w('models/tiny_brain.py', open('C:/project3/NovaCore/models/tiny_brain.py', 'r').read())

w('experiments/__init__.py', '# experiments\n')

w('experiments/phase3_verification.py', open('C:/project3/NovaCore/experiments/phase3_verification.py', 'r').read())

w('experiments/results/experiment_log.md', '''# Experiment Log
## Experiment #1 - Phase 3
**Date**: 2026-07-29
**Status**: PENDING
''')

print("All files created successfully!")

import subprocess
result = subprocess.run(['python', '-c', 'import sys; sys.path.insert(0,"C:/project3"); from novacore.models.tiny_brain import TinyBrainConfig; c=TinyBrainConfig(); print("Import OK, d=",c.hidden_size)'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:200])