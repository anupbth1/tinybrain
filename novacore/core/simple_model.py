import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NovaConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

    def forward(self, x, mask=None):
        B, S, d = x.shape
        q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, d)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.W2(F.gelu(self.W1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.attn = Attention(d, n_heads)
        self.ff = FeedForward(d, d_ff)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class NovaModel(nn.Module):
    """Minimal Transformer for baseline comparison."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.hidden_size, config.num_attention_heads, config.intermediate_size)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight
        self.label_smoothing = 0.0  # harness sets via --label_smooth

    def forward(self, input_ids, labels=None, **kw):
        B, S = input_ids.shape
        x = self.embed(input_ids)
        mask = torch.triu(torch.full((S, S), float('-inf'), device=input_ids.device), diagonal=1)
        mask = mask[None, None, :, :]
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if labels is not None:
            shift = logits[..., :-1, :].contiguous()
            target = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, self.config.vocab_size), target.view(-1),
                                   ignore_index=-100, label_smoothing=self.label_smoothing)
            out["loss"] = loss
        return out


def create_transformer(vocab_size=10000, hidden_size=512, num_layers=6, num_heads=8):
    """Create a transformer with ~50M params."""
    cfg = NovaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=int(8 * hidden_size / 3),
        max_seq_length=256,
    )
    return NovaModel(cfg)