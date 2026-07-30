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
            loss = nn.functional.cross_entropy(
                shift.view(-1, self.config.vocab_size),
                target.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out