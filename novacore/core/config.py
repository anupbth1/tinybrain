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