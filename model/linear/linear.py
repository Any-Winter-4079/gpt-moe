from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from .fp8.custom.dense import project
from .fp8.transformer_engine import TransformerEngineLinear


class Linear(TransformerEngineLinear):
    def __init__(self, in_features: int, out_features: int, *, role: str, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.role = role
        self.precision = "default"
        self.backend = "custom"

    def forward(self, x: Tensor) -> Tensor:
        if self.precision == "fp8":
            if self.backend == "custom":
                out = project(x, [self.weight])[0]
                return out if self.bias is None else out + self.bias.to(out.dtype)
            return self.forward_fp8(x)
        return F.linear(x, self.weight, self.bias)
