import torch
from torch.distributions import Categorical


class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        ninf = torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype)
        logits = torch.where(mask, logits, ninf)
        super(CategoricalMasked, self).__init__(logits=logits)
