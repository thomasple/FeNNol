import flax.linen as nn
from typing import Sequence, Callable


class FullyConnectedNet(nn.Module):
    dims: Sequence[int]
    act: Callable = nn.silu
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        for d in self.dims[:-1]:
            x = nn.Dense(d, use_bias=self.use_bias)(x)
            x = self.act(x)
        x = nn.Dense(self.dims[-1], use_bias=self.use_bias)(x)
        return x
