import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable
import math

from .atomic_units import au
from .spherical_harmonics import CG_SO3, generate_spherical_harmonics


class SpeciesEncoding(nn.Module):
    dim: int = 16

    @nn.compact
    def __call__(self, species):
        conv_tensor = self.param(
            "conv_tensor",
            lambda key, shape: jax.random.normal(key, shape),
            (50, self.dim),
        )
        return conv_tensor[species]


class RadialEmbedding(nn.Module):
    end: float
    start: float = 0.0
    dim: int = 8

    @nn.compact
    def __call__(self, x):
        x = x - self.start
        c = self.end - self.start
        bessel_roots = jnp.arange(1, self.dim + 1) * math.pi
        return (
            math.sqrt(2.0 / c)
            * jnp.sin(x[:, None] * bessel_roots[None, :] / c)
            / x[:, None]
        )
