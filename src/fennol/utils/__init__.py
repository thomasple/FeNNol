from .spherical_harmonics import CG_SO3, generate_spherical_harmonics
from .atomic_units import AtomicUnits
from typing import Dict, Any
import jax
import jax.numpy as jnp


def minmaxone(x, name=""):
    print(name, x.min(), x.max(), (x**2).mean())


def minmaxone_jax(x, name=""):
    jax.debug.print(
        "{name}  {min}  {max}  {mean}",
        name=name,
        min=x.min(),
        max=x.max(),
        mean=(x**2).mean(),
    )


def mask_filter_1d(mask, max_size, *values_fill):
    cumsum = jnp.cumsum(mask,dtype=jnp.int32)
    scatter_idx = jnp.where(mask, cumsum - 1, max_size)
    outputs = []
    for value, fill in values_fill:
        shape = list(value.shape)
        shape[0] = max_size
        output = (
            jnp.full(shape, fill, dtype=value.dtype)
            .at[scatter_idx]
            .set(value, mode="drop")
        )
        outputs.append(output)
    if cumsum.size == 0:
        return outputs, scatter_idx, 0
    return outputs, scatter_idx, cumsum[-1]


def deep_update(
    mapping: Dict[Any, Any], *updating_mappings: Dict[Any, Any]
) -> Dict[Any, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


class Counter:
    def __init__(self, nseg, startsave=1):
        self.i = 0
        self.i_avg = 0
        self.nseg = nseg
        self.startsave = startsave

    @property
    def count(self):
        return self.i

    @property
    def count_avg(self):
        return self.i_avg

    @property
    def nsample(self):
        return max(self.count_avg - self.startsave + 1, 1)

    @property
    def is_reset_step(self):
        return self.count == 0

    def reset_avg(self):
        self.i_avg = 0

    def reset_all(self):
        self.i = 0
        self.i_avg = 0

    def increment(self):
        self.i = self.i + 1
        if self.i >= self.nseg:
            self.i = 0
            self.i_avg = self.i_avg + 1
