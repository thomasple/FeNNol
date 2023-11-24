import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Any, Dict, Union, Callable, Sequence, Optional
from ...utils import AtomicUnits as au


class RepulsionZBL(nn.Module):
    _graphs_properties: Dict
    graph_key: str = "graph"
    energy_key: Optional[str] = None
    trainable: bool = True

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]

        rijs = graph["distances"] / au.BOHR

        d = 0.46850 / au.BOHR
        p = 0.23
        alphas = [3.19980, 0.94229, 0.40290, 0.20162]
        if self.trainable:
            d = jnp.abs(
                self.param(
                    "d",
                    lambda key, p: jnp.asarray(d, dtype=rijs.dtype),
                    d,
                )
            )
            p = jnp.abs(
                self.param(
                    "p",
                    lambda key, p: jnp.asarray(p, dtype=rijs.dtype),
                    p,
                )
            )
            cs = 0.5 * jax.nn.softmax(
                self.param(
                    "cs",
                    lambda key, cs: jnp.asarray(cs, dtype=rijs.dtype),
                    [0.1130, 1.1445, 0.5459, -1.7514],
                )
            )
            alphas = jnp.abs(
                self.param(
                    "alphas",
                    lambda key, alphas: jnp.asarray(alphas, dtype=rijs.dtype),
                    alphas,
                )
            )
        else:
            cs = jnp.asarray(
                0.5 * np.array([0.18175273, 0.5098655, 0.28021213, 0.0281697])
            )
            alphas = jnp.asarray(alphas)

        Z = jnp.where(species > 0, species.astype(rijs.dtype), 0.0)
        Zi, Zj = Z[edge_src], Z[edge_dst]
        Zp = Z**p / d
        x = rijs * (Zp[edge_src] + Zp[edge_dst])
        phi = (cs[None, :] * jnp.exp(-alphas[None, :] * x[:, None])).sum(axis=-1)

        ereppair = Zi * Zj * phi / rijs * graph["switch"]

        erep_atomic = jax.ops.segment_sum(ereppair, edge_src, species.shape[0])

        energy_key = self.energy_key if self.energy_key is not None else self.name
        return {**inputs, energy_key: erep_atomic}
