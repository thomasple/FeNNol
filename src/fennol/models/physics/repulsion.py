import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Any, Dict, Union, Callable, Sequence, Optional, ClassVar
from ...utils import AtomicUnits as au


class RepulsionZBL(nn.Module):
    """Repulsion energy based on the Ziegler-Biersack-Littmark potential

    FID: REPULSION_ZBL

    ### Reference
    J. F. Ziegler & J. P. Biersack , The Stopping and Range of Ions in Matter

    """

    _graphs_properties: Dict
    graph_key: str = "graph"
    """The key for the graph input."""
    energy_key: Optional[str] = None
    """The key for the output energy."""
    trainable: bool = True
    """Whether the parameters are trainable."""
    _energy_unit: str = "Ha"
    """The energy unit of the model. **Automatically set by FENNIX**"""

    FID: ClassVar[str] = "REPULSION_ZBL"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]

        training = "training" in inputs.get("flags", {})

        rijs = graph["distances"] / au.BOHR

        d_ = 0.46850 / au.BOHR
        p_ = 0.23
        alphas_ = np.array([3.19980, 0.94229, 0.40290, 0.20162])
        cs_ = 0.5 * np.array([0.18175273, 0.5098655, 0.28021213, 0.0281697])
        if self.trainable:
            d = jnp.abs(
                self.param(
                    "d",
                    lambda key, d: jnp.asarray(d, dtype=rijs.dtype),
                    d_,
                )
            )
            p = jnp.abs(
                self.param(
                    "p",
                    lambda key, p: jnp.asarray(p, dtype=rijs.dtype),
                    p_,
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
                    alphas_,
                )
            )

            if training:
                reg = jnp.asarray(
                    ((alphas_ - alphas) ** 2).sum()
                    + ((cs_ - cs) ** 2).sum()
                    + (p_ - p) ** 2
                    + (d_ - d) ** 2
                ).reshape(1)
        else:
            cs = jnp.asarray(cs_)
            alphas = jnp.asarray(alphas_)
            d = d_
            p = p_

        if "alch_group" in inputs:
            switch = graph["switch_raw"]
            lambda_v = inputs["alch_vlambda"]
            alch_group = inputs["alch_group"]
            alch_alpha = inputs.get("alch_alpha", 0.5)
            alch_m = inputs.get("alch_m", 2)

            mask = alch_group[edge_src] == alch_group[edge_dst]

            rijs = jnp.where(mask, rijs, (rijs**2 + alch_alpha**2 * (1 - lambda_v))**0.5)
            lambda_v = 0.5*(1-jnp.cos(jnp.pi*lambda_v))
            switch = jnp.where(
                mask,
                switch,
                (lambda_v**alch_m) * switch ,
            )
        else:
            switch = graph["switch"]

        Z = jnp.where(species > 0, species.astype(rijs.dtype), 0.0)
        Zij = Z[edge_src]*Z[edge_dst]
        Zp = Z**p / d
        x = rijs * (Zp[edge_src] + Zp[edge_dst])
        phi = (cs[None, :] * jnp.exp(-alphas[None, :] * x[:, None])).sum(axis=-1)

        ereppair = Zij * phi / rijs * switch

        erep_atomic = jax.ops.segment_sum(ereppair, edge_src, species.shape[0])

        energy_unit = au.get_multiplier(self._energy_unit)
        energy_key = self.energy_key if self.energy_key is not None else self.name
        output = {**inputs, energy_key: erep_atomic * energy_unit}
        if self.trainable and training:
            output[energy_key + "_regularization"] = reg

        return output
