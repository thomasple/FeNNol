import pathlib
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Any, Dict, Union, Callable, Sequence, Optional, ClassVar
from ...utils import AtomicUnits as au
from ...utils.periodic_table import D3_COV_RADII,UFF_VDW_RADII


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
    proportional_regularization: bool = True
    d: float = 0.46850/au.BOHR
    p: float = 0.23
    alphas: Sequence[float] = (3.19980, 0.94229, 0.40290, 0.20162)
    cs: Sequence[float] = (0.18175273, 0.5098655, 0.28021213, 0.0281697)
    cs_logits: Sequence[float] = (0.1130, 1.1445, 0.5459, -1.7514)

    FID: ClassVar[str] = "REPULSION_ZBL"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]

        training = "training" in inputs.get("flags", {})

        rijs = graph["distances"] / au.BOHR

        d_ = self.d
        p_ = self.p
        assert len(self.alphas) == 4, "alphas must be a sequence of length 4"
        alphas_ = np.array(self.alphas, dtype=rijs.dtype)
        assert len(self.cs) == 4, "cs must be a sequence of length 4"
        cs_ =  np.array(self.cs, dtype=rijs.dtype)
        cs_ = 0.5 * cs_ / np.sum(cs_)
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
                    np.array(self.cs_logits, dtype=rijs.dtype),
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
                if self.proportional_regularization:
                    reg = jnp.asarray(
                        ((1 - alphas/alphas_) ** 2).sum()
                        + ((1 - cs/cs_) ** 2).sum()
                        + (1 - p/p_) ** 2
                        + (1 - d/d_) ** 2
                    ).reshape(1)
                else:
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
    
class RepulsionNLH(nn.Module):
    """ NLH pairwise repulsive potential with pair-specific coefficients up to Z=92

    FID: REPULSION_NLH

    ### Reference
    K. Nordlund, S. Lehtola, G. Hobler, Repulsive interatomic potentials calculated at three levels of theory, Phys. Rev. A 111, 032818 
    https://doi.org/10.1103/PhysRevA.111.032818
    """

    _graphs_properties: Dict
    graph_key: str = "graph"
    """The key for the graph input."""
    energy_key: Optional[str] = None
    """The key for the output energy."""
    _energy_unit: str = "Ha"
    """The energy unit of the model. **Automatically set by FENNIX**"""
    trainable: bool = False

    FID: ClassVar[str] = "REPULSION_NLH"

    @nn.compact
    def __call__(self, inputs):

        path = str(pathlib.Path(__file__).parent.resolve()) + "/nlh_coeffs.dat"
        DATA_NLH = np.loadtxt(path,usecols=np.arange(0,8)) 
        zmax = int(np.max(DATA_NLH[:, 0]))
        AB = np.zeros(((zmax+1)**2,6), dtype=np.float32)
        for i in range(DATA_NLH.shape[0]):
            z1 = int(DATA_NLH[i, 0])
            z2 = int(DATA_NLH[i, 1])
            AB[z1+zmax*z2] = DATA_NLH[i, 2:8]
            AB[z2+zmax*z1] = DATA_NLH[i, 2:8]
        AB = AB.reshape((zmax+1)**2, 3,2)

        species = inputs["species"]
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        rijs = graph["distances"]

        # coefficients (a1,a2,a3)
        CS = jnp.array(AB[:, :, 0], dtype=rijs.dtype)
        # exponents (b1,b2,b3)
        ALPHAS = jnp.array(AB[:, :, 1], dtype=rijs.dtype)

        if self.trainable:
            cfact = jnp.abs(self.param(
                "c_fact",
                lambda key: jnp.ones(CS.shape[1], dtype=CS.dtype),
            ))
            CS = CS * cfact[None,:]
            CS = CS / jnp.sum(CS, axis=1, keepdims=True)
            alphas_fact = jnp.abs(self.param(
                "alpha_fact",
                lambda key: jnp.ones(ALPHAS.shape[1], dtype=ALPHAS.dtype),
            ))
            ALPHAS = ALPHAS * alphas_fact[None,:]

        s12 = species[edge_src] + zmax*species[edge_dst]
        cs = CS[s12]
        alphas = ALPHAS[s12]

        if "alch_group" in inputs:
            switch = graph["switch_raw"]
            lambda_v = inputs["alch_vlambda"]
            alch_group = inputs["alch_group"]
            alch_alpha = inputs.get("alch_alpha", 0.)
            alch_m = inputs.get("alch_m", 2)

            mask = alch_group[edge_src] == alch_group[edge_dst]

            rijs = jnp.where(mask, rijs, (rijs**2 + alch_alpha**2 * (1 - lambda_v))**0.5)
            lambda_v = 0.5*(1-jnp.cos(jnp.pi*lambda_v))
            switch = jnp.where(
                mask,
                switch,
                (lambda_v**alch_m) * switch ,
            )
            # alphas = jnp.where(
            #     mask[:,None],
            #     alphas,
            #     lambda_v * alphas ,
            # )
        else:
            switch = graph["switch"]

        Z = jnp.where(species > 0, species.astype(rijs.dtype), 0.0)
        phi = (cs * jnp.exp(-alphas*rijs[:,None])).sum(axis=-1)

        ereppair = Z[edge_src]*Z[edge_dst] * phi * switch/rijs


        energy_unit = au.get_multiplier(self._energy_unit)
        erep_atomic = (energy_unit*0.5*au.BOHR)*jax.ops.segment_sum(ereppair, edge_src, species.shape[0])

        energy_key = self.energy_key if self.energy_key is not None else self.name
        output = {**inputs, energy_key: erep_atomic }

        return output
