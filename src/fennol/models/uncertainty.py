import flax.linen as nn
from typing import Any, Sequence, Callable, Union
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
import dataclasses
from typing import Optional, Tuple, Dict, List, Union
from ..utils.periodic_table import PERIODIC_TABLE


class ConstrainEvidence(nn.Module):
    key: str
    output_key: Optional[str] = None
    beta_scale: Union[str,float] = 1.0
    alpha_init: float = 2.
    nu_init: float = 1.
    chemical_shift: Optional[float] = None
    trainable_beta: bool = False
    constant_beta: bool = False
    trainable_alpha: bool = False
    constant_alpha: bool = False
    trainable_nu: bool = False
    constant_nu: bool = False
    nualpha_coupling: Optional[float] = None
    graph_key: Optional[str] = None
    self_weight: float = 10.
    # target_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        dim = 3 
        assert (
            x.shape[-1] == dim
        ), f"Dimension {-1} must be {dim}, got {x.shape[-1]}"
        nui, alphai, betai = jnp.split(x, 3, axis=-1)

        if self.chemical_shift is not None:
            nu_shift = jnp.abs(self.param(
                "nu_shift",
                lambda key, shape: jnp.ones(shape)*self.chemical_shift,
                (len(PERIODIC_TABLE),),
            ))[inputs["species"]]
            if self.graph_key is not None:
                graph = inputs[self.graph_key]
                edge_dst = graph["edge_dst"]
                edge_src = graph["edge_src"]
                switch = graph["switch"]
                nushift_neigh = jax.ops.segment_sum(nu_shift[edge_dst]*switch, edge_src, nu_shift.shape[0])
                norm = self.self_weight+jax.ops.segment_sum(switch, edge_src, nu_shift.shape[0])
                nu_shift = (self.self_weight*nu_shift + nushift_neigh)/norm
            nu_shift = nu_shift[:,None]
        else:
            nu_shift = 1.0
        
        if self.nualpha_coupling is None:
            if self.constant_alpha:
                if self.trainable_alpha:
                    alphai = 1+jnp.abs(self.param(
                        "alpha",
                        lambda key: jnp.asarray(self.alpha_init-1),
                    ))*jnp.ones_like(alphai)
                else:
                    assert self.alpha_init >1, "alpha_init must be >1"
                    alphai = self.alpha_init*jnp.ones_like(alphai)
            elif self.constant_nu:
                alphai = 1 + (self.alpha_init-1)*nu_shift*jax.nn.softplus(alphai)
            else:
                alphai = 1 + (self.alpha_init-1)*jax.nn.softplus(alphai)
            
            if self.constant_nu:
                if self.trainable_nu:
                    nui = 1.0e-5 + jnp.abs(self.param(
                        "nu",
                        lambda key: jnp.asarray(self.nu_init),
                    ))*jnp.ones_like(nui)
                else:
                    nui = self.nu_init*jnp.ones_like(nui)
            else:
                nui = 1.0e-5 + self.nu_init*nu_shift*jax.nn.softplus(nui)
        else:
            alphai = 1 + nu_shift*jax.nn.softplus(alphai)
            if self.trainable_nu:
                nualpha_coupling = jnp.abs(self.param(
                    "nualpha_coupling",
                    lambda key: jnp.asarray(self.nualpha_coupling),
                ))
            else:
                nualpha_coupling = self.nualpha_coupling
            nui=nualpha_coupling * 2*alphai

        if self.constant_beta:
            if self.trainable_beta:
                betai = (jax.nn.softplus(self.param(
                    "beta",
                    lambda key: jnp.asarray(0.),
                ))/np.log(2.))*jnp.ones_like(alphai)
            else:
                betai = jnp.ones_like(alphai)
        else:
            betai = jax.nn.softplus(betai)
            
        betai = self.beta_scale * betai
        
        output = jnp.concatenate([nui, alphai, betai], axis=-1)
        # if self.target_dim is not None:
        #     output = output[...,None,:].repeat(self.target_dim, axis=-2)

        output_key = self.key if self.output_key is None else self.output_key
        out = {
            **inputs,
            output_key: output,
            output_key
            + "_var": jnp.squeeze(betai / (nui * (alphai - 1)), axis=-1),
            output_key
            + "_aleatoric": jnp.squeeze(betai / (alphai - 1), axis=-1),
            output_key
            + "_wst2": jnp.squeeze(betai*(1+nui) / (alphai *nui), axis=-1),
        }
        return out


UNCERTAINY_MODULES = {
    "CONSTRAIN_EVIDENCE": ConstrainEvidence,
}