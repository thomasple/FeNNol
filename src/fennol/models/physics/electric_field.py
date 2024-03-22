#!/usr/bin/env python3
"""Electric field model for FENNOL.

Created by C. Cattin 2024
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from fennol.utils import AtomicUnits as Au

class ElectricField(nn.Module):
    
    name: str = 'electric_field'
    damping_param: float = 0.7
    charges_key: str = 'charges'
    graph_key: str = 'graph'
    polarisability_key: str = 'polarisability'

    @nn.compact
    def __call__(self, inputs):
        species = inputs['species']
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph['edge_src'], graph['edge_dst']
        distances = graph['distances']
        rij = distances / Au.BOHR
        vec_ij = graph['vec'] / Au.BOHR
        charges = inputs[self.charges_key]
        rij = rij[:, None]
        q_ij = charges[edge_dst, None]
        polarisability = (
            inputs[self.polarisability_key] / Au.BOHR**3
        )
        pol_src = polarisability[edge_src]
        pol_dst = polarisability[edge_dst]
        alpha_ij = pol_dst * pol_src
        uij = rij / alpha_ij ** (1 / 6)
        damping_field = 1 - jnp.exp(
            -self.damping_param * uij**1.5
        )[:, None]
        eij = -q_ij * (vec_ij / rij**3) * damping_field
        electric_field = jax.ops.segment_sum(
            eij, edge_src, species.shape[0]
        ).flatten()
        ouput = {
            self.name: electric_field
        }
        return {**inputs, **ouput}


if __name__ == "__main__":
    pass
