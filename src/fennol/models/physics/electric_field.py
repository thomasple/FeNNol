#!/usr/bin/env python3
"""Electric field model for FENNOL.

Created by C. Cattin 2024
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import ClassVar

from fennol.utils import AtomicUnits as Au


class ElectricField(nn.Module):
    """Electric field from distributed point charges with short-range damping.

    FID: ELECTRIC_FIELD

    The short-range damping is defined as in AMOEBA+

    """

    damping_param: float = 0.7
    """Damping parameter for the electric field."""
    charges_key: str = 'charges'
    """Key of the charges in the input."""
    graph_key: str = 'graph'
    """Key of the graph in the input."""
    polarizability_key: str = 'polarizability'
    """Key of the polarizability in the input."""
    trainable: bool = False

    FID: ClassVar[str] = 'ELECTRIC_FIELD'

    @nn.compact
    def __call__(self, inputs):
        species = inputs['species']

        # Graph information
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph['edge_src'], graph['edge_dst']

        # Distance and vector between each pair of atoms in atomic units
        distances = graph['distances']
        rij = distances / Au.BOHR
        vec_ij = graph['vec'] / Au.BOHR
        polarizability = (
            inputs[self.polarizability_key] / Au.BOHR**3
        )
        pol_src = polarizability[edge_src]
        pol_dst = polarizability[edge_dst]
        alpha_ij = pol_dst * pol_src
        # Effective distance
        uij = rij / alpha_ij ** (1 / 6)

        # Charges and polarizability
        charges = inputs[self.charges_key]
        rij = rij[:, None]
        q_ij = charges[edge_dst, None]

        if self.trainable:
            damping_param = jnp.abs(self.param('damping_param', lambda key: jnp.array(self.damping_param)))
        else:
            damping_param = self.damping_param
        # Damping term
        damping_field = 1 - jnp.exp(
            -damping_param * uij**1.5
        )[:, None]

        # Electric field
        eij = -q_ij * (vec_ij / rij**3) * damping_field
        electric_field = jax.ops.segment_sum(
            eij, edge_src, species.shape[0]
        ).flatten()

        output = {
            'electric_field': electric_field
        }

        return {**inputs, **output}


if __name__ == "__main__":
    pass
