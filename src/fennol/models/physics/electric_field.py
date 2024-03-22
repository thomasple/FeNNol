#!/usr/bin/env python3
"""Electric field model for FENNOL.

Created by C. Cattin 2024
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from fennol.utils import AtomicUnits as Au


class ElectricField(nn.Module):
    """Electric field model for FENNOL.

    Attributes
    ----------
    name : str
        Name of the electric field model.
    damping_param : float
        Damping parameter for the electric field.
    charges_key : str
        Key of the charges in the input.
    graph_key : str
        Key of the graph in the input.
    polarisability_key : str
        Key of the polarisability in the input.
    """

    name: str = 'electric_field'
    damping_param: float = 0.7
    charges_key: str = 'charges'
    graph_key: str = 'graph'
    polarisability_key: str = 'polarisability'

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the electric field model.

        Parameters
        ----------
        inputs : dict
            Input dictionary containing all the info about the system.
            This dictionary is given from the FENNIX class.
        """
        species = inputs['species']

        # Graph information
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph['edge_src'], graph['edge_dst']

        # Distance and vector between each pair of atoms in atomic units
        distances = graph['distances']
        rij = distances / Au.BOHR
        vec_ij = graph['vec'] / Au.BOHR
        rij = rij[:, None]
        polarisability = (
            inputs[self.polarisability_key] / Au.BOHR**3
        )

        # For tests purposes
        testing = 'training_flag' not in inputs
        if testing:
            rij *= Au.BOHR
            vec_ij *= Au.BOHR
            polarisability *= Au.BOHR**3

        # Charges and polarisability
        charges = inputs[self.charges_key]
        q_ij = charges[edge_dst, None]
        pol_src = polarisability[edge_src]
        pol_dst = polarisability[edge_dst]
        alpha_ij = pol_dst * pol_src

        # Effective distance and damping term
        uij = rij / alpha_ij ** (1 / 6)
        damping_field = 1 - jnp.exp(
            -self.damping_param * uij**1.5
        )[:, None]

        # Electric field
        eij = -q_ij * (vec_ij / rij**3) * damping_field
        electric_field = jax.ops.segment_sum(
            eij, edge_src, species.shape[0]
        ).flatten()

        output = {
            self.name: electric_field
        }

        return {**inputs, **output}


if __name__ == "__main__":
    pass
