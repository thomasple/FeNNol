#!/usr/bin/env python3
"""Polarisation model for FENNOL.

Created by C. Cattin 2024
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class Polarisation(nn.Module):
    """Polarisation model for FENNOL.

    Attributes
    ----------
    name : str
        Name of the polarisation model.
    energy_key : str
        Key of the energy in the input.
    graph_key : str
        Key of the graph in the input.
    polarisability_key : str
        Key of the polarisability in the input.
    coulomb_key : str
        Key of the Coulomb matrix in the input.
    charges_key : str
        Key of the charges in the input.
    electric_field_key : str
        Key of the electric field in the input.
    induce_dipole_key : str
        Key of the induced dipole in the input.
    damping_param_mutual : float
        Damping parameter for mutual polarisation.
    damping_param_field : float
        Damping parameter for the electric field.
    """

    name: str
    energy_key: str = None
    graph_key: str = 'graph'
    polarisability_key: str = 'polarisability'
    coulomb_key: str = 'coulomb'
    charges_key: str = 'charges'
    electric_field_key: str = 'electric_field'
    induce_dipole_key: str = 'induce_dipole'
    damping_param_mutual: float = 0.39
    damping_param_field: float = 0.7


    @nn.compact
    def __call__(self, inputs):
        pass


if __name__ == "__main__":
    print("This is a module, not a script")
