#!/usr/bin/env python3
"""Polarisation model for FENNOL.

Created by C. Cattin 2024
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from fennol.utils import AtomicUnits as au


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

    name: str = 'polarisation'
    energy_key: str = None
    graph_key: str = 'graph'
    polarisability_key: str = 'polarisability'
    coulomb_key: str = 'coulomb'
    charges_key: str = 'charges'
    electric_field_key: str = 'electric_field'
    induce_dipole_key: str = 'induce_dipole'
    damping_param_mutual: float = 0.39
    damping_param_field: float = 0.7

    if energy_key is None:
        energy_key = name

    FID: str = 'POLARISATION'

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the polarisation model.

        Parameters
        ----------
        inputs : dict
            Input dictionary containing all the info about the system.
            This dictionary is given from the FENNIX class.
        """
        # Species
        species = inputs['species']
        graph = inputs[self.graph_key]
        # Graph
        edge_src, edge_dst = graph['edge_src'], graph['edge_dst']
        # Distances and vector between each pair of atoms
        distances = graph['distances']
        rij = distances / au.BOHR
        vec_ij = graph['vec'] / au.BOHR
        # Polarisability
        polarisability = (
            inputs[self.polarisability_key] / au.BOHR**3
        )

        pol_src = polarisability[edge_src]
        pol_dst = polarisability[edge_dst]
        alpha_ij = pol_dst * pol_src

        # The output is a dictionary with the polarisation energy
        output = {}

        # Effective distance
        uij = rij / alpha_ij ** (1 / 6)
        # Damping terms
        exp = jnp.exp(-self.damping_param_mutual * uij**3)
        lambda_3 = 1 - exp
        lambda_5 = 1 - (1 + self.damping_param_mutual * uij**3) * exp

        def matvec(x):
            pass

        return graph

        # Values of the system
        # n_batch, n_atoms = species.shape
        # polarisability = polarisability.reshape(n_batch, n_atoms)

        # Interaction matrix
        # t_matrix = self.get_T_matrix(
        #     vec=vec_ij,
        #     rij=rij,
        #     edge_src=edge_src,
        #     edge_dst=edge_dst,
        #     species=species,
        #     polarisability=polarisability,
        #     uij=uij,
        #     a=self.damping_param_mutual
        # )

        # Permanent electric field

        # Solve linear equation to get electric point dipole moment

        # Energy related to the polarisation


if __name__ == "__main__":

    from jax.random import PRNGKey

    from fennol import FENNIX

    model = FENNIX(
        cutoff=5.0,
        rng_key=PRNGKey(0),
        modules={
            # 'embedding': {
            #     'module_name': 'ALLEGRO',
            #     'dim': 64,
            #     'nchannels': 1,
            #     'lmax': 1,
            #     'nlayers': 1,
            #     'twobody_hidden': [32, 64],
            #     'latent_hidden': [64],
            #     'radial_basis': {
            #         'dim': 8,
            #         'basis': 'bessel',
            #         'trainable': False,
            #     },
            #     'species_encoding': {
            #         'encoding': 'onehot',
            #         'species_order': ['O', 'H'],
            #         'trainable': False
            #     },
            # },
            'energy': {
                'module_name': 'NEURAL_NET',
                'neurons': [32, 1],
                'input_key': 'coordinates',
            },
            'charges': {
                'module_name': 'CHEMICAL_CONSTANT',
                'value': {
                    'O': -0.504458,
                    'H': 0.252229
                },
            },
            'polarisability': {
            'module_name': 'CHEMICAL_CONSTANT',
                'value': {
                    'O': 0.976,
                    'H': 0.428
                },
            },
            'coulomb': {
                'module_name': 'COULOMB',
                'charges_key': 'charges',
            },
            'polarisation': {
                'module_name': 'POLARISATION',
            },
        }
    )

    species = jnp.array(
        [
            [8, 1, 1],
            [8, 1, 1]
        ]
    ).reshape(-1)

    coordinates = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ]
        ]
    ).reshape(-1, 3)

    natoms = jnp.array([3, 3])
    batch_index = jnp.array([0, 0, 0, 1, 1, 1])


    output = model(
        species=species,
        coordinates=coordinates,
        natoms=natoms,
        batch_index=batch_index
    )
