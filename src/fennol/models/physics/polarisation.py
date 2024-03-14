#!/usr/bin/env python3
"""Polarisation model for FENNOL.

Created by C. Cattin 2024
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from ...utils import AtomicUnits as au


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
        species = species.reshape(-1)
        graph = inputs[self.graph_key]
        # Graph
        edge_src, edge_dst = graph['edge_src'], graph['edge_dst']
        # Distances and vector between each pair of atoms
        distances = graph['distances']
        rij = distances / au.BOHR
        vec_ij = graph['vec'] / au.BOHR
        # Polarisability
        polarisability = (
            inputs[self.polarisability_key].reshape(-1) / au.BOHR**3
        )
        pol_src = polarisability[edge_src]
        pol_dst = polarisability[edge_dst]
        alpha_ij = pol_dst * pol_src

        # The output is a dictionary with the polarisation energy
        output = {}

        # Effective distance
        uij = self.get_uij(rij, alpha_ij)

        # Values of the system
        n_batch, n_atoms = species.shape
        polarisability = polarisability.reshape(n_batch, n_atoms)

        # Interaction matrix
        t_matrix = self.get_T_matrix(
            vec=vec_ij,
            rij=rij,
            edge_src=edge_src,
            edge_dst=edge_dst,
            species=species,
            polarisability=polarisability,
            uij=uij,
            a=self.damping_param_mutual
        )

        # Permanent electric field

        # Solve linear equation to get electric point dipole moment

        # Energy related to the polarisation

    def get_uij(
            self,
            rij: jnp.ndarray,
            alpha_ij: jnp.ndarray
        ) -> jnp.ndarray:
        """Compute the effective distance for every pair.

        Parameters
        ----------
        rij : jnp.ndarray
            Tensor containing the distances between every pair.
        alpha_ij : jnp.ndarray
            Product of the polarization between every pair

        Returns
        -------
        uij : jnp.ndarray
            Tensor containing the effective distances for every pair
        """
        return rij / alpha_ij ** (1 / 6)

    def get_T_matrix(
        self,
        vec: jnp.ndarray,
        rij: jnp.ndarray,
        edge_src: jnp.ndarray,
        edge_dst: jnp.ndarray,
        species: jnp.ndarray,
        polarisability: jnp.ndarray,
        uij: jnp.ndarray,
        a: float
    ) -> jnp.ndarray:
        n_batch, n_atoms = species.shape
        lambda_3, lambda_5 = self.damping(uij, a)

    def damping(
        self,
        uij: jnp.ndarray,
        a: float
    ) -> jnp.ndarray:
        """Compute the damping function.

        Parameters
        ----------
        uij : jnp.ndarray
            Tensor containing the effective distances for every pair.
        a : float
            Hyper parameter of the polarisation model.

        Returns
        -------
        lambda_3 : jnp.ndarray
            Damping function for the power 3 term.
        lambda_5 : jnp.ndarray
            Damping function for the power 5 term.
        """
        exp = jnp.exp(-a * uij**3)
        lambda_3 = 1 - exp
        lambda_5 = 1 - (1 + a * uij**3) * exp
        return lambda_3, lambda_5



if __name__ == "__main__":
    pola = Polarisation(name='polarisation')

    species = jnp.array([6, 1, 1])
    graph = {
        'edge_src': jnp.array([0, 0, 1]),
        'edge_dst': jnp.array([1, 2, 2]),
        'distances': jnp.array([1.0, 1.0, jnp.sqrt(2.0)]),
        'vec': jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }
    charges = jnp.array([-0.5, 0.25, 0.25])
    polarisability = jnp.array([1.0, 0.5, 0.5])
    inputs = {
        'species': species,
        'graph': graph,
        'charges': charges,
        'polarisability': polarisability
    }

    key = random.PRNGKey(0)

    variables = pola.init(key, inputs)

    results = pola.apply(variables, inputs)
