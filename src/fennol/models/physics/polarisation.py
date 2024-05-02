#!/usr/bin/env python3
"""Polarisation model for FENNOL.

Created by C. Cattin 2024
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional, ClassVar

from fennol.utils import AtomicUnits as Au


class Polarisation(nn.Module):
    """Polarisation model with Thole damping scheme."""

    energy_key: Optional[str] = None
    """Key of the energy in the outputs."""
    graph_key: str = 'graph'
    """Key of the graph in the inputs."""
    polarisability_key: str = 'polarisability'
    """Key of the polarisability in the inputs."""
    electric_field_key: str = 'electric_field'
    """Key of the electric field in the inputs."""
    induce_dipole_key: str = 'induce_dipole'
    """Key of the induced dipole in the outputs."""
    damping_param_mutual: float = 0.39
    """Damping parameter for mutual polarisation."""

    FID: ClassVar[str] = 'POLARISATION'

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
        # Distances and vector between each pair of atoms in atomic units
        distances = graph['distances']
        rij = distances / Au.BOHR
        vec_ij = graph['vec'] / Au.BOHR
        # Polarisability
        polarisability = (
            inputs[self.polarisability_key] / Au.BOHR**3
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

        ######################
        # Interaction matrix #
        ######################
        # Diagonal terms
        tii = 1 / polarisability[:, None]
        # Off-diagonal terms
        tij = (
            3 * lambda_5[:, None, None]
                * vec_ij[:, :, None] * vec_ij[:, None, :]
                / rij[:, None, None]**5
            - jnp.eye(3)[None, :, :] * lambda_3[:, None, None]
                / rij[:, None, None]**3
        )

        def matvec(mui):
            """Compute the matrix vector product of T and mu."""
            mui = mui.reshape(-1, 3)
            tmu_self = tii * mui
            tmu_pair = jnp.einsum("jab,jb->ja", tij, mui[edge_dst])
            tmu = (
                jax.ops.segment_sum(tmu_pair, edge_src, species.shape[0])
                + tmu_self
            )
            return tmu.flatten()

        ##################
        # Electric field #
        ##################
        electric_field = inputs[self.electric_field_key]

        ###############################
        # Electric point dipole moment#
        ###############################
        mu = jax.scipy.sparse.linalg.cg(matvec, electric_field)[0]
        mu_ = jax.lax.stop_gradient(mu)

        # Matrix vector product
        tmu = matvec(mu_)
        # Polarisation energy
        pol_energy = (
            (0.5 * tmu - electric_field) * mu_
        ).reshape(-1, 3).sum(axis=1)

        # Output
        output[self.electric_field_key] = electric_field.reshape(-1, 3)
        output[self.induce_dipole_key] = mu.reshape(-1, 3) * Au.BOHR
        energy_key = (
            self.energy_key if self.energy_key is not None else 'polarisation'
        )
        output[energy_key] = pol_energy
        output['tmu'] = tmu.reshape(-1, 3)

        return {**inputs, **output}


if __name__ == "__main__":
    pass
