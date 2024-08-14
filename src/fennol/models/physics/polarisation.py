#!/usr/bin/env python3
"""Polarization model for FENNOL.

Created by C. Cattin 2024
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional, ClassVar

from fennol.utils import AtomicUnits as Au


class Polarization(nn.Module):
    """Polarization model with Thole damping scheme.
    
    FID: POLARIZATION
    """

    energy_key: Optional[str] = None
    """Key of the energy in the outputs."""
    graph_key: str = 'graph'
    """Key of the graph in the inputs."""
    polarizability_key: str = 'polarizability'
    """Key of the polarizability in the inputs."""
    electric_field_key: str = 'electric_field'
    """Key of the electric field in the inputs."""
    induced_dipoles_key: str = 'induced_dipoles'
    """Key of the induced dipole in the outputs."""
    damping_param_mutual: float = 0.39
    """Damping parameter for mutual polarization."""
    neglect_mutual: bool = False
    """Neglect the mutual polarization term (like in iAMOEBA)."""
    _energy_unit: str = 'Ha'
    """The energy unit of the model. **Automatically set by FENNIX**"""

    FID: ClassVar[str] = 'POLARIZATION'

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the polarization model.

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
        # Polarizability
        polarizability = (
            inputs[self.polarizability_key] / Au.BOHR**3
        )

        pol_src = polarizability[edge_src]
        pol_dst = polarizability[edge_dst]
        alpha_ij = pol_dst * pol_src

        # The output is a dictionary with the polarization energy
        output = {}

        ######################
        # Interaction matrix #
        ######################
        # Diagonal terms
        tii = 1 / polarizability[:, None]
        

        ##################
        # Electric field #
        ##################
        electric_field = inputs[self.electric_field_key]

        ###############################
        # Electric point dipole moment#
        ###############################
        if self.neglect_mutual:
            electric_field = electric_field.reshape(-1, 3)
            mu = polarizability[:,None]*electric_field
            mu_ = jax.lax.stop_gradient(mu)
            tmu = tii*mu_
            
        else:
            # Effective distance
            uij = rij / alpha_ij ** (1 / 6)
            # Damping terms
            exp = jnp.exp(-self.damping_param_mutual * uij**3)
            lambda_3 = 1 - exp
            lambda_5 = 1 - (1 + self.damping_param_mutual * uij**3) * exp
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
            mu = jax.scipy.sparse.linalg.cg(matvec, electric_field)[0]
            mu_ = jax.lax.stop_gradient(mu)

            # Matrix vector product
            tmu = matvec(mu_)

        # Polarization energy
        pol_energy = (
            (0.5 * tmu - electric_field) * mu_
        ).reshape(-1, 3).sum(axis=1)

        # Output
        output[self.electric_field_key] = electric_field.reshape(-1, 3)
        output[self.induced_dipoles_key] = mu.reshape(-1, 3) * Au.BOHR
        energy_key = (
            self.energy_key if self.energy_key is not None else self.name
        )
        energy_unit = Au.get_multiplier(self._energy_unit)
        output[energy_key] = pol_energy*energy_unit
        output['tmu'] = tmu.reshape(-1, 3)

        return {**inputs, **output}


if __name__ == "__main__":
    pass
