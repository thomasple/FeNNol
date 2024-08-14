import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Any, Dict, Union, Callable, Sequence, Optional, ClassVar
from ...utils import AtomicUnits as au
from ...utils.periodic_table import (
    D3_ELECTRONEGATIVITIES,
    D3_HARDNESSES,
    D3_VDW_RADII,
    D3_COV_RADII,
    D3_KAPPA,
    VDW_RADII,
    POLARIZABILITIES,
    C6_FREE,
    VALENCE_ELECTRONS,
)


class VdwOQDO(nn.Module):
    """ Dispersion and exchange based on the Optimized Quantum Drude Oscillator model.
    
    FID : VDW_OQDO

    ### Reference
    A. Khabibrakhmanov, D. V. Fedorov, and A. Tkatchenko, Universal Pairwise Interatomic van der Waals Potentials Based on Quantum Drude Oscillators,
    J. Chem. Theory Comput. 2023, 19, 21, 7895–7907 (https://doi.org/10.1021/acs.jctc.3c00797)
    
    """
    graph_key: str = "graph"
    """ The key for the graph input."""
    include_exchange: bool = True
    """ Whether to compute the exchange part."""
    ratiovol_key: Optional[str] = None
    """ The key for the ratio between AIM volume and free-atom volume. 
         If None, the volume ratio is assumed to be 1.0."""
    energy_key: Optional[str] = None
    """ The key for the output energy. If None, the name of the module is used."""
    damped: bool = True
    """ Whether to use short-range damping."""
    _energy_unit: str = "Ha"
    """The energy unit of the model. **Automatically set by FENNIX**"""

    FID: ClassVar[str]  = "VDW_OQDO"

    @nn.compact
    def __call__(self, inputs):
        energy_unit = au.get_multiplier(self._energy_unit)

        species = inputs["species"]
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        rij = graph["distances"] / au.BOHR
        switch = graph["switch"]

        c6 = jnp.asarray(C6_FREE)[species]
        alpha = jnp.asarray(POLARIZABILITIES)[species]

        if self.ratiovol_key is not None:
            ratiovol = inputs[self.ratiovol_key] + 1.0e-6
            if ratiovol.shape[-1] == 1:
                ratiovol = jnp.squeeze(ratiovol, axis=-1)
            c6 = c6 * ratiovol**2
            alpha = alpha * ratiovol

        c6i, c6j = c6[edge_src], c6[edge_dst]
        alphai, alphaj = alpha[edge_src], alpha[edge_dst]

        # combination rules
        alphaij = 0.5 * (alphai + alphaj)
        c6ij = 2 * alphai * alphaj * c6i * c6j / (c6i * alphaj**2 + c6j * alphai**2)

        # equilibrium distance
        Re = (alphaij * (128.0 / au.FSC ** (4.0 / 3.0))) ** (1.0 / 7.0)
        Re2 = Re**2
        Re4 = Re**4
        # fit to largest root of eq (S33) of "Universal Pairwise Interatomic van der Waals Potentials Based On Quantum Drude Oscillators"
        if self.damped:
            muw = (
                4.83053463e-01
                - 3.76191669e-02 * Re
                + 1.27066988e-03 * Re2
                - 7.21940151e-07 * Re4
            ) / (3.84212120e-02 - 3.16915319e-02 * Re + 2.37410890e-02 * Re2)
        else:
            muw = (
                3.66316787e01
                - 5.79579187 * Re
                + 3.02674813e-01 * Re2
                - 3.65461255e-04 * Re4
            ) / (-1.46169102e01 + 7.32461225 * Re)

        c8ij = 5 * c6ij / muw
        c10ij = 245 * c6ij / (8 * muw**2)

        if self.damped:
            z = 0.5 * muw * rij**2
            ez = jnp.exp(-z)
            f6 = 1.0 - ez * (1.0 + z + 0.5 * z**2 + (1.0 / 6.0) * z**3)
            f8 = f6 - (1.0 / 24.0) * ez * z**4
            f10 = f8 - (1.0 / 120.0) * ez * z**5
            epair = (
                f6 * c6ij / rij**6 + f8 * c8ij / rij**8 + f10 * c10ij / rij**10
            )
        else:
            epair = c6ij / rij**6 + c8ij / rij**8 + c10ij / rij**10

        edisp = (-0.5*energy_unit) * jax.ops.segment_sum(epair * switch, edge_src, species.shape[0])

        output_key = self.name if self.energy_key is None else self.energy_key

        if not self.include_exchange:
            return {**inputs, output_key: edisp}

        ### exchange
        w = 4 * c6ij / (3 * alphaij**2)
        # q = (alphaij * mu*w**2)**0.5
        q2 = alphaij * muw * w
        # undamped case
        if self.damped:
            ze = 0.5 * muw * Re2
            eze = jnp.exp(-ze)

            s6 = eze * (1.0 + ze + 0.5 * ze**2 + (1.0 / 6.0) * ze**3)
            f6e = 1.0 - s6
            muwRe = muw * Re
            df6e = muwRe * s6 - eze * (
                muwRe + 0.5 * Re * muwRe**2 + (1.0 / 8.0) * Re2 * muwRe**3
            )

            s8 = (1.0 / 24.0) * eze * ze**4
            f8e = f6e - s8
            df8e = df6e + muwRe * s8 - (1.0 / 48.0) * eze * Re2 * Re * muwRe**4

            s10 = (1.0 / 120.0) * eze * ze**5
            f10e = f8e - s10
            df10e = df8e + muwRe * s10 - (1.0 / 384.0) * eze * Re2 * Re2 * muwRe**5

            den = 2 * c6ij * Re2 * (6 * f6e - Re * df6e)
            A = (
                0.5
                + c8ij * (8 * f8e - Re * df8e) / den
                + c10ij * (10 * f10e - Re * df10e) / (den * Re2)
            )
        else:
            A = 0.5 + 2 * c8ij / (3 * c6ij * Re2) + 5 * c10ij / (6 * c6ij * Re4)
            ez = jnp.exp(-0.5 * muw * rij**2)

        exij = A * q2 * ez / rij
        ex = (0.5*energy_unit) * jax.ops.segment_sum(exij * switch, edge_src, species.shape[0])

        return {
            **inputs,
            output_key + "_dispersion": edisp,
            output_key + "_exchange": ex,
            output_key: edisp + ex,
        }
