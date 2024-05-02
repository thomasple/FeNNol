import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict, ClassVar
import numpy as np
import dataclasses

from ...utils.periodic_table import PERIODIC_TABLE, VALENCE_STRUCTURE
from ..misc.encodings import SpeciesEncoding, RadialBasis


class EEACSF(nn.Module):
    """element-embracing Atom-Centered Symmetry Functions 

    FID : EEACSF

    This is an embedding similar to ANI that include simple chemical information in the AEVs
    without trainable parameters (fixed embedding).
    The angle embedding is computed using a low-order Fourier expansion.

    ### Reference
    Loosely inspired from M. Eckhoff and M. Reiher, Lifelong Machine Learning Potentials,
    J. Chem. Theory Comput. 2023, 19, 12, 3509–3525, https://doi.org/10.1021/acs.jctc.3c00279

    """

    _graphs_properties: Dict
    graph_angle_key: str
    """ The key in the input dictionary that corresponds to the angular graph."""
    nmax_angle: int = 4
    """ The maximum fourier order for the angle representation."""
    embedding_key: str = "embedding"
    """ The key to use for the output embedding in the returned dictionary."""
    graph_key: str = "graph"
    """ The key in the input dictionary that corresponds to the radial graph."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """ The species encoding parameters. See `fennol.models.misc.encodings.SpeciesEncoding`"""
    radial_basis: dict = dataclasses.field(default_factory=dict)
    """ The radial basis parameters the radial AEV. See `fennol.models.misc.encodings.RadialBasis`"""
    radial_basis_angle: dict = dataclasses.field(default_factory=dict)
    """ The radial basis parameters for the angular AEV. See `fennol.models.misc.encodings.RadialBasis`"""
    angle_combine_pairs: bool = False
    """ If True, the angular AEV is computed by combining pairs of radial AEV."""

    FID: ClassVar[str] = "EEACSF"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]

        # species encoding
        
        onehot = SpeciesEncoding(**self.species_encoding, name="SpeciesEncoding")(
            species
        )

        # Radial graph
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        switch = graph["switch"][:,None]
        edge_src = graph["edge_src"]
        edge_dst = graph["edge_dst"]

        # Radial BASIS
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        radial_terms = (
            RadialBasis(
                **{
                    **self.radial_basis,
                    "end": cutoff,
                    "name": f"RadialBasis",
                }
            )(distances)*switch
        )
        # aggregate radial AEV
        radial_aev = jax.ops.segment_sum(
            radial_terms[:, :, None] * onehot[edge_dst, None, :],
            edge_src,
            species.shape[0],
        ).reshape(species.shape[0], -1)

        # Angular graph
        graph_angle = inputs[self.graph_angle_key]     
        angles = graph_angle["angles"]
        dang = graph_angle["distances"]
        central_atom = graph_angle["central_atom"]
        angle_src, angle_dst = graph_angle["angle_src"], graph_angle["angle_dst"]
        switch_angles = graph_angle["switch"][:, None]
        angular_cutoff = self._graphs_properties[self.graph_angle_key]["cutoff"]
        edge_dst_ang = graph_angle["edge_dst"]


        radial_basis_angle = (
            self.radial_basis_angle
            if self.radial_basis_angle is not None
            else self.radial_basis
        )

        # Angular AEV parameters
        if self.angle_combine_pairs:
            factor2 = RadialBasis(
                **{
                    **radial_basis_angle,
                    "end": angular_cutoff,
                    "name": f"RadialBasisAng",
                }
            )(dang)*switch_angles
            radial_ang = (factor2[:, :, None] * onehot[edge_dst_ang, None, :]).reshape(-1, onehot.shape[1]*factor2.shape[1])

            radial_aev_ang = radial_ang[angle_src]*radial_ang[angle_dst]

            # Angular AEV
            nangles = jnp.asarray(
                np.arange(self.nmax_angle + 1)[None, :], dtype=angles.dtype
            )
            factor1 = jnp.cos(nangles * angles[:, None])

            angular_aev = jax.ops.segment_sum(
                factor1[:, None, :] * radial_aev_ang[:, :, None],
                central_atom,
                species.shape[0],
            ).reshape(species.shape[0], -1)

        else:
            valence = SpeciesEncoding(
                encoding="sjs_coordinates", name="SJSEncoding", trainable=False
            )(species)
            d12 = 0.5 * (dang[angle_src] + dang[angle_dst])
            switch12 = switch_angles[angle_src] * switch_angles[angle_dst]

            
            factor2 = RadialBasis(
                **{
                    **radial_basis_angle,
                    "end": angular_cutoff,
                    "name": f"RadialBasisAng",
                }
            )(d12)

            # Angular AEV
            nangles = jnp.asarray(
                np.arange(self.nmax_angle + 1)[None, :], dtype=angles.dtype
            )
            factor1 = jnp.cos(nangles * angles[:, None]) * switch12

            angular_terms = (factor1[:, None, :] * factor2[:, :, None]).reshape(
                -1, factor1.shape[1] * factor2.shape[1]
            )

            valence_dst = valence[edge_dst_ang]
            vangsrc = valence_dst[angle_src]
            vangdst = valence_dst[angle_dst]
            valence_ang_p = vangsrc + vangdst
            valence_ang_m = vangsrc * vangdst
            valence_ang = (valence_ang_p[:, :, None] * valence_ang_m[:, None, :]).reshape(
                -1, valence_ang_p.shape[1] * valence_ang_m.shape[1]
            )

            angular_aev = jax.ops.segment_sum(
                angular_terms[:, :, None] * valence_ang[:, None, :],
                central_atom,
                species.shape[0],
            ).reshape(species.shape[0], -1)

        
        embedding = jnp.concatenate((onehot, radial_aev, angular_aev), axis=-1)
        if self.embedding_key is None:
            return embedding
        return {**inputs, self.embedding_key: embedding}
