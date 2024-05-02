import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics
from ..misc.encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import  Dict, Optional, ClassVar


class FOAMEmbedding(nn.Module):
    """Filtered Overlap of Atomic Moments
    
    FID : FOAM

    Similar to SOAP embedding but for each rank l, we do not take all combinations
    of each channels but linearly project on 2 nchannels elements and then take the
    scalar product. This is then kind of a linearly filtered SOAP embedding.

    """
    _graphs_properties: Dict
    lmax: int = 2
    """The maximum order of spherical tensors."""
    nchannels: Optional[int] = None
    """The number of channels."""
    graph_key: str = "graph"
    """The key for the graph input."""
    embedding_key: str = "embedding"
    """The key for the embedding output."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """The species encoding parameters. See `fennol.models.misc.encodings.SpeciesEncoding`"""
    radial_basis: dict = dataclasses.field(default_factory=dict)
    """The radial basis parameters. See `fennol.models.misc.encodings.RadialBasis`"""
    include_species: bool = True
    """Whether to include the species encoding in the embedding."""

    FID: ClassVar[str] = "FOAM"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        assert (
            len(species.shape) == 1
        ), "Species must be a 1D array (batches must be flattened)"

        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        switch = graph["switch"][:, None]
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        radial_basis = (
            RadialBasis(**{**self.radial_basis, "end": cutoff, "name": "RadialBasis"})(
                graph["distances"]
            )
            * switch
        )

        species_encoding = SpeciesEncoding(
            **self.species_encoding, name="SpeciesEncoding"
        )(species)

        Dij = (radial_basis[:, :, None] * species_encoding[edge_dst, None, :]).reshape(
            -1, species_encoding.shape[-1] * radial_basis.shape[-1]
        )

        Yij = generate_spherical_harmonics(lmax=self.lmax, normalize=False)(
            graph["vec"] / graph["distances"][:, None]
        )[:, :, None]

        rhoi = jax.ops.segment_sum(Dij[:, None, :] * Yij, edge_src, species.shape[0])

        nbasis = rhoi.shape[-1]
        nchannels = self.nchannels if self.nchannels is not None else nbasis

        if self.include_species:
            xis = [species_encoding, rhoi[:, 0, :]]
        else:
            xis = [rhoi[:, 0, :]]

        for l in range(self.lmax + 1):
            rhoil = rhoi[:, l**2 : (l + 1) ** 2, :]
            xl, yl = jnp.split(
                nn.Dense(2 * nchannels, use_bias=False, name=f"xy_l{l}")(rhoil),
                2,
                axis=-1,
            )
            xil = (xl*yl).sum(axis=1) / (2 * l + 1) ** 0.5
            xis.append(xil)
        xi = jnp.concatenate(xis, axis=-1)

        if self.embedding_key is None:
            return xi
        return {**inputs, self.embedding_key: xi}
