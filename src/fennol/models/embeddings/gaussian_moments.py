import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics, CG_SO3
from ..misc.encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import Dict, ClassVar


class GaussianMomentsEmbedding(nn.Module):
    """Gaussian moments embedding

    The construction of this embedding is similar to ACE but with a fixed lmax=3 and
    a subset of tensor product paths chosen by hand.

    ### Reference 
    adapted from J. Chem. Theory Comput. 2020, 16, 8, 5410–5421
    (https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00347)

    """

    _graphs_properties: Dict
    nchannels: int = 7
    """The number of chemical-radial (chemrad) channels for the density representation."""
    graph_key: str = "graph"
    """The key in the input dictionary that corresponds to the molecular graph."""
    embedding_key: str = "embedding"
    """The key in the output dictionary where the computed embedding will be stored."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """A dictionary of parameters for the species encoding. See `fennol.models.misc.encodings.SpeciesEncoding`"""
    radial_basis: dict = dataclasses.field(default_factory=dict)
    """A dictionary of parameters for the radial basis. See `fennol.models.misc.encodings.RadialBasis`"""

    FID: ClassVar[str] = "GAUSSIAN_MOMENTS"


    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        assert len(species.shape) == 1, "Species must be a 1D array (batches must be flattened)"

        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]

        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        radial_basis = RadialBasis(
            **{**self.radial_basis, "end": cutoff, "name": "RadialBasis"}
        )(graph["distances"])
        radial_size = radial_basis.shape[-1]

        species_encoding = SpeciesEncoding(**self.species_encoding,name="SpeciesEncoding")(species)
        afvs_size = species_encoding.shape[-1]

        chemrad_coupling = self.param(
            "chemrad_coupling",
            nn.initializers.normal(stddev=1.0 / (afvs_size * radial_size) ** 0.5),
            (afvs_size, radial_size, self.nchannels),
        )
        xij = (
            jnp.einsum(
                "ai,aj,ijk->ak",
                species_encoding[edge_dst],
                radial_basis,
                chemrad_coupling,
            )
            * graph["switch"][:, None]
        )

        Yij = generate_spherical_harmonics(lmax=3, normalize=False)(
            graph["vec"] / graph["distances"][:, None]
        )
        rhoij = xij[:, :, None] * Yij[:, None, :]

        rhoi = jax.ops.segment_sum(rhoij, edge_src, species.shape[0])

        xi0 = jax.lax.index_in_dim(rhoi, 0, axis=-1, keepdims=False)

        rhoi1 = jax.lax.dynamic_slice_in_dim(rhoi, start_index=1, slice_size=3, axis=-1)
        rhoi2 = jax.lax.dynamic_slice_in_dim(rhoi, start_index=4, slice_size=5, axis=-1)
        rhoi3 = jax.lax.dynamic_slice_in_dim(rhoi, start_index=9, slice_size=7, axis=-1)

        pairs = []
        triplets = []
        for i in range(self.nchannels):
            for j in range(i, self.nchannels):
                pairs.append([i, j])
                for k in range(j, self.nchannels):
                    triplets.append([i, j, k])

        p1, p2 = np.array(pairs).T
        p1, p2 = jnp.array(p1), jnp.array(p2)
        xi11 = jnp.sum(rhoi1[:, p1, :] * rhoi1[:, p2, :], axis=-1)
        xi22 = jnp.sum(rhoi2[:, p1, :] * rhoi2[:, p2, :], axis=-1)
        xi33 = jnp.sum(rhoi3[:, p1, :] * rhoi3[:, p2, :], axis=-1)

        t1, t2, t3 = np.array(triplets).T
        t1, t2, t3 = jnp.array(t1), jnp.array(t2), jnp.array(t3)
        rhoi2t1 = rhoi2[:, t1, :]
        rhoi1t2 = rhoi1[:, t2, :]
        rhoi1t3 = rhoi1[:, t3, :]
        w112 = jnp.array(CG_SO3(1, 1, 2))
        xi211 = jnp.einsum("...m,...n,...o,nom->...", rhoi2t1, rhoi1t2, rhoi1t3, w112)

        rhoi2t2 = rhoi2[:, t2, :]
        rhoi2t3 = rhoi2[:, t3, :]
        w222 = jnp.array(CG_SO3(2, 2, 2))
        xi222 = jnp.einsum("...m,...n,...o,nom->...", rhoi2t1, rhoi2t2, rhoi2t3, w222)

        rhoi3t1 = rhoi3[:, t1, :]
        w123 = jnp.array(CG_SO3(1, 2, 3))
        xi312 = jnp.einsum("...m,...n,...o,nom->...", rhoi3t1, rhoi1t2, rhoi2t3, w123)

        rhoi3t3 = rhoi3[:, t3, :]
        w233 = jnp.array(CG_SO3(2, 3, 3))
        xi323 = jnp.einsum("...m,...n,...o,nom->...", rhoi3t1, rhoi2t2, rhoi3t3, w233)

        embedding = jnp.concatenate(
            [species_encoding, xi0, xi11, xi22, xi33, xi211, xi222, xi312, xi323],
            axis=-1,
        )

        if self.embedding_key is None:
            return embedding
        return {**inputs, self.embedding_key: embedding}
