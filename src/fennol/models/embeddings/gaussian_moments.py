import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics, CG_SO3
from ...utils import SpeciesEncoding, RadialEmbedding
import dataclasses


class GaussianMomentsEmbedding(nn.Module):
    """
    Gaussian moments embedding adapted from J. Chem. Theory Comput. 2020, 16, 8, 5410–5421
    (https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00347)

    The construction of this embedding is similar to ACE but with a fixed lmax=3 and
    a subset of tensor product paths chosen by hand.

    Args:
        nchannels (int): The number of chemical-radial (chemrad) channels for the density representation.
        cutoff (float): The distance cutoff for the radial embedding.
        graph_key (str): The key in the input dictionary that corresponds to the molecular graph.
        embedding_key (str): The key in the output dictionary where the computed embedding will be stored.
        species_encoding (dict): A dictionary of parameters for species encoding.
        radial_embedding (dict): A dictionary of parameters for radial embedding.

    Call Args:
        data (dict): A dictionary containing the following keys:
            species (jax.numpy.ndarray): An array of atomic species.
            graph (dict): dictionary containing the molecular graph information. It must contain the following keys:
                edge_src (jax.numpy.ndarray): An array of source atoms for each edge.
                edge_dst (jax.numpy.ndarray): An array of destination atoms for each edge.
                distances (jax.numpy.ndarray): An array of distances for each edge.
    """

    nchannels: int = 7
    cutoff: float = 5.2
    graph_key: str = "graph"
    embedding_key: str = "embedding"
    species_encoding: dict = dataclasses.field(default_factory=dict)
    radial_embedding: dict = dataclasses.field(default_factory=dict)

    def setup(self):
        self.radial_terms = RadialEmbedding(
            **{**self.radial_embedding, "end": self.cutoff}
        )
        self.radial_size = self.radial_terms.dim
        self.species_encoder = SpeciesEncoding(**self.species_encoding)
        self.afvs_size = self.species_encoder.dim
        self.spherical_harmonics = generate_spherical_harmonics(lmax=3, normalize=True)
        self.chemrad_coupling = self.param(
            "chemrad_coupling",
            nn.initializers.normal(
                stddev=1.0 / (self.afvs_size * self.radial_size) ** 0.5
            ),
            (self.afvs_size, self.radial_size, self.nchannels),
        )

        pairs = []
        triplets = []
        for i in range(self.nchannels):
            for j in range(i, self.nchannels):
                pairs.append([i, j])
                for k in range(j, self.nchannels):
                    triplets.append([i, j, k])
        npairs = len(pairs)
        ntriplets = len(triplets)

        self.dim = self.afvs_size + self.nchannels + 3 * npairs + 4 * ntriplets

        self.pairs = jnp.array(pairs, dtype=jnp.int32).T
        self.triplets = jnp.array(triplets, dtype=jnp.int32).T
        self.w112 = jnp.array(CG_SO3(1, 1, 2))
        self.w222 = jnp.array(CG_SO3(2, 2, 2))
        self.w123 = jnp.array(CG_SO3(1, 2, 3))
        self.w233 = jnp.array(CG_SO3(2, 3, 3))

    def __call__(self, data):
        species = data["species"]
        # species_mask = (species >= 0)

        graph = data[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        # edge_mask = graph["edge_mask"]

        radial_terms = self.radial_terms(graph["distances"])
        onehot = self.species_encoder(species)

        xij = (
            jnp.einsum(
                "ai,aj,ijk->ak", onehot[edge_dst], radial_terms, self.chemrad_coupling
            )
            * graph["switch"][:, None]
        )

        Yij = self.spherical_harmonics(graph["vec"])
        rhoij = xij[:, :, None] * Yij[:, None, :]

        rhoi = jnp.zeros((species.shape[0], *rhoij.shape[1:]))
        rhoi = rhoi.at[edge_src].add(rhoij)

        rhoi1 = rhoi[:, :, 1:4]
        rhoi2 = rhoi[:, :, 4:9]
        rhoi3 = rhoi[:, :, 9:]

        xi0 = rhoi[:, :, 0]
        ps, pd = self.pairs
        t1, t2, t3 = self.triplets
        xi11 = jnp.sum(rhoi1[:, ps, :] * rhoi1[:, pd, :], axis=-1)
        xi22 = jnp.sum(rhoi2[:, ps, :] * rhoi2[:, pd, :], axis=-1)
        xi33 = jnp.sum(rhoi3[:, ps, :] * rhoi3[:, pd, :], axis=-1)
        xi211 = jnp.einsum(
            "...m,...n,...o,nom->...",
            rhoi2[:, t1, :],
            rhoi1[:, t2, :],
            rhoi1[:, t3, :],
            self.w112,
        )
        xi222 = jnp.einsum(
            "...m,...n,...o,nom->...",
            rhoi2[:, t1, :],
            rhoi2[:, t2, :],
            rhoi2[:, t3, :],
            self.w222,
        )
        xi312 = jnp.einsum(
            "...m,...n,...o,nom->...",
            rhoi3[:, t1, :],
            rhoi1[:, t2, :],
            rhoi2[:, t3, :],
            self.w123,
        )
        xi323 = jnp.einsum(
            "...m,...n,...o,nom->...",
            rhoi3[:, t1, :],
            rhoi2[:, t2, :],
            rhoi3[:, t3, :],
            self.w233,
        )

        xi = jnp.concatenate(
            [onehot, xi0, xi11, xi22, xi33, xi211, xi222, xi312, xi323], axis=-1
        )

        return {**data, self.embedding_key: xi}
