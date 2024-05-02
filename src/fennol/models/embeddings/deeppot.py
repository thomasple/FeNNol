import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict, Optional, Union, Callable, ClassVar
import numpy as np
import dataclasses

from ..misc.nets import FullyConnectedNet, ChemicalNet

from ...utils.periodic_table import PERIODIC_TABLE, VALENCE_STRUCTURE
from ..misc.encodings import SpeciesEncoding, RadialBasis


class DeepPotEmbedding(nn.Module):
    """Deep Potential embedding

    FID : DEEPPOT

    ### Reference
    Zhang, L., Han, J., Wang, H., Car, R., & E, W. (2018). Deep Potential Molecular dynamics: A scalable model with the accuracy of quantum mechanics. Phys. Rev. Lett., 120(14), 143001. https://doi.org/10.1103/PhysRevLett.120.143001

    """
    _graphs_properties: Dict
    dim: int = 64
    """The dimension of the embedding."""
    subdim: int = 8
    """The first dimensions to select for the embedding tensor product."""
    radial_dim: Optional[int] = None
    """The dimension of the radial embedding for tensor combination. 
        If None, we use a neural net to combine chemical and radial information, like in the original DeepPot."""
    embedding_key: str = "embedding"
    """The key to use for the output embedding in the returned dictionary."""
    graph_key: str = "graph"
    """The key in the input dictionary that corresponds to the radial graph."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """The species encoding parameters. See `fennol.models.misc.encodings.SpeciesEncoding`"""
    radial_basis: Optional[dict] = None
    """The radial basis parameters. See `fennol.models.misc.encodings.RadialBasis`. 
        If None, the radial basis is the s_ij like in the original DeepPot."""
    embedding_hidden: Sequence[int] = dataclasses.field(
        default_factory=lambda: [64, 64, 64]
    )
    """The hidden layers of the embedding network."""
    activation: Union[Callable, str] = "silu"
    """The activation function."""
    concatenate_species: bool = False
    """Whether to concatenate the species encoding with the embedding."""
    divide_distances: bool = True
    """Whether to divide the switch by the distance in s_ij."""
    species_order: Optional[Union[str,Sequence[str]]] = None
    """Species considered by the network when using species-specialized embedding network."""

    FID: ClassVar[str] = "DEEPPOT"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]

        # species encoding
        if self.species_order is None or self.concatenate_species:
            onehot = SpeciesEncoding(**self.species_encoding, name="SpeciesEncoding")(
                species
            )

        # Radial graph
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        distances = graph["distances"][:, None]
        switch = graph["switch"][:, None]
        vec = graph["vec"] / distances
        sij = switch / distances if self.divide_distances else switch
        Rij = jnp.concatenate((sij, sij * vec), axis=-1)

        # Radial BASIS
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        if self.radial_basis is not None:
            radial_terms = RadialBasis(
                **{
                    **self.radial_basis,
                    "end": cutoff,
                    "name": f"RadialBasis",
                }
            )(graph["distances"])
        else:
            radial_terms = sij

        if self.species_order is not None:
            Gij = ChemicalNet(
                self.species_order,
                [*self.embedding_hidden, self.dim],
                activation=self.activation,
            )((species[edge_dst], radial_terms))
        elif self.radial_dim is not None:
            Gij = FullyConnectedNet(
                [*self.embedding_hidden, self.radial_dim], activation=self.activation
            )(radial_terms)
            Wa = self.param(
                f"Wa",
                nn.initializers.normal(
                    stddev=1.0 / (Gij.shape[1] * onehot.shape[1]) ** 0.5
                ),
                (onehot.shape[1], Gij.shape[1], self.dim),
            )
            Gij = jnp.einsum(
                "...i,...j,ijk->...k",
                onehot[edge_dst],
                Gij,
                Wa,
            )
        else:
            Gij = FullyConnectedNet(
                [*self.embedding_hidden, self.dim], activation=self.activation
            )(jnp.concatenate((radial_terms, onehot[edge_dst]), axis=-1))

        GRi = jax.ops.segment_sum(
            Gij[:, None, :] * Rij[:, :, None], edge_src, species.shape[0]
        )
        if self.subdim > 0:
            GRisub = GRi[:, :, : self.subdim]

            embedding = (
                (GRi[:, :, :, None] * GRisub[:, :, None, :])
                .sum(axis=1)
                .reshape((species.shape[0], -1))
            )
        else:
            GRisub = nn.Dense(self.dim, use_bias=False, name="Gri_linear")(GRi)
            embedding = (GRi * GRisub).sum(axis=1)

        if self.concatenate_species:
            embedding = jnp.concatenate((onehot, embedding), axis=-1)

        if self.embedding_key is None:
            return embedding
        return {**inputs, self.embedding_key: embedding}


class DeepPotE3Embedding(nn.Module):
    """Deep Potential embedding with angle information

    FID : DEEPPOT_E3

    ### Reference
    L. Zhang, J. Han, H. Wang, W. A. Saidi, R. Car, Weinan E, End-to-end Symmetry Preserving Inter-atomic Potential Energy Model for Finite and Extended Systems,
    Conference on Neural Information Processing Systems (NeurIPS), 2018,
    https://doi.org/10.48550/arXiv.1805.09003

    """
    _graphs_properties: Dict
    dim: int = 64
    """The dimension of the embedding."""
    embedding_key: str = "embedding"
    """The key to use for the output embedding in the returned dictionary."""
    graph_key: str = "graph"
    """The key in the input dictionary that corresponds to the graph."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """The species encoding parameters. See `fennol.models.misc.encodings.SpeciesEncoding`"""
    embedding_hidden: Sequence[int] = dataclasses.field(
        default_factory=lambda: [64, 64, 64]
    )
    """The hidden layers of the embedding network."""
    activation: Union[Callable, str] = "silu"
    """The activation function."""
    concatenate_species: bool = False
    """Whether to concatenate the species encoding with the embedding."""
    divide_distances: bool = True
    """Whether to divide the switch by the distance in s_ij."""

    FID: ClassVar[str] = "DEEPPOT_E3"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]

        # species encoding
        onehot = SpeciesEncoding(**self.species_encoding, name="SpeciesEncoding")(
            species
        )

        # Radial graph
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        distances = graph["distances"][:, None]
        switch = graph["switch"][:, None]
        vec = graph["vec"] / distances
        sij = switch / distances if self.divide_distances else switch
        Rij = jnp.concatenate((sij, sij * vec), axis=-1)

        zdest = onehot[edge_dst]

        angle_src, angle_dst = graph["angle_src"], graph["angle_dst"]
        z_angsrc = zdest[angle_src]
        z_angdst = zdest[angle_dst]

        # Radial BASIS
        assert (
            "angles" in graph
        ), "Error: DeepPotE3 requires angles (GRAPH_ANGLE_EXTENSION)"
        theta = (Rij[angle_src] * Rij[angle_dst]).sum(axis=-1,keepdims=True)

        Ne3 = FullyConnectedNet(
            [*self.embedding_hidden, self.dim],
            activation=self.activation,
            name="Ne3",
        )
        Gijk = Ne3(jnp.concatenate((theta, z_angsrc, z_angdst), axis=-1)) + Ne3(
            jnp.concatenate((theta, z_angdst, z_angsrc), axis=-1)
        )

        embedding = jax.ops.segment_sum(
            Gijk * theta, graph["central_atom"], species.shape[0]
        )

        if self.concatenate_species:
            embedding = jnp.concatenate((onehot, embedding), axis=-1)

        if self.embedding_key is None:
            return embedding
        return {**inputs, self.embedding_key: embedding}
