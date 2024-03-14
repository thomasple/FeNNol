import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict, Optional, Union, Callable
import numpy as np
import dataclasses

from ..misc.nets import FullyConnectedNet, ChemicalNet

from ...utils.periodic_table import PERIODIC_TABLE, VALENCE_STRUCTURE
from ..misc.encodings import SpeciesEncoding, RadialBasis


class DeepPotEmbedding(nn.Module):
    _graphs_properties: Dict
    dim: int = 64
    subdim: int = 8
    radial_dim: Optional[int] = None
    embedding_key: str = "embedding"
    graph_key: str = "graph"
    species_encoding: dict = dataclasses.field(default_factory=dict)
    radial_basis: Optional[dict] = None
    embedding_hidden: Sequence[int] = dataclasses.field(
        default_factory=lambda: [64, 64, 64]
    )
    activation: Union[Callable, str] = nn.silu
    concatenate_species: bool = False
    divide_distances: bool = True
    species_order: Optional[Sequence[str]] = None

    FID: str = "DEEPPOT"

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
    _graphs_properties: Dict
    dim: int = 64
    embedding_key: str = "embedding"
    graph_key: str = "graph"
    species_encoding: dict = dataclasses.field(default_factory=dict)
    embedding_hidden: Sequence[int] = dataclasses.field(
        default_factory=lambda: [64, 64, 64]
    )
    activation: Union[Callable, str] = nn.silu
    concatenate_species: bool = False
    divide_distances: bool = True

    FID: str = "DEEPPOT_E3"

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
