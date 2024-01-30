import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics
from ..encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import Any, Dict, List, Union, Callable, Tuple, Sequence, Optional
from ..nets import FullyConnectedNet
from ..e3 import FilteredTensorProduct, ChannelMixingE3, ChannelMixing


class MiniMaceEmbedding(nn.Module):
    _graphs_properties: Dict
    dim: int = 128
    nchannels: int = 16
    message_dim: int = 16
    nlayers: int = 2
    ntp: int = 2
    lmax: int = 2
    embedding_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [])
    latent_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
    activation: Union[Callable, str] = nn.silu
    graph_key: str = "graph"
    embedding_key: str = "embedding"
    tensor_embedding_key: str = "tensor_embedding"
    species_encoding: dict = dataclasses.field(default_factory=dict)
    radial_basis: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        assert (
            len(species.shape) == 1
        ), "Species must be a 1D array (batches must be flattened)"
        # nchannels_density = (
        #     self.nchannels_density
        #     if self.nchannels_density is not None
        #     else self.nchannels
        # )
        # nrep = np.array([2 * l + 1 for l in range(self.lmax + 1)])

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

        Yij = generate_spherical_harmonics(lmax=self.lmax, normalize=False)(
            graph["vec"] / graph["distances"][:, None]
        )[:, None, :]

        species_encoding = SpeciesEncoding(
            **self.species_encoding, name="SpeciesEncoding"
        )(species)

        xi = FullyConnectedNet(
            neurons=[*self.embedding_hidden, self.dim],
            activation=self.activation,
            use_bias=True,
            name="species_embedding",
        )(species_encoding)

        nchannels_density = self.message_dim * radial_basis.shape[1]

        for layer in range(self.nlayers):
            mi = nn.Dense(
                self.message_dim,
                use_bias=True,
                name=f"species_linear_{layer}",
            )(xi)
            xij = (mi[edge_dst, :, None] * radial_basis[:, None, :]).reshape(
                -1, nchannels_density
            )
            if layer == 0:
                rhoij = xij[:, :, None] * Yij
                density = jax.ops.segment_sum(rhoij, edge_src, species.shape[0])
                Vi = ChannelMixingE3(
                    self.lmax,
                    nchannels_density,
                    self.nchannels,
                    name=f"Vi_initial",
                )(density)
            else:
                rhoi = ChannelMixingE3(
                    self.lmax,
                    self.nchannels,
                    nchannels_density,
                    name=f"rho_mixing_{layer}",
                )(Vi)
                rhoij = xij[:, :, None] * rhoi[edge_dst]
                density = density + jax.ops.segment_sum(
                    rhoij, edge_src, species.shape[0]
                )

            scals = [jax.lax.index_in_dim(density, 0, axis=-1, keepdims=False)]
            for i in range(self.ntp):
                Hi = ChannelMixing(
                    self.lmax,
                    nchannels_density,
                    self.nchannels,
                    name=f"density_mixing_{layer}_{i}",
                )(density)
                Li = FilteredTensorProduct(
                    self.lmax,
                    self.lmax,
                    self.lmax,
                    name=f"TP_{layer}_{i}",
                    ignore_parity=True,
                )(Vi, Hi)
                scals.append(jax.lax.index_in_dim(Li, 0, axis=-1, keepdims=False))
                Vi = Vi + Li

            dxi = FullyConnectedNet(
                [*self.latent_hidden, self.dim],
                activation=self.activation,
                use_bias=True,
                name=f"latent_net_{layer}",
            )(jnp.concatenate([xi, *scals], axis=-1))
            xi = xi + dxi

        if self.embedding_key is None:
            return xi, Vi
        return {**inputs, self.embedding_key: xi, self.tensor_embedding_key: Vi}
