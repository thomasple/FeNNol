import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics
from ..misc.encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import Any, Dict, List, Union, Callable, Tuple, Sequence, Optional
from ..misc.nets import FullyConnectedNet
from ..misc.e3 import FilteredTensorProduct, ChannelMixingE3


class CaimanEmbedding(nn.Module):
    """Covariant Atom-In-Molecule Network

    FID : CAIMAN

    This is en E(3) equivariant embedding that forms an equivariant neighbor density
    and then uses multiple self-interaction tensor products to generate a tensorial embedding
    along with a scalar embedding (similar to the tensor/scalar tracks in allegro).

    By default, the model is purely local so that multiple iterations are similar to one MACE layer
    with multiple tensor products. Optionally, the model can use message-passing to update the
    neighborhood density.

    Parameters:
        dim (int): The dimension of the embedding. Default is 128.
        nchannels (int): The number of channels. Default is 16.
        nchannels_density (Optional[int]): The number of density channels. Default is None.
        nlayers (int): The number of layers. Default is 3.
        lmax (int): The maximum value of l. Default is 2.
        twobody_hidden (Sequence[int]): The hidden layers for the two-body interaction. Default is [128].
        embedding_hidden (Sequence[int]): The hidden layers for the embedding. Default is [].
        latent_hidden (Sequence[int]): The hidden layers for the latent network. Default is [128].
        activation (Union[Callable, str]): The activation function. Default is nn.silu.
        graph_key (str): The key for the graph input. Default is "graph".
        embedding_key (str): The key for the embedding output. Default is "embedding".
        tensor_embedding_key (str): The key for the tensor embedding output. Default is "tensor_embedding".
        species_encoding (dict): The species encoding parameters. Default is an empty dictionary.
        radial_basis (dict): The radial basis parameters. Default is an empty dictionary.
        message_passing (bool): Whether to use message passing. Default is False.
        FID (str): The FID value. Default is "CAIMAN".

    """

    _graphs_properties: Dict
    dim: int = 128
    nchannels: int = 16
    nchannels_density: Optional[int] = None
    nlayers: int = 3
    lmax: int = 2
    twobody_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
    embedding_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [])
    latent_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
    activation: Union[Callable, str] = nn.silu
    graph_key: str = "graph"
    embedding_key: str = "embedding"
    tensor_embedding_key: str = "tensor_embedding"
    species_encoding: dict = dataclasses.field(default_factory=dict)
    radial_basis: dict = dataclasses.field(default_factory=dict)
    message_passing: bool = False

    FID: str = "CAIMAN"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        assert (
            len(species.shape) == 1
        ), "Species must be a 1D array (batches must be flattened)"
        nchannels_density = (
            self.nchannels_density
            if self.nchannels_density is not None
            else self.nchannels
        )

        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        switch = graph["switch"][:, None]
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        radial_basis = RadialBasis(
            **{**self.radial_basis, "end": cutoff, "name": "RadialBasis"}
        )(graph["distances"])

        Dij = (
            nn.Dense(nchannels_density, use_bias=True, name="Dij")(radial_basis)
            * switch
        )

        species_encoding = SpeciesEncoding(
            **self.species_encoding, name="SpeciesEncoding"
        )(species)

        xi = FullyConnectedNet(
            neurons=[*self.embedding_hidden, self.dim],
            activation=self.activation,
            use_bias=True,
            name="species_embedding",
        )(species_encoding)
        Zs, Zd = jnp.split(
            nn.Dense(2 * nchannels_density, use_bias=True, name="species_linear")(xi),
            2,
            axis=-1,
        )
        xij = Zs[edge_src] * Zd[edge_dst] * Dij

        Yij = generate_spherical_harmonics(lmax=self.lmax, normalize=False)(
            graph["vec"] / graph["distances"][:, None]
        )[:, None, :]

        rhoij = xij[:, :, None] * Yij

        nrep = np.array([2 * l + 1 for l in range(self.lmax + 1)])
        wsh = self.param(
            "wsh",
            lambda key, shape: jax.random.normal(key, shape, dtype=jnp.float32),
            (nchannels_density, self.lmax + 1),
        ).repeat(nrep, axis=-1)
        density = (
            jax.ops.segment_sum(rhoij, edge_src, species.shape[0]) * wsh[None, :, :]
        )

        nel = (self.lmax + 1) ** 2
        Vi = ChannelMixingE3(self.lmax, nchannels_density, self.nchannels)(
            density[..., :nel]
        )
        lambda_message = self.param(
            "lambda_message",
            lambda key: jnp.asarray(0.1, dtype=density.dtype),
        )

        for layer in range(self.nlayers):
            if self.message_passing:
                Zs, Zs = jnp.split(
                    nn.Dense(
                        2 * nchannels_density,
                        use_bias=True,
                        name=f"message_linear_{layer}",
                    )(xi),
                    2,
                    axis=-1,
                )
                mij = (
                    nn.Dense(
                        nchannels_density, use_bias=False, name=f"radial_linear_{layer}"
                    )(Dij)
                    * Zs[edge_src]
                    * Zd[edge_dst]
                )
                rhoij = (
                    mij[:, :, None]
                    * ChannelMixingE3(
                        self.lmax,
                        self.nchannels,
                        nchannels_density,
                        name=f"message_mixing_{layer}",
                    )(Vi)[edge_dst]
                )
                rhoi = jax.ops.segment_sum(rhoij, edge_src, species.shape[0])
                density = density + lambda_message * ChannelMixingE3(
                    self.lmax,
                    nchannels_density,
                    nchannels_density,
                    name=f"density_update_{layer}",
                )(rhoi)

            Hi = ChannelMixingE3(
                self.lmax,
                nchannels_density,
                self.nchannels,
                name=f"density_mixing_{layer}",
            )(density)

            Li = FilteredTensorProduct(
                self.lmax, self.lmax, self.lmax, name=f"TP_{layer}"
            )(Vi, Hi)
            scals = jax.lax.index_in_dim(Li, 0, axis=-1, keepdims=False)
            li = FullyConnectedNet(
                [*self.latent_hidden, self.dim],
                activation=self.activation,
                use_bias=True,
                name=f"latent_net_{layer}",
            )(jnp.concatenate((xi, scals), axis=-1))

            xi = xi + li
            Vi = Vi + ChannelMixingE3(
                self.lmax, self.nchannels, self.nchannels, name=f"mixing_{layer}"
            )(Li)

        if self.embedding_key is None:
            return xi, Vi
        return {**inputs, self.embedding_key: xi, self.tensor_embedding_key: Vi}
