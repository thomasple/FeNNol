import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics
from ..misc.encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import Any, Dict, List, Union, Callable, Tuple, Sequence, Optional, ClassVar
from ..misc.nets import FullyConnectedNet
from ..misc.e3 import (
    FilteredTensorProduct,
    ChannelMixingE3,
    ChannelMixing,
    E3NN_AVAILABLE,
    E3NN_EXCEPTION,
    Irreps,
)


class AllegroEmbedding(nn.Module):
    """Allegro equivariant pair embedding

    FID : ALLEGRO

    ### Reference
    Musaelian, A., Batzner, S., Johansson, A. et al. Learning local equivariant representations for large-scale atomistic dynamics.
      Nat Commun 14, 579 (2023). https://doi.org/10.1038/s41467-023-36329-y

    """

    _graphs_properties: Dict
    dim: int = 128
    """ The dimension of the embedding."""
    nchannels: int = 16
    """ The number of equivariant channels."""
    nlayers: int = 3
    """ The number of interaction layers."""
    lmax: int = 2
    """ The maximum degree of tensorial embedding."""
    lmax_density: Optional[int] = None
    """ The maximum degree of spherical harmonics for density.
        If None, it will be set to lmax. Must be greater or equal to lmax."""
    twobody_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
    """ The number of hidden neurons in the two-body network."""
    embedding_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [])
    """ The number of hidden neurons in the embedding network."""
    latent_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
    """ The number of hidden neurons in the latent network."""
    activation: Union[Callable, str] = "silu"
    """ The activation function to use."""
    graph_key: str = "graph"
    """ The key in the input dictionary that corresponds to the graph."""
    embedding_key: str = "embedding"
    """ The key to use for the output embedding in the returned dictionary."""
    tensor_embedding_key: str = "tensor_embedding"
    """ The key to use for the output tensor embedding in the returned dictionary."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """ The species encoding parameters.  See `fennol.models.misc.encodings.SpeciesEncoding`"""
    radial_basis: dict = dataclasses.field(default_factory=dict)
    """ The radial basis parameters. See `fennol.models.misc.encodings.RadialBasis`"""

    FID: ClassVar[str] = "ALLEGRO"

    @nn.compact
    def __call__(self, inputs):
        """ Forward pass of the Allegro model. """
        species = inputs["species"]

        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        switch = graph["switch"][:, None]
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        radial_basis = RadialBasis(
            **{**self.radial_basis, "end": cutoff, "name": "RadialBasis"}
        )(graph["distances"])

        species_encoding = SpeciesEncoding(
            **self.species_encoding, name="SpeciesEncoding"
        )(species)

        xij = (
            FullyConnectedNet(
                neurons=[*self.twobody_hidden, self.dim], activation=self.activation
            )(
                jnp.concatenate(
                    [
                        species_encoding[edge_src],
                        species_encoding[edge_dst],
                        radial_basis,
                    ],
                    axis=-1,
                )
            )
            * switch
        )

        lmax_density = self.lmax_density if self.lmax_density is not None else self.lmax
        assert lmax_density >= self.lmax

        Yij = generate_spherical_harmonics(lmax=lmax_density, normalize=False)(
            graph["vec"] / graph["distances"][:, None]  
        )[:, None, :]

        nel = (self.lmax + 1) ** 2
        Vij = (
            ChannelMixingE3(self.lmax, 1, self.nchannels)(Yij[..., :nel])
            * nn.Dense(self.nchannels, use_bias=False)(xij)[:, :, None]
        )

        for _ in range(self.nlayers):
            rhoij = (
                FullyConnectedNet(
                    neurons=[*self.embedding_hidden, self.nchannels],
                    activation=self.activation,
                )(xij)
                * switch
            )[:, :, None] * Yij
            density = (
                jnp.zeros((species.shape[0], *rhoij.shape[1:])).at[edge_src].add(rhoij)
            )

            Lij = FilteredTensorProduct(self.lmax, lmax_density)(Vij, density[edge_src])
            scals = jax.lax.index_in_dim(Lij, 0, axis=-1, keepdims=False)
            lij = FullyConnectedNet(neurons=[*self.latent_hidden, self.dim])(
                jnp.concatenate((xij, scals), axis=-1)
            )

            xij = xij + lij * switch
            Vij = ChannelMixing(self.lmax, self.nchannels, self.nchannels)(Lij)

        if self.embedding_key is None:
            return xij, Vij
        return {**inputs, self.embedding_key: xij, self.tensor_embedding_key: Vij}


if E3NN_AVAILABLE:
    import e3nn_jax as e3nn

    class AllegroE3NNEmbedding(nn.Module):
        """Allegro equivariant pair embedding

        FID : ALLEGRO_E3NN

        in this version, equivariant operations use the e3nn library.

        Reference
        ---------
        Musaelian, A., Batzner, S., Johansson, A. et al. Learning local equivariant representations for large-scale atomistic dynamics.
        Nat Commun 14, 579 (2023). https://doi.org/10.1038/s41467-023-36329-y

        """

        _graphs_properties: Dict
        dim: int = 128
        """ The dimension of the embedding."""
        nchannels: int = 16
        """ The number of equivariant channels."""
        nlayers: int = 3
        """ The number of interaction layers."""
        irreps_Vij: Union[str, int, 'Irreps'] = 2
        """ Irreps used for the tensor embedding."""
        lmax_density: int = None
        """ The maximum degree of spherical harmonics for density.
            If None, it will be set to lmax. Must be greater or equal to lmax."""
        twobody_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
        """ The number of hidden neurons in the two-body network."""
        embedding_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [])
        """ The number of hidden neurons in the embedding network."""
        latent_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
        """ The number of hidden neurons in the latent network."""
        activation: Union[Callable, str] = "silu"
        """ The activation function to use."""
        graph_key: str = "graph"
        """ The key in the input dictionary that corresponds to the graph."""
        embedding_key: str = "embedding"
        """ The key to use for the output embedding in the returned dictionary."""
        tensor_embedding_key: str = "tensor_embedding"
        """ The key to use for the output tensor embedding in the returned dictionary."""
        species_encoding: dict = dataclasses.field(default_factory=dict)
        """ The species encoding parameters."""
        radial_basis: dict = dataclasses.field(default_factory=dict)
        """ The radial basis parameters."""

        FID: ClassVar[str] = "ALLEGRO_E3NN"
        """ Identification of the module when building a model."""

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
            radial_basis = RadialBasis(
                **{**self.radial_basis, "end": cutoff, "name": "RadialBasis"}
            )(graph["distances"])
            radial_size = radial_basis.shape[-1]

            species_encoding = SpeciesEncoding(
                **self.species_encoding, name="SpeciesEncoding"
            )(species)
            afvs_size = species_encoding.shape[-1]

            xij = (
                FullyConnectedNet(
                    neurons=[*self.twobody_hidden, self.dim], activation=self.activation
                )(
                    jnp.concatenate(
                        [
                            species_encoding[edge_src],
                            species_encoding[edge_dst],
                            radial_basis,
                        ],
                        axis=-1,
                    )
                )
                * switch
            )
            if isinstance(self.irreps_Vij, int):
                irreps_Vij = e3nn.Irreps.spherical_harmonics(self.irreps_Vij)
            elif isinstance(self.irreps_Vij, str):
                irreps_Vij = e3nn.Irreps(self.irreps_Vij)
            else:
                irreps_Vij = self.irreps_Vij
            lmax = max(irreps_Vij.ls)
            lmax_density = self.lmax_density or lmax
            irreps_density = e3nn.Irreps.spherical_harmonics(lmax_density)

            # Yij = e3nn.IrrepsArray(
            #     irreps_density,
            #     generate_spherical_harmonics(lmax=lmax_density, normalize=False)(
            #         graph["vec"] / graph["distances"][:, None]
            #     ),
            # )[:, None, :]
            Yij = e3nn.spherical_harmonics(
                irreps_density, graph["vec"], normalize=True
            )[:, None, :]

            Vij = (
                e3nn.flax.Linear(irreps_Vij, channel_out=self.nchannels)(Yij)
                * nn.Dense(self.nchannels, use_bias=False)(xij)[:, :, None]
            )

            for _ in range(self.nlayers):
                rhoij = (
                    FullyConnectedNet(
                        neurons=[*self.embedding_hidden, self.nchannels],
                        activation=self.activation,
                    )(xij)
                    * switch
                )[:, :, None] * Yij
                density = e3nn.scatter_sum(
                    rhoij, dst=edge_src, output_size=species_encoding.shape[0]
                )

                Lij = e3nn.tensor_product(
                    Vij, density[edge_src], filter_ir_out=irreps_Vij
                )
                scals = Lij.filter(["0e"]).array.reshape(Lij.shape[0], -1)
                lij = FullyConnectedNet(neurons=[*self.latent_hidden, self.dim])(
                    jnp.concatenate((xij, scals), axis=-1)
                )

                xij = xij + lij * switch
                # filtering
                Lij = e3nn.flax.Linear(irreps_Vij)(Lij)
                # channel mixing
                Vij = e3nn.flax.Linear(irreps_Vij, channel_out=self.nchannels)(Lij)

            if self.embedding_key is None:
                return xij, Vij
            return {**inputs, self.embedding_key: xij, self.tensor_embedding_key: Vij}

else:

    class AllegroE3NNEmbedding(nn.Module):
        FID: ClassVar[str] = "ALLEGRO_E3NN"

        def __call__(self, *args, **kwargs) -> Any:
            raise E3NN_EXCEPTION
