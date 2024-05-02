import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics, CG_SO3
from ..misc.encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import Dict, Union, Callable, Sequence, Optional, ClassVar
from ...utils.activations import activation_from_str, tssr3
from ..misc.nets import FullyConnectedNet


class PAINNEmbedding(nn.Module):
    """polarizable atom interaction neural network
    
    ### Reference
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller. SchNet - a deep learning architecture for molecules and materials. The Journal of Chemical Physics 148(24), 241722 (2018) 
    https://doi.org/10.1063/1.5019779

    """

    _graphs_properties: Dict
    dim: int = 128
    """ The dimension of the embedding. """
    nlayers: int = 3
    """ The number of interaction layers. """
    nchannels: Optional[int] = None
    """ The number of equivariant channels. If None, it is set to dim. """
    message_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
    """ The hidden layers for the message network."""
    update_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
    """ The hidden layers for the update network."""
    activation: Union[Callable, str] = "silu"
    """ The activation function."""
    graph_key: str = "graph"
    """ The key for the graph input."""
    embedding_key: str = "embedding"
    """ The key for the embedding output."""
    tensor_embedding_key: str = "embedding_vectors"
    """ The key for the tensor embedding output."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """ The species encoding parameters. See `fennol.models.misc.encodings.SpeciesEncoding`."""
    radial_basis: dict = dataclasses.field(default_factory=dict)
    """ The radial basis function parameters. See `fennol.models.misc.encodings.RadialBasis`."""
    keep_all_layers: bool = False
    """ Whether to keep the embedding from each layer in the output."""

    FID: ClassVar[str] = "PAINN"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        assert (
            len(species.shape) == 1
        ), "Species must be a 1D array (batches must be flattened)"

        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]

        cutoff = self._graphs_properties[self.graph_key]["cutoff"]

        onehot = SpeciesEncoding(**self.species_encoding, name="SpeciesEncoding")(
            species
        )
        xi = nn.Dense(self.dim, name="species_linear", use_bias=True)(onehot)

        nchannels = self.nchannels if self.nchannels is not None else self.dim

        distances = graph["distances"]
        switch = graph["switch"][:, None]
        dirij = (graph["vec"] / distances[:, None])[:, :,None]
        Vi = jnp.zeros((xi.shape[0], 3, nchannels), dtype=xi.dtype)

        radial_basis = RadialBasis(
            **{
                **self.radial_basis,
                "end": cutoff,
                "name": f"RadialBasis",
            }
        )(distances)

        if self.keep_all_layers:
            xis = []
        for layer in range(self.nlayers):
            # compute messages
            phi = FullyConnectedNet(
                [*self.message_hidden, self.dim + 2 * nchannels],
                activation=self.activation,
                name=f"message_{layer}",
                use_bias=True,
            )(xi)
            w = (
                nn.Dense(
                    self.dim + 2 * nchannels,
                    name=f"radial_linear_{layer}",
                    use_bias=True,
                )(radial_basis)
                * switch
            )
            dxij, hvv, hvs = jnp.split(
                phi[edge_dst] * w, [self.dim, self.dim + nchannels], axis=-1
            )

            dvij = dirij * hvs[:, None,:]
            if layer > 0:
                dvij = dvij + Vi[edge_dst] * hvv[:, None,:]

            # aggregate messages
            v_message = Vi + jax.ops.segment_sum(dvij, edge_src, Vi.shape[0])
            x_message = xi + jax.ops.segment_sum(dxij, edge_src, xi.shape[0])

            # update
            u,v = jnp.split(
                nn.Dense(
                    2 * self.nchannels,
                    use_bias=False,
                    name=f"UV_{layer}",
                )(v_message),
                2,
                axis=-1,
            )

            scals = (u * v).sum(axis=1)
            norms = tssr3((v**2).sum(axis=1))

            A = FullyConnectedNet(
                [*self.update_hidden, self.dim + 2 * nchannels],
                activation=self.activation,
                name=f"update_{layer}",
                use_bias=True,
            )(jnp.concatenate((x_message, norms), axis=-1))

            ass, asv, avv = jnp.split(
                A,
                [self.dim, self.dim + nchannels],
                axis=-1,
            )

            Vi = Vi + u * avv[:, None,:]
            if self.dim != nchannels:
                dxi = nn.Dense(self.dim, name=f"resize_{layer}", use_bias=False)(
                    scals * asv
                )
            else:
                dxi = scals * asv

            xi = xi + ass + dxi

            if self.keep_all_layers:
                xis.append(xi)

        output = {
            **inputs,
            self.embedding_key: xi,
            self.tensor_embedding_key: Vi,
        }
        if self.keep_all_layers:
            output[self.embedding_key + "_layers"] = jnp.stack(xis, axis=1)
        return output
