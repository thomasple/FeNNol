import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics, CG_SO3
from ..misc.encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import Dict, Union, Callable, Sequence, Optional, ClassVar
from ...utils.activations import activation_from_str
from ..misc.nets import FullyConnectedNet


class NewtonNetEmbedding(nn.Module):
    """ Newtonian message passing network

    ### Reference
    Haghighatlari et al., NewtonNet: a Newtonian message passing network for deep learning of interatomic potentials and forces
    https://doi.org/10.1039/D2DD00008C

    """
    
    _graphs_properties: Dict
    dim: int = 128
    """The dimension of the embedding."""
    nlayers: int = 3
    """The number of interaction layers."""
    nchannels: Optional[int] = None
    """The number of vector channels. If None, it is set to dim."""
    embedding_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
    """The hidden layers for the embedding networks."""
    latent_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [128])
    """The hidden layers for the latent update network."""
    activation: Union[Callable, str] = "silu"
    """The activation function."""
    graph_key: str = "graph"
    """The key for the graph input."""
    embedding_key: str = "embedding"
    """The key for the output embedding."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """The species encoding parameters. See `fennol.models.misc.encodings.SpeciesEncoding`."""
    radial_basis: dict = dataclasses.field(default_factory=dict)
    """The radial basis parameters. See `fennol.models.misc.encodings.RadialBasis`."""
    keep_all_layers: bool = False
    """Whether to keep embeddings from each layer in the output."""

    FID: ClassVar[str] = "NEWTONNET"


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
        dirij = graph["vec"] / distances[:, None] * switch

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
            ai = FullyConnectedNet(
                [*self.embedding_hidden, self.dim],
                activation=self.activation,
                name=f"phi_a_{layer}",
                use_bias=True,
            )(xi)
            Dij = nn.Dense(self.dim, name=f"radial_linear_{layer}", use_bias=True)(
                radial_basis
            )
            mij = ai[edge_src] * ai[edge_dst] * Dij * switch

            mi = jax.ops.segment_sum(mij, edge_src, xi.shape[0])
            xi = xi + mi

            Fij = (
                FullyConnectedNet(
                    [*self.embedding_hidden, 1],
                    activation=self.activation,
                    name=f"phi_F_{layer}",
                    use_bias=True,
                )(mij)
                * dirij
            )

            fij = (
                FullyConnectedNet(
                    [*self.embedding_hidden, nchannels],
                    activation=self.activation,
                    name=f"phi_f_{layer}",
                    use_bias=True,
                )(mij)[:, :, None]
                * Fij[:, None, :]
            )

            if layer == 0:
                fi = jax.ops.segment_sum(fij, edge_src, xi.shape[0])
            else:
                fi = fi + jax.ops.segment_sum(fij, edge_src, xi.shape[0])

            deltai = (
                FullyConnectedNet(
                    [*self.embedding_hidden, nchannels],
                    activation=self.activation,
                    name=f"phi_R_{layer}",
                    use_bias=True,
                )(xi)[:, :, None]
                * fi
            )
            if layer == 0:
                di = deltai
            else:
                phi_rij = FullyConnectedNet(
                    [*self.embedding_hidden, nchannels],
                    activation=self.activation,
                    name=f"phi_r_{layer}",
                    use_bias=True,
                )(mij)
                
                phi_r = jax.ops.segment_sum(phi_rij * switch, edge_src, xi.shape[0])
                di = phi_r[:, :, None] * di + deltai

            scal = jnp.sum(fi * di, axis=-1)
            dui = (
                -FullyConnectedNet(
                    [*self.latent_hidden, nchannels],
                    activation=self.activation,
                    name=f"phi_u_{layer}",
                    use_bias=True,
                )(xi)
                * scal
            )

            if nchannels != self.dim:
                dui = nn.Dense(self.dim, name=f"reshape_{layer}", use_bias=False)(dui)

            xi = xi + dui
            if self.keep_all_layers:
                xis.append(xi)

        output = {
            **inputs,
            self.embedding_key: xi,
        }
        if self.keep_all_layers:
            output[self.embedding_key + "_layers"] = jnp.stack(xis, axis=1)
        return output
