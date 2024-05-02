import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics, CG_SO3
from ..misc.encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import Dict, Union, Callable, ClassVar
from ...utils.activations import activation_from_str, tssr3
from ..misc.nets import FullyConnectedNet


class HIPNNEmbedding(nn.Module):
    """Hierarchically Interacting Particle Neural Network

    ### Reference
    Adapted from N. Lubbers, J. S. Smith and K. Barros, Hierarchical modeling of molecular energies using a deep neural network
    J. Chem. Phys. 148, 241715 (2018) (J. Chem. Phys. 148, 241715 (2018)) (https://doi.org/10.1063/1.5011181)

    """
    
    _graphs_properties: Dict
    dim: int = 80
    """The dimension of the embedding."""
    n_onsite: int = 3
    """The number of onsite layers per interaction layer."""
    nlayers: int = 2
    """The number of interaction layers."""
    lmax: int = 0
    """The maximum value degree of spherical harmonics."""
    n_message: int = 0
    """The number of layers for the message NN."""
    activation: Union[Callable, str] = "silu"
    """The activation function."""
    graph_key: str = "graph"
    """The key for the graph input."""
    embedding_key: str = "embedding"
    """The key for the embedding output."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """The species encoding parameters. See `fennol.models.misc.encodings.SpeciesEncoding`."""
    radial_basis: dict = dataclasses.field(default_factory=lambda: {"dim": 20})
    """The radial basis parameters. See `fennol.models.misc.encodings.RadialBasis`."""
    keep_all_layers: bool = True
    """Whether to keep embeddings from each layer in the output."""
    graph_l_key: str = "graph"
    """The key for the graph input for the spherical harmonics."""

    FID: ClassVar[str] = "HIPNN"


    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        assert (
            len(species.shape) == 1
        ), "Species must be a 1D array (batches must be flattened)"

        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]

        cutoff = self._graphs_properties[self.graph_key]["cutoff"]

        zi = SpeciesEncoding(**self.species_encoding, name="SpeciesEncoding")(species)

        act = (
            activation_from_str(self.activation)
            if isinstance(self.activation, str)
            else self.activation
        )

        distances = graph["distances"]
        if self.lmax > 0:
            filtered_l = "parent_graph" in self._graphs_properties[self.graph_l_key]

            correct_graph = (
                self.graph_l_key == self.graph_key
                or self._graphs_properties[self.graph_l_key]["parent_graph"]
                == self.graph_key
            )
            assert (
                correct_graph
            ), f"graph_l_key={self.graph_l_key} must be a subgraph of graph_key={self.graph_key}"

            graph_l = inputs[self.graph_l_key]
            Yij = generate_spherical_harmonics(lmax=self.lmax, normalize=False)(
                graph_l["vec"] / graph_l["distances"][:, None]
            )[:, None, :]
            if filtered_l:
                Yij = Yij[:, :, 1:]
            # reps_l = np.array([2 * l + 1 for l in range(1, self.lmax + 1)])

        switch = graph["switch"][:, None]
        if self.keep_all_layers:
            zis = []
        for layer in range(self.nlayers):
            ### interaction layer
            s = RadialBasis(
                **{
                    "basis": "gaussian_rinv",
                    "end": cutoff,
                    **self.radial_basis,
                    "name": f"RadialBasis_{layer}",
                }
            )(distances)

            zself = nn.Dense(self.dim, name=f"self_int_{layer}", use_bias=True)(zi)
            if self.n_message == 0:
                V = self.param(
                    f"V_{layer}",
                    jax.nn.initializers.glorot_normal(batch_axis=0),
                    (s.shape[-1], zi.shape[-1], self.dim),
                )
                mij = jnp.einsum("...j,...k,jkl->...l", s, zi[edge_dst], V)
            else:
                mij = FullyConnectedNet(
                    [self.dim] * (self.n_message + 1),
                    activation=act,
                    name=f"mij_{layer}",
                )(jnp.concatenate([zi[edge_dst], s], axis=-1))

            if self.lmax == 0:
                zi = act(zself.at[edge_src].add(mij * switch,mode="drop"))
            else:
                if filtered_l:
                    zself = zself.at[edge_src].add(mij * switch,mode="drop")
                    filter_indices = graph_l["filter_indices"]
                    mij = mij[filter_indices]

                mij = mij * graph_l["switch"][:, None]
                Mij = mij[:, :, None] * Yij
                Mi = jax.ops.segment_sum(Mij, graph_l["edge_src"], zi.shape[0])

                if filtered_l:
                    zint = jnp.zeros_like(zself)
                else:
                    zint = jax.lax.index_in_dim(Mi, 0, axis=-1, keepdims=False)
                    Mi = Mi[:, :, 1:]

                ts = self.param(
                    f"ts_{layer}",
                    lambda key, shape: jax.random.normal(key, shape, dtype=zint.dtype),
                    (self.lmax, 3),
                )  # .repeat(reps_l)

                # zint = zint + jnp.sum(ts[None,None,:]*Mi**2,axis=-1)
                for l in range(1, self.lmax + 1):
                    Ml = jax.lax.dynamic_slice_in_dim(
                        Mi, start_index=l**2 - 1, slice_size=2 * l + 1, axis=-1
                    )
                    zint = zint + ts[l, 2] * tssr3(
                        nn.Dense(self.dim, name=f"linear_{layer}_l{l}")(
                            jnp.sum(Ml**2, axis=-1)
                        )
                    )
                zi = act(zself + zint)

            ### onsite layers
            # zi = zi + FullyConnectedNet(
            #     [self.dim] * (self.n_onsite + 1),
            #     activation=act,
            #     use_bias=True,
            #     name=f"onsite_{layer}",
            # )(zi)
            for j in range(self.n_onsite):
                zi = zi + FullyConnectedNet(
                    (self.dim, self.dim),
                    activation=act,
                    use_bias=True,
                    name=f"onsite_{layer}_{j}",
                )(zi)
            if self.keep_all_layers:
                zis.append(zi)

        output = {
            **inputs,
            self.embedding_key: zi,
        }
        if self.keep_all_layers:
            output[self.embedding_key + "_layers"] = jnp.stack(zis, axis=1)
        return output
