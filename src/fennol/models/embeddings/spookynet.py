import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics, CG_SO3
from ..encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import Dict, Union, Callable, Sequence, Optional
from ...utils.activations import activation_from_str, tssr2
from ...utils.initializers import initializer_from_str, scaled_orthogonal
from ..nets import FullyConnectedNet, ResMLP


class SpookyNetEmbedding(nn.Module):
    _graphs_properties: Dict
    dim: int = 128
    nlayers: int = 3
    graph_key: str = "graph"
    embedding_key: str = "embedding"
    species_encoding: dict = dataclasses.field(
        default_factory=lambda: {"encoding": "electronic_structure"}
    )
    radial_basis: dict = dataclasses.field(default_factory=lambda: {"basis": "spooky"})
    kernel_init: Union[Callable, str] = scaled_orthogonal(scale=1.0, mode="fan_avg")

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        assert (
            len(species.shape) == 1
        ), "Species must be a 1D array (batches must be flattened)"

        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]

        cutoff = self._graphs_properties[self.graph_key]["cutoff"]

        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )

        onehot = SpeciesEncoding(**self.species_encoding, name="SpeciesEncoding")(
            species
        )
        zrand = SpeciesEncoding(
            encoding="random", dim=self.dim, name="RandSpeciesEncoding"
        )(species)

        xi = (
            nn.Dense(
                self.dim,
                name="species_linear",
                use_bias=False,
                kernel_init=nn.initializers.zeros,
            )(onehot)
            + zrand
        )

        distances = graph["distances"]
        switch = graph["switch"][:, None]
        dirij = graph["vec"] / distances[:, None]
        Yij = generate_spherical_harmonics(lmax=2, normalize=False)(dirij)[:, None, :]

        radial_basis = (
            RadialBasis(
                **{
                    **self.radial_basis,
                    "end": cutoff,
                    "name": f"RadialBasis",
                }
            )(distances)
            * switch
        )

        gij = radial_basis[:, :, None] * Yij

        gsij = gij[:, :, 0]
        gpij = jnp.transpose(gij[:, :, 1:4], (0, 2, 1))
        gdij = jnp.transpose(gij[:, :, 4:], (0, 2, 1))

        y = 0.0

        for layer in range(self.nlayers):
            xtilde = ResMLP(res_only=True, name=f"xtilde_{layer}")(xi)

            ### compute local update
            c = ResMLP(name=f"c_{layer}", kernel_init=kernel_init)(xtilde)
            sj = ResMLP(name=f"s_{layer}", kernel_init=kernel_init)(xtilde)
            pj = ResMLP(name=f"p_{layer}", kernel_init=kernel_init)(xtilde)
            dj = ResMLP(name=f"d_{layer}", kernel_init=kernel_init)(xtilde)

            Gs = nn.Dense(
                self.dim, use_bias=False, name=f"Gs_{layer}", kernel_init=kernel_init
            )(gsij)
            Gp = nn.Dense(
                self.dim, use_bias=False, name=f"Gp_{layer}", kernel_init=kernel_init
            )(gpij)
            Gd = nn.Dense(
                self.dim, use_bias=False, name=f"Gd_{layer}", kernel_init=kernel_init
            )(gdij)

            si = jax.ops.segment_sum(sj[edge_dst] * Gs, edge_src, xi.shape[0])
            pi = jax.ops.segment_sum(pj[edge_dst, None, :] * Gp, edge_src, xi.shape[0])
            di = jax.ops.segment_sum(dj[edge_dst, None, :] * Gd, edge_src, xi.shape[0])

            P1, P2 = jnp.split(
                nn.Dense(
                    2 * self.dim,
                    use_bias=False,
                    name=f"P12_{layer}",
                    kernel_init=kernel_init,
                )(pi),
                2,
                axis=-1,
            )
            D1, D2 = jnp.split(
                nn.Dense(
                    2 * self.dim,
                    use_bias=False,
                    name=f"D12_{layer}",
                    kernel_init=kernel_init,
                )(di),
                2,
                axis=-1,
            )

            P12 = (P1 * P2).sum(axis=1)
            D12 = (D1 * D2).sum(axis=1)

            l = ResMLP(name=f"l_{layer}", kernel_init=kernel_init)(c + si + P12 + D12)

            ### aggregate and update
            xi = ResMLP(name=f"xi_{layer}", kernel_init=kernel_init)(xtilde + l)
            y = y + ResMLP(name=f"y_{layer}", kernel_init=kernel_init)(xi)

        output = {
            **inputs,
            self.embedding_key: y,
        }
        return output
