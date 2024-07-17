import jax
import jax.numpy as jnp
import flax.linen as nn
from ...utils.spherical_harmonics import generate_spherical_harmonics, CG_SO3
from ..misc.encodings import SpeciesEncoding, RadialBasis
import dataclasses
import numpy as np
from typing import Dict, Union, Callable, Sequence, Optional, ClassVar
from ...utils.initializers import initializer_from_str, scaled_orthogonal
from ..misc.nets import  ResMLP


class SpookyNetEmbedding(nn.Module):
    """SpookyNet equivariant message-passing embedding with electronic encodings (charge + spin).

    FID : SPOOKYNET

    ### Reference
    Unke, O.T., Chmiela, S., Gastegger, M. et al. SpookyNet: Learning force fields with electronic degrees of freedom and nonlocal effects. Nat Commun 12, 7273 (2021). https://doi.org/10.1038/s41467-021-27504-0

    
    ### Warning
    non-local attention interaction is not yet implemented !

    """

    _graphs_properties: Dict
    dim: int = 128
    """ The dimension of the embedding."""
    nlayers: int = 3
    """ The number of interaction layers."""
    graph_key: str = "graph"
    """ The key for the graph input."""
    embedding_key: str = "embedding"
    """ The key for the embedding output."""
    species_encoding: dict = dataclasses.field(
        default_factory=lambda: {"encoding": "electronic_structure"}
    )
    """ The species encoding parameters. See `fennol.models.misc.encodings.SpeciesEncoding`. """
    radial_basis: dict = dataclasses.field(default_factory=lambda: {"basis": "spooky"})
    """ The radial basis parameters. See `fennol.models.misc.encodings.RadialBasis`. """
    kernel_init: Union[Callable, str] = "scaled_orthogonal(scale=1.0, mode='fan_avg')"
    """ The kernel initializer for Dense operations."""
    use_spin_encoding: bool = True
    """ Whether to use spin encoding."""
    use_charge_encoding: bool = True
    """ Whether to use charge encoding."""
    total_charge_key: str = "total_charge"
    """ The key for the total charge input."""

    FID: ClassVar[str] = "SPOOKYNET"

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

        eZ = (
            nn.Dense(
                self.dim,
                name="species_linear",
                use_bias=False,
                kernel_init=nn.initializers.zeros,
            )(onehot)
            + zrand
        )
        xi = eZ

        # encode charge information
        batch_index = inputs["batch_index"]
        natoms = inputs["natoms"]
        if self.use_charge_encoding and (
            self.total_charge_key in inputs or self.is_initializing()
        ):
            Q = inputs.get(self.total_charge_key, jnp.zeros(natoms.shape[0], dtype=xi.dtype))
            kq_pos, kq_neg, vq_pos, vq_neg = self.param(
                "kv_charge",
                lambda key, shape: jax.random.normal(key, shape, dtype=xi.dtype),
                (4, self.dim),
            )
            qi = nn.Dense(self.dim, kernel_init=kernel_init, name="q_linear")(eZ)
            pos_mask = Q >= 0
            kq = jnp.where(pos_mask[:,None], kq_pos[None, :], kq_neg[None, :])
            vq = jnp.where(pos_mask[:,None], vq_pos[None, :], vq_neg[None, :])
            qik = (qi * kq[batch_index]).sum(axis=-1) / self.dim**0.5
            wi =  jax.nn.softplus(qik)
            wnorm = jax.ops.segment_sum(wi, batch_index, Q.shape[0])
            avi = wi[:,None] * ((Q / wnorm)[:, None] * vq)[batch_index]
            eQ = ResMLP(use_bias=False, name="eQ", kernel_init=kernel_init)(avi)
            xi = xi + eQ

        # encode spin information
        if self.use_spin_encoding and (
            "total_spin" in inputs or self.is_initializing()
        ):
            S = inputs.get("total_spin", jnp.zeros(natoms.shape[0], dtype=xi.dtype))
            ks, vs = self.param(
                "kv_spin",
                lambda key, shape: jax.random.normal(key, shape, dtype=xi.dtype),
                (2, self.dim),
            )
            si = nn.Dense(self.dim, kernel_init=kernel_init, name="s_linear")(eZ)
            sik = (si * ks[None, :]).sum(axis=-1) / self.dim**0.5
            wi = jax.nn.softplus(sik)
            wnorm = jax.ops.segment_sum(wi, batch_index, S.shape[0])
            avi = wi[:,None] * ((S / wnorm)[:, None] * vs[None, :])[batch_index]
            eS = ResMLP(use_bias=False, name="eS", kernel_init=kernel_init)(avi)
            xi = xi + eS

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
        if self.use_charge_encoding and "total_charge" in inputs:
            output[self.embedding_key + "_eQ"] = eQ
        if self.use_spin_encoding and "total_spin" in inputs:
            output[self.embedding_key + "_eS"] = eS
        return output
