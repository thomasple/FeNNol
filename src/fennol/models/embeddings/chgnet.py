import jax
import jax.numpy as jnp
import flax.linen as nn
import dataclasses
import numpy as np
from typing import Any, Dict, Union, Callable, Sequence, Optional, ClassVar

from ...utils.initializers import initializer_from_str
from ..misc.nets import GatedPerceptron


class CHGNetEmbedding(nn.Module):
    """ Crystal Hamiltonian Graph Neural Network

    FID: CHGNET

    ### Reference
    Deng, B., Zhong, P., Jun, K. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.
        Nat Mach Intell 5, 1031–1041 (2023). https://doi.org/10.1038/s42256-023-00716-3
    
    """
    _graphs_properties: Dict
    dim: int = 64
    """The dimension of the embedding."""
    nmax_angle: int = 4
    """ The maximum fourier order for the angle representation."""
    nlayers: int = 3
    """The number of layers."""
    graph_key: str = "graph"
    """The key for the graph input."""
    graph_angle_key: Optional[str] = None
    """The key for the angular graph input."""
    embedding_key: str = "embedding"
    """The key for the embedding output."""
    species_encoding: dict = dataclasses.field(default_factory=dict)
    """The species encoding parameters. See `fennol.models.misc.encodings.SpeciesEncoding`"""
    radial_basis: dict = dataclasses.field(default_factory=dict)
    """The radial basis parameters. See `fennol.models.misc.encodings.RadialBasis`"""
    radial_basis_angle: Optional[dict] = None
    """The radial basis parameters for angle embedding. See `fennol.models.misc.encodings.RadialBasis`"""
    keep_all_layers: bool = False
    """Whether to keep all layers in the output."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initializer for Dense operations."""

    FID: ClassVar[str]  = "CHGNET"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        assert (
            len(species.shape) == 1
        ), "Species must be a 1D array (batches must be flattened)"

        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )

        ##################################################
        # Check that the graph_angle is a subgraph of graph
        graph = inputs[self.graph_key]
        graph_angle_key = (
            self.graph_angle_key if self.graph_angle_key is not None else self.graph_key
        )
        graph_angle = inputs[graph_angle_key]

        correct_graph = (
            graph_angle_key == self.graph_key
            or self._graphs_properties[graph_angle_key]["parent_graph"]
            == self.graph_key
        )
        assert (
            correct_graph
        ), f"graph_angle_key={graph_angle_key} must be a subgraph of graph_key={self.graph_key}"
        assert "angles" in graph_angle, f"Graph {graph_angle_key} must contain angles"
        # check if graph_angle is a filtered graph
        filtered = "parent_graph" in self._graphs_properties[graph_angle_key]
        if filtered:
            filter_indices = graph_angle["filter_indices"]

        ##################################################
        ### SPECIES ENCODING ###
        zi = SpeciesEncoding(**self.species_encoding, name="SpeciesEncoding")(species)

        vi = nn.Dense(self.dim, name="vi0", use_bias=True, kernel_init=kernel_init)(zi)

        ##################################################
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        distances = graph["distances"]
        switch = graph["switch"][:, None]

        ### COMPUTE RADIAL BASIS ###
        radial_basis = RadialBasis(
            **{
                **self.radial_basis,
                "end": cutoff,
                "name": f"RadialBasis",
            }
        )(distances)

        ##################################################
        ### GET ANGLES ###
        angles = graph_angle["angles"][:, None]
        angle_src, angle_dst = graph_angle["angle_src"], graph_angle["angle_dst"]
        switch_angles = graph_angle["switch"][:, None]
        central_atom = graph_angle["central_atom"]

        ### COMPUTE RADIAL BASIS FOR ANGLES ###
        if self.radial_basis_angle is not None:
            dangles = graph_angle["distances"]
            radial_basis_angle = (
                RadialBasis(
                    **{
                        **self.radial_basis_angle,
                        "end": self._graphs_properties[graph_angle_key]["cutoff"],
                        "name": f"RadialBasisAngle",
                    }
                )(dangles)
                * switch_angles
            )

        else:
            if filtered:
                radial_basis_angle = radial_basis[filter_indices] * switch_angles
            else:
                radial_basis_angle = radial_basis * switch

        radial_basis = radial_basis * switch

        eij, eija = jnp.split(
            nn.Dense(
                2 * self.dim, use_bias=False, kernel_init=kernel_init, name="eij0"
            )(radial_basis),
            2,
            axis=-1,
        )

        eijb = nn.Dense(self.dim, use_bias=False, kernel_init=kernel_init, name="eijb")(
            radial_basis_angle
        )
        eijkb = eijb[angle_src] * eijb[angle_dst]

        ##################################################
        ### ANGULAR BASIS ###
        # build fourier series for angles
        nangles = jnp.asarray(
            np.arange(self.nmax_angle + 1, dtype=distances.dtype)[None, :]
        )

        Ac = jnp.cos(nangles * angles)
        As = jnp.sin(nangles[:, 1:] * angles)
        aijk = nn.Dense(
            self.dim, use_bias=False, kernel_init=kernel_init, name="aijk0"
        )(jnp.concatenate([Ac, As], axis=-1))
        aikj = aijk

        ##################################################
        if self.keep_all_layers:
            vis = []

        ### LOOP OVER LAYERS ###
        for layer in range(self.nlayers):
            phiv = GatedPerceptron(
                self.dim,
                use_bias=True,
                kernel_init=kernel_init,
                activation=nn.silu,
                name=f"phiv{layer+1}",
            )(jnp.concatenate([vi[edge_src], vi[edge_dst], eij], axis=-1))

            vi = vi + nn.Dense(
                self.dim, use_bias=True, kernel_init=kernel_init, name=f"vi{layer+1}"
            )(jax.ops.segment_sum(phiv * eija, edge_src, vi.shape[0]))

            if self.keep_all_layers:
                vis.append(vi)
                
            if layer == self.nlayers - 1:
                break
                
            eij_ang = eij[filter_indices] if filtered else eij
            eij_angsrc = eij_ang[angle_src]
            eij_angdst = eij_ang[angle_dst]

            phie = GatedPerceptron(
                self.dim,
                use_bias=True,
                kernel_init=kernel_init,
                activation=nn.silu,
                name=f"phie{layer+1}",
            )

            vi_ang = vi[central_atom]
            phie_ijk = phie(
                jnp.concatenate(
                    [eij_angsrc, eij_angdst, aijk, vi_ang],
                    axis=-1,
                )
            )
            phie_ikj = phie(
                jnp.concatenate(
                    [eij_angdst, eij_angsrc, aikj, vi_ang],
                    axis=-1,
                )
            )

            eij_ang = eij_ang + nn.Dense(
                self.dim, use_bias=False, kernel_init=kernel_init, name=f"Le{layer+1}"
            )(
                jax.ops.segment_sum(phie_ikj * eijkb, angle_dst, eij_ang.shape[0])
                + jax.ops.segment_sum(phie_ijk * eijkb, angle_src, eij_ang.shape[0])
            )
            eij_angsrc = eij_ang[angle_src]
            eij_angdst = eij_ang[angle_dst]
            eij = eij.at[filter_indices].set(eij_ang)

            phia = GatedPerceptron(
                self.dim,
                use_bias=True,
                kernel_init=kernel_init,
                activation=nn.silu,
                name=f"phia{layer+1}",
            )
            aijk = aijk + phia(
                jnp.concatenate([eij_angsrc, eij_angdst, aijk, vi_ang], axis=-1)
            )
            aikj = aikj + phia(
                jnp.concatenate([eij_angdst, eij_angsrc, aikj, vi_ang], axis=-1)
            )

            

        output = {
            **inputs,
            self.embedding_key: vi,
        }
        if self.keep_all_layers:
            output[self.embedding_key + "_layers"] = jnp.stack(vis, axis=1)
        return output
