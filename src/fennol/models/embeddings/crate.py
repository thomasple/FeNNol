import jax
import jax.numpy as jnp
import flax.linen as nn
import dataclasses
import numpy as np
from typing import Dict, Union, Callable, Sequence, Optional, Tuple, ClassVar

from ..misc.encodings import SpeciesEncoding, RadialBasis, positional_encoding
from ...utils.spherical_harmonics import generate_spherical_harmonics, CG_SO3
from ...utils.activations import activation_from_str, tssr3
from ...utils.initializers import initializer_from_str
from ..misc.nets import FullyConnectedNet, ZAcNet
from ..misc.e3 import ChannelMixing, ChannelMixingE3, FilteredTensorProduct
from ...utils.periodic_table import D3_COV_RADII, D3_VDW_RADII, VALENCE_ELECTRONS
from ...utils import AtomicUnits as au


class CRATEmbedding(nn.Module):
    """Configurable Resources ATomic Environment

    FID : CRATE

    This class represents the CRATE (Configurable Resources ATomic Environment) embedding model.
    It is used to encode atomic environments using multiple sources of information
      (radial, angular, E(3), message-passing, LODE, etc...)
    """

    _graphs_properties: Dict

    dim: int = 256
    """The size of the embedding vectors."""
    nlayers: int = 2
    """The number of interaction layers in the model."""
    keep_all_layers: bool = False
    """Whether to output all layers."""

    dim_src: int = 64
    """The size of the source embedding vectors."""
    dim_dst: int = 32
    """The size of the destination embedding vectors."""

    angle_style: str = "fourier"
    """The style of angle representation."""
    dim_angle: int = 8
    """The size of the pairwise vectors use for triplet combinations."""
    nmax_angle: int = 4
    """The dimension of the angle representation (minus one)."""
    zeta: float = 14.1
    """The zeta parameter for the model ANI angular representation."""
    angle_combine_pairs: bool = True
    """Whether to combine angle pairs instead of average distance embedding like in ANI."""

    message_passing: bool = True
    """Whether to use message passing in the model."""
    att_dim: int = 1
    """The hidden size for the attention mechanism (only used when message-passing is disabled)."""

    lmax: int = 0
    """The maximum order of spherical tensors."""
    nchannels_l: int = 16
    """The number of channels for spherical tensors."""
    n_tp: int = 1
    """The number of tensor products performed at each layer."""
    ignore_irreps_parity: bool = False
    """Whether to ignore the parity of the irreps in the tensor product."""

    mixing_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [])
    """The hidden layer sizes for the mixing network."""
    pair_mixing_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [])
    """The hidden layer sizes for the pair mixing network."""
    activation: Union[Callable, str] = "silu"
    """The activation function for the mixing network."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization function for Dense operations."""
    activation_mixing: Union[Callable, str] = "tssr3"
    """The activation function applied after mixing."""
    use_zacnet: bool = False
    """Whether to use ZacNet. See `fennol.models.misc.nets.ZAcNet`."""

    graph_key: str = "graph"
    """The key for the graph data in the inputs dictionary."""
    graph_angle_key: Optional[str] = None
    """The key for the angle graph data in the inputs dictionary."""
    embedding_key: Optional[str] = None
    """The key for the embedding data in the output dictionary."""
    pair_embedding_key: Optional[str] = None
    """The key for the pair embedding data in the output dictionary."""

    species_encoding: dict = dataclasses.field(default_factory=dict)
    """The dictionary of parameters for species encoding. See `fennol.models.misc.encodings.SpeciesEncoding`."""
    radial_basis: dict = dataclasses.field(default_factory=dict)
    """The dictionary of parameters for radial basis functions. See `fennol.models.misc.encodings.RadialBasis`."""
    radial_basis_angle: Optional[dict] = None
    """The dictionary of parameters for radial basis functions for angle embedding. 
        If None, the radial basis for angles is the same as the radial basis for distances."""

    graph_lode: Optional[str] = None
    """The key for the lode graph data in the inputs dictionary."""
    lode_channels: int = 16
    """The number of channels for lode."""
    lode_hidden: Sequence[int] = dataclasses.field(default_factory=lambda: [])
    """The hidden layer sizes for the lode network."""
    lode_switch: float = 2.0
    """The switch parameter for lode."""
    lode_shift: float = 1.0
    """The shift parameter for lode."""

    charge_embedding: bool = False
    """Whether to include charge embedding."""
    total_charge_key: str = "total_charge"
    """The key for the total charge data in the inputs dictionary."""

    FID: ClassVar[str] = "CRATE"

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

        actmix = activation_from_str(self.activation_mixing)

        ##################################################
        graph = inputs[self.graph_key]
        use_angles = self.graph_angle_key is not None
        if use_angles:
            graph_angle = inputs[self.graph_angle_key]

            # Check that the graph_angle is a subgraph of graph
            correct_graph = (
                self.graph_angle_key == self.graph_key
                or self._graphs_properties[self.graph_angle_key]["parent_graph"]
                == self.graph_key
            )
            assert (
                correct_graph
            ), f"graph_angle_key={self.graph_angle_key} must be a subgraph of graph_key={self.graph_key}"
            assert (
                "angles" in graph_angle
            ), f"Graph {self.graph_angle_key} must contain angles"
            # check if graph_angle is a filtered graph
            filtered = "parent_graph" in self._graphs_properties[self.graph_angle_key]
            if filtered:
                filter_indices = graph_angle["filter_indices"]

        ##################################################
        ### SPECIES ENCODING ###
        species_encoder = SpeciesEncoding(
            **self.species_encoding, name="SpeciesEncoding"
        )
        zi = species_encoder(species)

        if self.charge_embedding:
            xi, qi = jnp.split(
                nn.Dense(self.dim + 1, use_bias=False, name="ChargeEncoding")(zi),
                [self.dim],
                axis=-1,
            )
            batch_index = inputs["batch_index"]
            natoms = inputs["natoms"]
            nsys = natoms.shape[0]
            Zi = jnp.asarray(VALENCE_ELECTRONS)[species]
            Ntot = jax.ops.segment_sum(Zi, batch_index, nsys) - inputs.get(
                self.total_charge_key, jnp.zeros(nsys)
            )
            ai = jax.nn.softplus(qi.squeeze(-1))
            A = jax.ops.segment_sum(ai, batch_index, nsys)
            Ni = ai * (Ntot / A)[batch_index]
            charge_embedding = positional_encoding(Ni, self.dim)
            xi = xi + charge_embedding
        else:
            xi = zi

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

        do_lode = self.graph_lode is not None
        if do_lode:
            graph_lode = inputs[self.graph_lode]
            rij_lr = graph_lode["distances"]/au.BOHR
            # unswitch = 0.5 - 0.5 * jnp.cos(
            #     (np.pi / self.lode_switch) * (rij_lr - cutoff + self.lode_shift)
            # )
            # unswitch = jnp.where(
            #     rij_lr > cutoff + self.lode_switch - self.lode_shift,
            #     1.0,
            #     jnp.where(rij_lr < cutoff - self.lode_shift, 0.0, unswitch),
            # )
            ai = jnp.abs(self.param("a_lode", lambda key: jnp.asarray(D3_VDW_RADII)))[
                species
            ]
            ai2 = ai**2
            gamma_ij = (
                ai2[graph_lode["edge_src"]] + ai2[graph_lode["edge_dst"]] + 1.0e-3
            ) ** (-0.5)
            qi, gi = jnp.split(
                FullyConnectedNet(
                    [*self.lode_hidden, 2 * self.lode_channels],
                    activation=self.activation,
                    name=f"lode_qi",
                )(zi),
                2,
                axis=-1,
            )
            gi = jax.nn.softplus(gi)
            unswitch = jax.scipy.special.erf(
                gi[graph_lode["edge_src"]] * (gamma_ij * rij_lr)[:, None]
            )
            Mij = (
                qi[graph_lode["edge_dst"]]
                * unswitch
                * (graph_lode["switch"] / rij_lr)[:, None]
            )
            Mi = jax.ops.segment_sum(Mij, graph_lode["edge_src"], species.shape[0])

        ##################################################
        ### GET ANGLES ###
        if use_angles:
            angles = graph_angle["angles"][:, None]
            angle_src, angle_dst = graph_angle["angle_src"], graph_angle["angle_dst"]
            switch_angles = graph_angle["switch"][:, None]
            central_atom = graph_angle["central_atom"]

            if not self.angle_combine_pairs:
                assert (
                    self.radial_basis_angle is not None
                ), "radial_basis_angle must be specified if angle_combine_pairs=False"

            ### COMPUTE RADIAL BASIS FOR ANGLES ###
            if self.radial_basis_angle is not None:
                dangles = graph_angle["distances"]
                swang = switch_angles
                if not self.angle_combine_pairs:
                    dangles = 0.5 * (dangles[angle_src] + dangles[angle_dst])
                    swang = switch_angles[angle_src] * switch_angles[angle_dst]
                radial_basis_angle = (
                    RadialBasis(
                        **{
                            **self.radial_basis_angle,
                            "end": self._graphs_properties[self.graph_angle_key][
                                "cutoff"
                            ],
                            "name": f"RadialBasisAngle",
                        }
                    )(dangles)
                    * swang
                )

            else:
                if filtered:
                    radial_basis_angle = radial_basis[filter_indices] * switch_angles
                else:
                    radial_basis_angle = radial_basis * switch

        radial_basis = radial_basis * switch

        # # add covalent indicator
        # rc = jnp.asarray([d/au.BOHR for d in D3_COV_RADII])[species]
        # rcij = rc[edge_src] + rc[edge_dst]
        # fact = graph["switch"]*(2*distances/rcij)*jnp.exp(-0.5 * ((distances - rcij)/(0.1*rcij)) ** 2)
        # radial_basis = jnp.concatenate([radial_basis,fact[:,None]],axis=-1)
        # if use_angles:
        #     rcij = rc[graph_angle["edge_src"]] + rc[graph_angle["edge_dst"]]
        #     dangles = graph_angle["distances"]
        #     fact = graph_angle["switch"]*((2*dangles/rcij))*jnp.exp(-0.5 * ((dangles - rcij)/(0.1*rcij))**2)
        #     radial_basis_angle = jnp.concatenate([radial_basis_angle,fact[:,None]],axis=-1)


        ##################################################
        if self.lmax > 0:
            Yij = generate_spherical_harmonics(lmax=self.lmax, normalize=False)(
                graph["vec"] / graph["distances"][:, None]
            )[:, None, :]
            #ls = [0]
            #for l in range(1, self.lmax + 1):
            #    ls = ls + [l] * (2 * l + 1)
            #ls = jnp.asarray(np.array(ls)[None, :], dtype=distances.dtype)
            #lcut = (0.5 + 0.5 * jnp.cos((np.pi / cutoff) * distances[:, #None])) ** (
            #    ls + 1
            #)
            # lcut = jnp.where(graph["edge_mask"][:, None], lcut, 0.0)
            # rijl1 = (lcut * distances[:, None] ** ls)[:, None, :]

        ##################################################
        if use_angles:
            ### ANGULAR BASIS ###
            if self.angle_style == "fourier":
                # build fourier series for angles
                nangles = self.param(
                    f"nangles",
                    lambda key, dim: jnp.arange(dim, dtype=distances.dtype)[None, :],
                    self.nmax_angle + 1,
                )

                phi = self.param(
                    f"phi",
                    lambda key, dim: jnp.zeros((1, dim), dtype=distances.dtype),
                    self.nmax_angle + 1,
                )
                xa = jnp.cos(nangles * angles + phi)
            elif self.angle_style == "fourier_full":
                # build fourier series for angles including sin terms
                nangles = self.param(
                    f"nangles",
                    lambda key, dim: jnp.arange(dim, dtype=distances.dtype)[None, :],
                    self.nmax_angle + 1,
                )

                phi = self.param(
                    f"phi",
                    lambda key, dim: jnp.zeros((1, dim), dtype=distances.dtype),
                    2 * self.nmax_angle + 1,
                )
                xac = jnp.cos(nangles * angles + phi[:, : self.nmax_angle + 1])
                xas = jnp.sin(nangles[:, 1:] * angles + phi[:, self.nmax_angle + 1 :])
                xa = jnp.concatenate([xac, xas], axis=-1)
            elif self.angle_style == "ani":
                # ANI-style angle embedding
                angle_start = np.pi / (2 * (self.nmax_angle + 1))
                shiftZ = self.param(
                    f"shiftZ",
                    lambda key, dim: jnp.asarray(
                        (np.linspace(0, np.pi, dim + 1) + angle_start)[None, :-1],
                        dtype=distances.dtype,
                    ),
                    self.nmax_angle + 1,
                )
                zeta = self.param(
                    f"zeta",
                    lambda key: jnp.asarray(self.zeta, dtype=distances.dtype),
                )
                xa = (0.5 + 0.5 * jnp.cos(angles - shiftZ)) ** zeta
            else:
                raise ValueError(f"Unknown angle style {self.angle_style}")
            xa = xa[:, None, :]
            if not self.angle_combine_pairs:
                xa = (xa * radial_basis_angle[:, :, None]).reshape(
                    -1, 1, xa.shape[1] * radial_basis_angle.shape[1]
                )

            if self.pair_embedding_key is not None:
                if filtered:
                    ang_pair_src = filter_indices[angle_src]
                    ang_pair_dst = filter_indices[angle_dst]
                else:
                    ang_pair_src = angle_src
                    ang_pair_dst = angle_dst
                ang_pairs = jnp.concatenate((ang_pair_src, ang_pair_dst))

        ##################################################
        ### DIMENSIONS ###
        dim_src = (
            [self.dim_src] * self.nlayers
            if isinstance(self.dim_src, int)
            else self.dim_src
        )
        assert (
            len(dim_src) == self.nlayers
        ), f"dim_src must be an integer or a list of length {self.nlayers}"
        dim_dst = self.dim_dst
        # dim_dst = (
        #     [self.dim_dst] * self.nlayers
        #     if isinstance(self.dim_dst, int)
        #     else self.dim_dst
        # )
        # assert (
        #     len(dim_dst) == self.nlayers
        # ), f"dim_dst must be an integer or a list of length {self.nlayers}"

        if use_angles:
            dim_angle = (
                [self.dim_angle] * self.nlayers
                if isinstance(self.dim_angle, int)
                else self.dim_angle
            )
            assert (
                len(dim_angle) == self.nlayers
            ), f"dim_angle must be an integer or a list of length {self.nlayers}"
            # nmax_angle = [self.nmax_angle]*self.nlayers if isinstance(self.nmax_angle, int) else self.nmax_angle
            # assert len(nmax_angle) == self.nlayers, f"nmax_angle must be an integer or a list of length {self.nlayers}"

        message_passing = (
            [self.message_passing] * self.nlayers
            if isinstance(self.message_passing, bool)
            else self.message_passing
        )
        assert (
            len(message_passing) == self.nlayers
        ), f"message_passing must be a boolean or a list of length {self.nlayers}"

        ##################################################
        ### INITIALIZE PAIR EMBEDDING ###
        if self.pair_embedding_key is not None:
            xij_s,xij_d = jnp.split(nn.Dense(2*dim_dst, name="pair_init_linear")(zi), [dim_dst], axis=-1)
            xij = xij_s[edge_src]*xij_d[edge_dst]

        ##################################################
        if self.keep_all_layers:
            xis = []

        ### LOOP OVER LAYERS ###
        for layer in range(self.nlayers):
            ##################################################
            ### COMPACT DESCRIPTORS ###
            si, si_dst = jnp.split(
                nn.Dense(
                    dim_src[layer] + dim_dst,
                    name=f"species_linear_{layer}",
                    use_bias=True,
                )(xi),
                [
                    dim_src[layer],
                ],
                axis=-1,
            )

            ##################################################
            if message_passing[layer] or layer == 0:
                ### MESSAGE PASSING ###
                si_mp = si_dst[edge_dst]
            else:
                # if layer == 0:
                #     si_mp = si_dst[edge_dst]
                ### ATTENTION TO SIMULATE MP ###
                Q = nn.Dense(
                    dim_dst * self.att_dim, name=f"queries_{layer}", use_bias=False
                )(si_dst).reshape(-1, dim_dst, self.att_dim)[edge_src]
                K = nn.Dense(
                    dim_dst * self.att_dim, name=f"keys_{layer}", use_bias=False
                )(zi).reshape(-1, dim_dst, self.att_dim)[edge_dst]

                si_mp = (K * Q).sum(axis=-1) / self.att_dim**0.5
                # Vmp = jax.ops.segment_sum(
                #     (KQ * switch)[:, :, None] * Yij, edge_src, species.shape[0]
                # )
                # si_mp = (Vmp[edge_src] * Yij).sum(axis=-1)
                # Q = nn.Dense(
                #     dim_dst * dim_dst, name=f"queries_{layer}", use_bias=False
                # )(si_dst).reshape(-1, dim_dst, dim_dst)
                # si_mp = (
                #     si_mp + jax.vmap(jnp.dot)(Q[edge_src], si_mp) / self.dim_dst**0.5
                # )

            if self.pair_embedding_key is not None:
                si_mp = si_mp + xij

            ##################################################
            ### PAIR EMBEDDING ###
            Lij = (si_mp[:, None, :] * radial_basis[:, :, None]).reshape(
                radial_basis.shape[0], si_mp.shape[1] * radial_basis.shape[1]
            )

            ### AGGREGATE PAIR EMBEDDING ###
            Li = jax.ops.segment_sum(Lij, edge_src, species.shape[0])

            ##################################################
            ### EQUIVARIANT EMBEDDING ###
            if self.lmax > 0:
                if layer == 0 or not message_passing[layer]:
                    Vij = Yij
                else:
                    Vij = Vi[edge_dst]
                drhoi = jax.ops.segment_sum(
                    nn.Dense(
                        self.nchannels_l, name=f"e3_channel_{layer}", use_bias=False
                    )(Lij)[:, :, None]
                    * Vij,
                    edge_src,
                    species.shape[0],
                )

                if layer == 0:
                    rhoi = drhoi
                    Vi = ChannelMixingE3(
                        self.lmax,
                        self.nchannels_l,
                        self.nchannels_l,
                        name=f"e3_initial_mixing_{layer}",
                    )(rhoi)
                else:
                    rhoi = rhoi + drhoi
                Vi0 = []
                for itp in range(self.n_tp):
                    dVi = FilteredTensorProduct(
                        self.lmax, self.lmax, name=f"tensor_product_{layer}_{itp}",ignore_parity=self.ignore_irreps_parity,weights_by_channel=False
                    )(rhoi, Vi)
                    Vi = ChannelMixing(
                            self.lmax,
                            self.nchannels_l,
                            self.nchannels_l,
                            name=f"tp_mixing_{layer}_{itp}",
                        )(Vi + dVi)
                    Vi0.append(dVi[:, :, 0])
                Vi0 = jnp.concatenate(Vi0, axis=-1)
                if self.pair_embedding_key is not None:
                    Vij = Vi[edge_src]*Yij
                    Vij0 = [Vij[...,0]]
                    for l in range(1,self.lmax+1):
                        Vij0.append(Vij[...,l**2:(l+1)**2].sum(axis=-1))
                    Vij0 = jnp.concatenate(Vij0,axis=-1)


            ##################################################
            ### ANGLE EMBEDDING ###
            if use_angles:
                si_mp_ang = si_mp[filter_indices] if filtered else si_mp
                if self.angle_combine_pairs:
                    Wa = self.param(
                        f"Wa_{layer}",
                        nn.initializers.normal(
                            stddev=1.0
                            / (si_mp.shape[1] * radial_basis_angle.shape[1]) ** 0.5
                        ),
                        (si_mp.shape[1], radial_basis_angle.shape[1], dim_angle[layer]),
                    )
                    Da = jnp.einsum(
                        "...i,...j,ijk->...k",
                        si_mp_ang,
                        radial_basis_angle,
                        Wa,
                    )

                else:
                    if message_passing[layer] or layer == 0:
                        Da = nn.Dense(dim_angle[layer], name=f"angle_linear_{layer}")(
                            xi
                        )[graph_angle["edge_dst"]]
                    else:
                        Da = nn.Dense(dim_angle[layer], name=f"angle_linear_{layer}")(
                            si_mp_ang
                        )

                Da = Da[:, :, None]
                # combine pair and angle info
                radang = (xa * (Da[angle_dst] * Da[angle_src])).reshape(
                    (-1, Da.shape[1] * xa.shape[2])
                )

                ### AGGREGATE  ANGLE EMBEDDING ###
                ang_embedding = jax.ops.segment_sum(
                    radang, central_atom, species.shape[0]
                )

                if self.pair_embedding_key is not None:
                    ang_ij = jax.ops.segment_sum(
                        jnp.concatenate((radang, radang)),
                        ang_pairs,
                        edge_src.shape[0],
                    )

            ##################################################
            ### CONCATENATE EMBEDDING COMPONENTS ###
            components = [si, Li]
            if use_angles:
                components.append(ang_embedding)
            if self.lmax > 0:
                components.append(Vi0)
            if do_lode:
                components.append(Mi)

            dxi = jnp.concatenate(components, axis=-1)

            ##################################################
            ### CONCATENATE PAIR EMBEDDING COMPONENTS ###
            if self.pair_embedding_key is not None:
                components = [si[edge_src], xij, Lij]
                if use_angles:
                    components.append(ang_ij)
                if self.lmax > 0:
                    components.append(Vij0)
                dxij = jnp.concatenate(components, axis=-1)

            ##################################################
            ### MIX AND APPLY NONLINEARITY ###
            if self.use_zacnet:
                dxi = actmix(
                    ZAcNet(
                        [*self.mixing_hidden, self.dim],
                        zmax=species_encoder.zmax,
                        activation=self.activation,
                        name=f"dxi_{layer}",
                        use_bias=True,
                        kernel_init=kernel_init,
                    )((species, dxi))
                )
            else:
                dxi = actmix(
                    FullyConnectedNet(
                        [*self.mixing_hidden, self.dim],
                        activation=self.activation,
                        name=f"dxi_{layer}",
                        use_bias=True,
                        kernel_init=kernel_init,
                    )(dxi)
                )

            if self.pair_embedding_key is not None:
                ### UPDATE PAIR EMBEDDING ###
                # dxij = tssr3(nn.Dense(dim_dst, name=f"dxij_{layer}",use_bias=False)(dxij))
                dxij = actmix(
                    FullyConnectedNet(
                        [*self.pair_mixing_hidden, dim_dst],
                        activation=self.activation,
                        name=f"dxij_{layer}",
                        use_bias=False,
                        kernel_init=kernel_init,
                    )(dxij)
                )
                xij = xij + dxij

            ##################################################
            ### UPDATE EMBEDDING ###
            if layer == 0:
                xi = dxi
            else:
                ### FORGET GATE ###
                R = jax.nn.sigmoid(
                    self.param(
                        f"retention_{layer}",
                        nn.initializers.normal(),
                        (xi.shape[-1],),
                    )
                )
                xi = R[None, :] * xi + dxi

            if self.keep_all_layers:
                xis.append(xi)

        embedding_key = (
            self.embedding_key if self.embedding_key is not None else self.name
        )
        output = {
            **inputs,
            embedding_key: xi,
        }
        if self.lmax > 0:
            output[embedding_key + "_tensor"] = Vi
        if self.keep_all_layers:
            output[embedding_key + "_layers"] = jnp.stack(xis, axis=1)
        if self.charge_embedding:
            output[embedding_key + "_charge"] = charge_embedding
        if self.pair_embedding_key is not None:
            output[self.pair_embedding_key] = xij
        return output
