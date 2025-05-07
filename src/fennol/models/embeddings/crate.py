import jax
import jax.numpy as jnp
import flax.linen as nn
import dataclasses
import numpy as np
from typing import Dict, Union, Callable, Sequence, Optional, Tuple, ClassVar

from ..misc.encodings import SpeciesEncoding, RadialBasis, positional_encoding
from ...utils.spherical_harmonics import generate_spherical_harmonics, CG_SO3
from ...utils.activations import activation_from_str
from ...utils.initializers import initializer_from_str
from ..misc.nets import FullyConnectedNet, BlockIndexNet
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
    edge_tp: bool = False
    """Whether to perform a tensor product on edges before sending messages."""
    resolve_wij_l: bool = False
    """Equivariant message weights are l-dependent."""

    species_init: bool = False
    """Whether to initialize the embedding using the species encoding."""
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
    layer_normalization: bool = False
    """Whether to apply layer normalization after each layer."""
    use_bias: bool = True
    """Whether to use bias in the Dense operations."""

    graph_key: str = "graph"
    """The key for the graph data in the inputs dictionary."""
    graph_angle_key: Optional[str] = None
    """The key for the angle graph data in the inputs dictionary."""
    embedding_key: Optional[str] = None
    """The key for the embedding data in the output dictionary."""
    pair_embedding_key: Optional[str] = None
    """The key for the pair embedding data in the output dictionary."""

    species_encoding: Union[dict, str] = dataclasses.field(default_factory=dict)
    """If `str`, it is the key in the inputs dictionary that contains species encodings. Else, it is the dictionary of parameters for species encoding. See `fennol.models.misc.encodings.SpeciesEncoding`."""
    radial_basis: dict = dataclasses.field(default_factory=dict)
    """The dictionary of parameters for radial basis functions. See `fennol.models.misc.encodings.RadialBasis`."""
    radial_basis_angle: Optional[dict] = None
    """The dictionary of parameters for radial basis functions for angle embedding. 
        If None, the radial basis for angles is the same as the radial basis for distances."""

    graph_lode: Optional[str] = None
    """The key for the lode graph data in the inputs dictionary."""
    lode_channels: Union[int, Sequence[int]] = 8
    """The number of channels for lode."""
    lmax_lode: int = 0
    """The maximum order of spherical tensors for lode."""
    a_lode: float = -1.
    """The cutoff for the lode graph. If negative, the value is trainable with starting value -a_lode."""
    lode_resolve_l: bool = True
    """Whether to resolve the lode channels by l."""
    lode_multipole_interaction: bool = True
    """Whether to interact with the multipole moments of the lode graph."""
    lode_direct_multipoles: bool = True
    """Whether to directly use the first local equivariants to interact with long-range equivariants. If false, local equivariants are mixed before interaction."""
    lode_equi_full_combine: bool = False
    lode_normalize_l: bool = False
    lode_use_field_norm: bool = True
    lode_rshort: Optional[float] = None
    lode_dshort: float = 0.5
    lode_extra_powers: Sequence[int] = ()
    

    charge_embedding: bool = False
    """Whether to include charge embedding."""
    total_charge_key: str = "total_charge"
    """The key for the total charge data in the inputs dictionary."""

    block_index_key: Optional[str] = None
    """The key for the block index. If provided, will use a BLOCK_INDEX_NET as a mixing network."""

    FID: ClassVar[str] = "CRATE"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        assert (
            len(species.shape) == 1
        ), "Species must be a 1D array (batches must be flattened)"
        reduce_memory =  "reduce_memory" in inputs.get("flags", {})

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
        if isinstance(self.species_encoding, str):
            zi = inputs[self.species_encoding]
        else:
            zi = SpeciesEncoding(
                **self.species_encoding, name="SpeciesEncoding"
            )(species)

        
        if self.layer_normalization:
            def layer_norm(x):
                mu = jnp.mean(x,axis=-1,keepdims=True)
                dx = x-mu
                var = jnp.mean(dx**2,axis=-1,keepdims=True)
                sig = (1.e-6 + var)**(-0.5)
                return dx*sig
        else:
            layer_norm = lambda x:x
                

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
            xi = layer_norm(xi + charge_embedding)
        elif self.species_init:
            xi = layer_norm(nn.Dense(self.dim, use_bias=False, name="SpeciesInit")(zi))
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
            switch_lode = graph_lode["switch"][:, None]

            edge_src_lr, edge_dst_lr = graph_lode["edge_src"], graph_lode["edge_dst"]
            r = graph_lode["distances"][:, None]
            rc = self._graphs_properties[self.graph_lode]["cutoff"]
            
            lmax_lr = self.lmax_lode
            equivariant_lode = lmax_lr > 0
            assert lmax_lr >=0, f"lmax_lode must be >= 0, got {lmax_lr}"
            if self.lode_multipole_interaction:
                assert lmax_lr <= self.lmax, f"lmax_lode must be <= lmax for multipole interaction, got {lmax_lr} > {self.lmax}"
            nrep_lr = np.array([2 * l + 1 for l in range(lmax_lr + 1)], dtype=np.int32)
            if self.lode_resolve_l and equivariant_lode:
                ls_lr = np.arange(lmax_lr + 1)
            else:
                ls_lr = np.array([0])

            nextra_powers = len(self.lode_extra_powers)
            if nextra_powers > 0:
                ls_lr = np.concatenate([self.lode_extra_powers,ls_lr])

            if self.a_lode > 0:
                a = self.a_lode**2
            else:
                a = (
                    self.param(
                        "a_lr",
                        lambda key: jnp.asarray([-self.a_lode] * ls_lr.shape[0])[None, :],
                    )
                    ** 2
                )
            rc2a = rc**2 + a
            ls_lr = 0.5 * (ls_lr[None, :] + 1)
            ### minimal radial basis for long range (damped coulomb)
            eij_lr = (
                1.0 / (r**2 + a) ** ls_lr
                - 1.0 / rc2a**ls_lr
                + (r - rc) * rc * (2 * ls_lr) / rc2a ** (ls_lr + 1)
            ) * switch_lode

            if self.lode_rshort is not None:
                rs = self.lode_rshort
                d = self.lode_dshort
                switch_short = 0.5 * (1 - jnp.cos(jnp.pi * (r - rs) / d)) * (r > rs) * (
                    r < rs + d
                ) + (r >= rs + d)
                eij_lr = eij_lr * switch_short
            
            if nextra_powers>0:
                eij_lr_extra = eij_lr[:,:nextra_powers]
                eij_lr = eij_lr[:,nextra_powers:]


            # dim_lr = self.nchannels_lode
            nchannels_lode = (
                [self.lode_channels] * self.nlayers
                if isinstance(self.lode_channels, int)
                else self.lode_channels
            )
            dim_lr = nchannels_lode
            
            if equivariant_lode:
                if self.lode_resolve_l:
                    eij_lr = eij_lr.repeat(nrep_lr, axis=-1)
                Yij = generate_spherical_harmonics(lmax=lmax_lr, normalize=False)(
                    graph_lode["vec"] / r
                )
                eij_lr = (eij_lr * Yij)[:, None, :]
                dim_lr = [d * (lmax_lr + 1) for d in dim_lr]
            
            if nextra_powers > 0:
                eij_lr_extra = eij_lr_extra[:,None,:]
                extra_dims = [nextra_powers*d for d in nchannels_lode]
                dim_lr = [d + ed for d,ed in zip(dim_lr,extra_dims)]
            

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
            Yij = jnp.broadcast_to(Yij, (Yij.shape[0], self.nchannels_l, Yij.shape[2]))
            nrep_l = np.array([2 * l + 1 for l in range(self.lmax + 1)], dtype=np.int32)
            # ls = [0]
            # for l in range(1, self.lmax + 1):
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
                if reduce_memory: raise NotImplementedError("Angle embedding not implemented with reduce_memory")
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
        
        initialize_e3 = True
        if self.lmax > 0:
            n_tp = (
                [self.n_tp] * self.nlayers
                if isinstance(self.n_tp, int)
                else self.n_tp
            )
            assert (
                len(n_tp) == self.nlayers
            ), f"n_tp must be an integer or a list of length {self.nlayers}"


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
            xij = layer_norm(xij_s[edge_src]*xij_d[edge_dst])

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
                    use_bias=self.use_bias,
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
            if reduce_memory:
                Li = jnp.zeros((species.shape[0]* radial_basis.shape[1],si_mp.shape[1]),dtype=si_mp.dtype)
                for i in range(radial_basis.shape[1]):
                    indices = i + edge_src*radial_basis.shape[1]
                    Li = Li.at[indices].add(si_mp*radial_basis[:,i,None])
                Li = Li.reshape(species.shape[0], radial_basis.shape[1]*si_mp.shape[1])
            else:
                Lij = (si_mp[:, None, :] * radial_basis[:, :, None]).reshape(
                    radial_basis.shape[0], si_mp.shape[1] * radial_basis.shape[1]
                )
                ### AGGREGATE PAIR EMBEDDING ###
                Li = jax.ops.segment_sum(Lij, edge_src, species.shape[0])

            ### CONCATENATE EMBEDDING COMPONENTS ###
            components = [si, Li]
            if self.pair_embedding_key is not None:
                if reduce_memory: raise NotImplementedError("Pair embedding not implemented with reduce_memory")
                components_pair = [si[edge_src], xij, Lij]


            ##################################################
            ### ANGLE EMBEDDING ###
            if use_angles and dim_angle[layer]>0:
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

                Da = Da[angle_dst] * Da[angle_src]
                ## combine pair and angle info
                if reduce_memory:
                    ang_embedding = jnp.zeros((species.shape[0]* Da.shape[-1],xa.shape[-1]),dtype=Da.dtype)
                    for i in range(Da.shape[-1]):
                        indices = i + central_atom*Da.shape[-1]
                        ang_embedding = ang_embedding.at[indices].add(Da[:,i,None]*xa[:,0,:])
                    ang_embedding = ang_embedding.reshape(species.shape[0], xa.shape[-1]*Da.shape[-1])
                else:
                    radang = (xa * Da[:, :, None]).reshape(
                        (-1, Da.shape[1] * xa.shape[2])
                    )
                    ### AGGREGATE  ANGLE EMBEDDING ###
                    ang_embedding = jax.ops.segment_sum(
                        radang, central_atom, species.shape[0]
                    )
                    

                components.append(ang_embedding)

                if self.pair_embedding_key is not None:
                    ang_ij = jax.ops.segment_sum(
                        jnp.concatenate((radang, radang)),
                        ang_pairs,
                        edge_src.shape[0],
                    )
                    components_pair.append(ang_ij)
            
            ##################################################
            ### EQUIVARIANT EMBEDDING ###
            if self.lmax > 0 and n_tp[layer] >= 0:
                if initialize_e3 or not message_passing[layer]:
                    Vij = Yij
                elif self.edge_tp:
                    Vij = FilteredTensorProduct(
                            self.lmax, self.lmax, name=f"edge_tp_{layer}",ignore_parity=self.ignore_irreps_parity,weights_by_channel=False
                        )(Vi[edge_dst], Yij)
                else:
                    Vij = Vi[edge_dst]

                ### compute channel weights
                dim_wij = self.nchannels_l
                if self.resolve_wij_l:
                    dim_wij=self.nchannels_l*(self.lmax+1)

                eij = Lij if self.pair_embedding_key is None else jnp.concatenate([Lij,xij*switch],axis=-1)
                wij = nn.Dense(
                        dim_wij, name=f"e3_channel_{layer}", use_bias=False
                    )(eij)
                if self.resolve_wij_l:
                    wij = jnp.repeat(wij.reshape(-1,self.nchannels_l,self.lmax+1),nrep_l,axis=-1)
                else:
                    wij = wij[:,:,None]
                
                ### aggregate equivariant messages
                drhoi = jax.ops.segment_sum(
                    wij * Vij,
                    edge_src,
                    species.shape[0],
                )

                Vi0 = []
                if initialize_e3:
                    rhoi = drhoi
                    Vi = ChannelMixingE3(
                        self.lmax,
                        self.nchannels_l,
                        self.nchannels_l,
                        name=f"e3_initial_mixing_{layer}",
                    )(rhoi)
                    # assert n_tp[layer] > 0, "n_tp must be > 0 for the first equivariant layer."
                else:
                    rhoi = rhoi + drhoi
                    # if message_passing[layer]:
                        # Vi0.append(drhoi[:, :, 0])
                initialize_e3 = False
                if n_tp[layer] > 0:
                    for itp in range(n_tp[layer]):
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
                    components.append(Vi0)

                if self.pair_embedding_key is not None:
                    Vij = Vi[edge_src]*Yij
                    Vij0 = [Vij[...,0]]
                    for l in range(1,self.lmax+1):
                        Vij0.append(Vij[...,l**2:(l+1)**2].sum(axis=-1))
                    Vij0 = jnp.concatenate(Vij0,axis=-1)
                    components_pair.append(Vij0)

            ##################################################
            ### CONCATENATE EMBEDDING COMPONENTS ###
            if do_lode and nchannels_lode[layer] > 0:
                zj = nn.Dense(dim_lr[layer], use_bias=False, name=f"LODE_{layer}")(xi)
                if nextra_powers > 0:
                    zj_extra = zj[:,:nextra_powers*nchannels_lode[layer]].reshape(
                        (species.shape[0],nchannels_lode[layer], nextra_powers)
                    )
                    zj = zj[:,nextra_powers*nchannels_lode[layer]:]
                    xi_lr_extra = jax.ops.segment_sum(
                        eij_lr_extra * zj_extra[edge_dst_lr], edge_src_lr, species.shape[0]
                    )
                    components.append(xi_lr_extra.reshape(species.shape[0],-1))

                if equivariant_lode:
                    zj = zj.reshape(
                        (species.shape[0], nchannels_lode[layer], lmax_lr + 1)
                    ).repeat(nrep_lr, axis=-1)
                xi_lr = jax.ops.segment_sum(
                    eij_lr * zj[edge_dst_lr], edge_src_lr, species.shape[0]
                )
                if equivariant_lode:
                    assert self.lode_use_field_norm or self.lode_multipole_interaction, "equivariant LODE requires field norm or multipole interaction"
                    if self.lode_multipole_interaction:
                        if initialize_e3:
                            raise ValueError("equivariant LODE used before local equivariants initialized")
                        size_l_lr = (lmax_lr+1)**2
                        if self.lode_direct_multipoles:
                            assert nchannels_lode[layer] <= self.nchannels_l
                            Mi = Vi[:, : nchannels_lode[layer], :size_l_lr]
                        else:
                            Mi = ChannelMixingE3(
                                lmax_lr,
                                self.nchannels_l,
                                nchannels_lode[layer],
                                name=f"e3_LODE_{layer}",
                            )(Vi[...,:size_l_lr])
                        Mi_lr = Mi * xi_lr
                    components.append(xi_lr[:, :, 0])
                    if self.lode_use_field_norm and self.lode_equi_full_combine:
                        xi_lr1 = ChannelMixing(
                            lmax_lr,
                            nchannels_lode[layer],
                            nchannels_lode[layer],
                            name=f"LODE_mixing_{layer}",
                        )(xi_lr)
                    norm = 1.
                    for l in range(1, lmax_lr + 1):
                        if self.lode_normalize_l:
                            norm = 1. / (2 * l + 1)
                        if self.lode_multipole_interaction:
                            components.append(Mi_lr[:, :, l**2 : (l + 1) ** 2].sum(axis=-1)*norm)

                        if self.lode_use_field_norm:
                            if self.lode_equi_full_combine:
                                components.append((xi_lr[:,:,l**2 : (l + 1) ** 2]*xi_lr1[:,:,l**2 : (l + 1) ** 2]).sum(axis=-1)*norm)
                            else:
                                components.append(
                                    ((xi_lr[:, :, l**2 : (l + 1) ** 2]) ** 2).sum(axis=-1)*norm
                                )
                else:
                    components.append(xi_lr)

            dxi = jnp.concatenate(components, axis=-1)

            ##################################################
            ### CONCATENATE PAIR EMBEDDING COMPONENTS ###
            if self.pair_embedding_key is not None:
                dxij = jnp.concatenate(components_pair, axis=-1)

            ##################################################
            ### MIX AND APPLY NONLINEARITY ###
            if self.block_index_key is not None:
                block_index = inputs[self.block_index_key]
                dxi = actmix(BlockIndexNet(
                        output_dim=self.dim,
                        hidden_neurons=self.mixing_hidden,
                        activation=self.activation,
                        name=f"dxi_{layer}",
                        use_bias=self.use_bias,
                        kernel_init=kernel_init,
                    )((species,dxi, block_index))
                )
            else:
                dxi = actmix(
                    FullyConnectedNet(
                        [*self.mixing_hidden, self.dim],
                        activation=self.activation,
                        name=f"dxi_{layer}",
                        use_bias=self.use_bias,
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
                xij = layer_norm(xij + dxij)

            ##################################################
            ### UPDATE EMBEDDING ###
            if layer == 0 and not (self.species_init or self.charge_embedding):
                xi = layer_norm(dxi)
            else:
                ### FORGET GATE ###
                R = jax.nn.sigmoid(
                    self.param(
                        f"retention_{layer}",
                        nn.initializers.normal(),
                        (xi.shape[-1],),
                    )
                )
                xi = layer_norm(R[None, :] * xi + dxi)

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
