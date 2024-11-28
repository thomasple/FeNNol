import functools
import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict, Union, ClassVar, Optional, Set
import dataclasses

from ..misc.encodings import RadialBasis
from ...utils.activations import activation_from_str


try:
    import e3nn_jax as e3nn

    E3NN_AVAILABLE = True
    E3NN_EXCEPTION = None
    Irreps = e3nn.Irreps
    Irrep = e3nn.Irrep
except Exception as e:
    E3NN_AVAILABLE = False
    E3NN_EXCEPTION = e
    e3nn = None

    class Irreps(tuple):
        pass

    class Irrep(tuple):
        pass


class MACE(nn.Module):
    """MACE equivariant message passing neural network.

    adapted from MACE-jax github repo by M. Geiger and I. Batatia
    
    T. Plé reordered some operations and changed defaults to match the recent mace-torch version 
    -> compatibility with pretrained torch models requires some work on the parameters:
        - normalization of activation functions in e3nn differ between jax and pytorch => need rescaling
        - multiplicity ordering and signs of U matrices in SymmetricContraction differ => need to reorder and flip signs in the weight tensors
        - we use a maximum Z instead of a list of species => need to adapt species-dependent parameters
    
    References:
        - I. Batatia et al. "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields." Advances in Neural Information Processing Systems 35 (2022): 11423-11436.
        https://doi.org/10.48550/arXiv.2206.07697
        - I. Batatia et al. "The design space of e(3)-equivariant atom-centered interatomic potentials." arXiv preprint arXiv:2205.06643 (2022).
        https://doi.org/10.48550/arXiv.2205.06643

    """
    _graphs_properties: Dict
    output_irreps: Union[Irreps, str] = "1x0e"
    """The output irreps of the model."""
    hidden_irreps: Union[Irreps, str] = "128x0e + 128x1o"
    """The hidden irreps of the model."""
    readout_mlp_irreps: Union[Irreps, str] = "16x0e"
    """The hidden irreps of the readout MLP."""
    graph_key: str = "graph"
    """The key in the input dictionary that corresponds to the molecular graph to use."""
    output_key: Optional[str] = None
    """The key of the embedding in the output dictionary."""
    avg_num_neighbors: float = 1.0
    """The expected average number of neighbors."""
    ninteractions: int = 2
    """The number of interaction layers."""
    num_features: Optional[int] = None
    """The number of features per node. default gcd of hidden_irreps multiplicities"""
    radial_basis: dict = dataclasses.field(
        default_factory=lambda: {"basis": "bessel", "dim": 8, "trainable": False}
    )
    """The dictionary of parameters for radial basis functions. See `fennol.models.misc.encodings.RadialBasis`."""
    lmax: int = 1
    """The maximum angular momentum to consider."""
    correlation: int = 3
    """The correlation order at each layer."""
    activation: str = "silu"
    """The activation function to use."""
    symmetric_tensor_product_basis: bool = False
    """Whether to use the symmetric tensor product basis."""
    interaction_irreps: Union[Irreps, str] = "o3_restricted"
    skip_connection_first_layer: bool = True
    radial_network_hidden: Sequence[int] = dataclasses.field(
        default_factory=lambda: [64, 64, 64]
    )
    scalar_output: bool = False
    zmax: int = 86
    """The maximum atomic number to consider."""
    convolution_mode: int = 1

    FID: ClassVar[str] = "MACE"

    @nn.compact
    def __call__(self, inputs):
        if not E3NN_AVAILABLE:
            raise E3NN_EXCEPTION

        species_indices = inputs["species"]
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        vec = e3nn.IrrepsArray("1o", graph["vec"])
        switch = graph["switch"]
        edge_src = graph["edge_src"]
        edge_dst = graph["edge_dst"]

        output_irreps = e3nn.Irreps(self.output_irreps)
        hidden_irreps = e3nn.Irreps(self.hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(self.readout_mlp_irreps)

        # extract or set num_features
        if self.num_features is None:
            num_features = functools.reduce(math.gcd, (mul for mul, _ in hidden_irreps))
            hidden_irreps = e3nn.Irreps(
                [(mul // num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            num_features = self.num_features

        # get interaction irreps
        if self.interaction_irreps == "o3_restricted":
            interaction_irreps = e3nn.Irreps.spherical_harmonics(self.lmax)
        elif self.interaction_irreps == "o3_full":
            interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(self.lmax))
        else:
            interaction_irreps = e3nn.Irreps(self.interaction_irreps)
        convol_irreps = num_features * interaction_irreps

        # convert species to internal indices
        # maxidx = max(PERIODIC_TABLE_REV_IDX.values())
        # conv_tensor = [0] * (maxidx + 2)
        # if isinstance(self.species_order, str):
        #     species_order = [el.strip() for el in self.species_order.split(",")]
        # else:
        #     species_order = [el for el in self.species_order]
        # for i, s in enumerate(species_order):
        #     conv_tensor[PERIODIC_TABLE_REV_IDX[s]] = i
        # species_indices = jnp.asarray(conv_tensor, dtype=jnp.int32)[species]
        num_species = self.zmax + 2

        # species encoding
        encoding_irreps: e3nn.Irreps = (
            (num_features * hidden_irreps).filter("0e").regroup()
        )
        species_encoding = self.param(
            "species_encoding",
            lambda key, shape: jax.nn.standardize(
                jax.random.normal(key, shape, dtype=jnp.float32)
            ),
            (num_species, encoding_irreps.dim),
        )[species_indices]
        # convert to IrrepsArray
        node_feats = e3nn.IrrepsArray(encoding_irreps, species_encoding)

        # radial embedding
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        radial_embedding = (
            RadialBasis(
                **{
                    **self.radial_basis,
                    "end": cutoff,
                    "name": f"RadialBasis",
                }
            )(distances)
            * switch[:, None]
        )

        # spherical harmonics
        assert self.convolution_mode in [0,1,2], "convolution_mode must be 0, 1 or 2"
        if self.convolution_mode == 0:
            Yij = e3nn.spherical_harmonics(range(0, self.lmax + 1), vec, True)
        elif self.convolution_mode == 1:
            Yij = e3nn.spherical_harmonics(range(1, self.lmax + 1), vec, True)

        outputs = []
        node_feats_all = []
        for layer in range(self.ninteractions):
            first = layer == 0
            last = layer == self.ninteractions - 1

            layer_irreps = num_features * (
                hidden_irreps if not last else hidden_irreps.filter(output_irreps)
            )

            # Linear skip connection
            sc = None
            if not first or self.skip_connection_first_layer:
                sc = e3nn.flax.Linear(
                    layer_irreps,
                    num_indexed_weights=num_species,
                    name=f"skip_tp_{layer}",
                    force_irreps_out=True,
                )(species_indices, node_feats)

            ################################################
            # Interaction block (Message passing convolution)
            node_feats = e3nn.flax.Linear(node_feats.irreps, name=f"linear_up_{layer}")(
                node_feats
            )


            messages = node_feats[edge_src]
            if self.convolution_mode == 0:
                messages = e3nn.tensor_product(
                    messages,
                    Yij,
                    filter_ir_out=convol_irreps,
                    regroup_output=True,
                )
            elif self.convolution_mode == 1:
                messages = e3nn.concatenate(
                    [
                        messages.filter(convol_irreps),
                        e3nn.tensor_product(
                            messages,
                            Yij,
                            filter_ir_out=convol_irreps,
                        ),
                        # e3nn.tensor_product_with_spherical_harmonics(
                        #     messages, vectors, self.max_ell
                        # ).filter(convol_irreps),
                    ]
                ).regroup()
            else:
                messages = e3nn.tensor_product_with_spherical_harmonics(
                    messages, vec, self.lmax
                ).filter(convol_irreps).regroup()

            # mix = FullyConnectedNet(
            #     [*self.radial_network_hidden, messages.irreps.num_irreps],
            #     activation=activation_from_str(self.activation),
            #     name=f"radial_network_{layer}",
            #     use_bias=False,
            # )(radial_embedding)
            mix = e3nn.flax.MultiLayerPerceptron(
                [*self.radial_network_hidden, messages.irreps.num_irreps],
                act=activation_from_str(self.activation),
                output_activation=False,
                name=f"radial_network_{layer}",
                gradient_normalization="element",
            )(
                radial_embedding
            )

            messages = messages * mix
            node_feats = (
                e3nn.IrrepsArray.zeros(
                    messages.irreps, node_feats.shape[:1], messages.dtype
                )
                .at[edge_dst]
                .add(messages)
            )
            # print("irreps_mid jax",node_feats.irreps)
            # jax.debug.print("node_feats={n}", n=jnp.sum(node_feats.array,axis=0)[550:570])

            node_feats = (
                e3nn.flax.Linear(convol_irreps, name=f"linear_dn_{layer}")(node_feats)
                / self.avg_num_neighbors
            )

            if first and not self.skip_connection_first_layer:
                node_feats = e3nn.flax.Linear(
                    node_feats.irreps,
                    num_indexed_weights=num_species,
                    name=f"skip_tp_{layer}",
                )(species_indices, node_feats)

            ################################################
            # Equivariant product basis block

            # symmetric contractions
            node_feats = SymmetricContraction(
                keep_irrep_out={ir for _, ir in layer_irreps},
                correlation=self.correlation,
                num_species=num_species,
                gradient_normalization="element",  # NOTE: This is to copy mace-torch
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            )(
                node_feats, species_indices
            )


            node_feats = e3nn.flax.Linear(
                layer_irreps, name=f"linear_contraction_{layer}"
            )(node_feats)


            if sc is not None:
                # add skip connection
                node_feats = node_feats + sc


            ################################################
            
            # Readout block
            if last:
                num_vectors = readout_mlp_irreps.filter(drop=["0e", "0o"]).num_irreps
                layer_out = e3nn.flax.Linear(
                    (readout_mlp_irreps + e3nn.Irreps(f"{num_vectors}x0e")).simplify(),
                    name=f"hidden_linear_readout_last",
                )(node_feats)
                layer_out = e3nn.gate(
                    layer_out,
                    even_act=activation_from_str(self.activation),
                    even_gate_act=None,
                )
                layer_out = e3nn.flax.Linear(
                    output_irreps, name=f"linear_readout_last"
                )(layer_out)
            else:
                layer_out = e3nn.flax.Linear(
                    output_irreps,
                    name=f"linear_readout_{layer}",
                )(node_feats)

            if self.scalar_output:
                layer_out = layer_out.filter("0e").array

            outputs.append(layer_out)
            node_feats_all.append(node_feats.filter("0e").array)

        if self.scalar_output:
            output = jnp.stack(outputs, axis=1)
        else:
            output = e3nn.stack(outputs, axis=1)

        node_feats_all = jnp.concatenate(node_feats_all, axis=-1)

        output_key = self.output_key if self.output_key is not None else self.name
        return {
            **inputs,
            output_key: output,
            output_key + "_node_feats": node_feats_all,
        }


class SymmetricContraction(nn.Module):

    correlation: int
    keep_irrep_out: Set[Irrep]
    num_species: int
    gradient_normalization: Union[str, float]
    symmetric_tensor_product_basis: bool

    @nn.compact
    def __call__(self, input, index):
        if not E3NN_AVAILABLE:
            raise E3NN_EXCEPTION

        if self.gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        else:
            gradient_normalization = self.gradient_normalization
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]

        keep_irrep_out = self.keep_irrep_out
        if isinstance(keep_irrep_out, str):
            keep_irrep_out = e3nn.Irreps(keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)

        keep_irrep_out = {e3nn.Irrep(ir) for ir in keep_irrep_out}

        input = input.mul_to_axis().remove_nones()

        ### PREPARE WEIGHTS
        ws = []
        Us = []
        for order in range(1, self.correlation + 1):  # correlation, ..., 1
            if self.symmetric_tensor_product_basis:
                U = e3nn.reduced_symmetric_tensor_product_basis(
                    input.irreps, order, keep_ir=keep_irrep_out
                )
            else:
                U = e3nn.reduced_tensor_product_basis(
                    [input.irreps] * order, keep_ir=keep_irrep_out
                )
            # U = U / order  # normalization TODO(mario): put back after testing
            # NOTE(mario): The normalization constants (/order and /mul**0.5)
            # has been numerically checked to be correct.

            # TODO(mario) implement norm_p
            Us.append(U)

            wsorder = []
            for (mul, ir_out), u in zip(U.irreps, U.list):
                u = u.astype(input.array.dtype)
                # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]

                w = self.param(
                    f"w{order}_{ir_out}",
                    nn.initializers.normal(
                        stddev=(mul**-0.5) ** (1.0 - gradient_normalization)
                    ),
                    (self.num_species, mul, input.shape[-2]),
                )
                w = w * (mul**-0.5) ** gradient_normalization
                wsorder.append(w)
            ws.append(wsorder)

        def fn(input: e3nn.IrrepsArray, index: jnp.ndarray):
            # - This operation is parallel on the feature dimension (but each feature has its own parameters)
            # This operation is an efficient implementation of
            # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
            # up to x power self.correlation
            assert input.ndim == 2  # [num_features, irreps_x.dim]
            assert index.ndim == 0  # int

            out = dict()
            x_ = input.array

            for order in range(self.correlation, 0, -1):  # correlation, ..., 1

                U = Us[order - 1]

                # ((w3 x + w2) x + w1) x
                #  \-----------/
                #       out

                for ii, ((mul, ir_out), u) in enumerate(zip(U.irreps, U.list)):
                    u = u.astype(x_.dtype)
                    # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]

                    w = ws[order - 1][ii][index]
                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kc,cj->c...i", u, w, x_),
                        )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                    else:
                        out[ir_out] += jnp.einsum(
                            "...ki,kc->c...i", u, w
                        )  # [num_features, (irreps_x.dim)^order, ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \----------------/
                #         out (in the normal case)

                for ir_out in out:
                    if isinstance(out[ir_out], tuple):
                        out[ir_out] = out[ir_out][1]
                        continue  # already done (special case optimization above)

                    out[ir_out] = jnp.einsum(
                        "c...ji,cj->c...i", out[ir_out], x_
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \-------------------/
                #           out

            # out[irrep_out] : [num_features, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.IrrepsArray.from_list(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (input.shape[0],),
            )

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(input.shape[:-2], index.shape)
        input = input.broadcast_to(shape + input.shape[-2:])
        index = jnp.broadcast_to(index, shape)

        fn_mapped = fn
        for _ in range(input.ndim - 2):
            fn_mapped = jax.vmap(fn_mapped)

        return fn_mapped(input, index).axis_to_mul()


# class SymmetricContraction(nn.Module):

#     correlation: int
#     keep_irrep_out: Set[Irrep]
#     num_species: int
#     gradient_normalization: Union[str, float]
#     symmetric_tensor_product_basis: bool

#     @nn.compact
#     def __call__(self, input: IrrepsArray, index: jnp.ndarray):
#         if not E3NN_AVAILABLE:
#             raise E3NN_EXCEPTION

#         if self.gradient_normalization is None:
#             gradient_normalization = e3nn.config("gradient_normalization")
#         else:
#             gradient_normalization = self.gradient_normalization
#         if isinstance(gradient_normalization, str):
#             gradient_normalization = {"element": 0.0, "path": 1.0}[
#                 gradient_normalization
#             ]

#         keep_irrep_out = self.keep_irrep_out
#         if isinstance(keep_irrep_out, str):
#             keep_irrep_out = e3nn.Irreps(keep_irrep_out)
#             assert all(mul == 1 for mul, _ in keep_irrep_out)

#         keep_irrep_out = {e3nn.Irrep(ir) for ir in keep_irrep_out}

#         onehot = jnp.eye(self.num_species)[index]

#         ### PREPARE WEIGHTS
#         ws = []
#         us = []
#         for ir_out in keep_irrep_out:
#             usorder = []
#             wsorder = []
#             for order in range(1, self.correlation + 1):  # correlation, ..., 1
#                 if self.symmetric_tensor_product_basis:
#                     U = e3nn.reduced_symmetric_tensor_product_basis(
#                         input.irreps, order, keep_ir=[ir_out]
#                     )
#                 else:
#                     U = e3nn.reduced_tensor_product_basis(
#                         [input.irreps] * order, keep_ir=[ir_out]
#                     )
#                 u = jnp.moveaxis(U.list[0].astype(input.array.dtype), -1, 0)
#                 usorder.append(u)

#                 mul, _ = U.irreps[0]
#                 w = self.param(
#                     f"w{order}_{ir_out}",
#                     nn.initializers.normal(
#                         stddev=(mul**-0.5) ** (1.0 - gradient_normalization)
#                     ),
#                     (self.num_species, mul, input.shape[-2]),
#                 )
#                 w = w * (mul**-0.5) ** gradient_normalization
#                 wsorder.append(w)
#             ws.append(wsorder)
#             us.append(usorder)

#         x = input.array

#         outs = []
#         for i, ir in enumerate(keep_irrep_out):
#             w = ws[i][-1]  # [index]
#             u = us[i][-1]
#             out = jnp.einsum("...jk,ekc,bcj,be->bc...", u, w, x, onehot)

#             for order in range(self.correlation - 1, 0, -1):
#                 w = ws[i][order - 1]  # [index]
#                 u = us[i][order - 1]

#                 c_tensor = jnp.einsum("...k,ekc,be->bc...", u, w, onehot) + out
#                 out = jnp.einsum("bc...j,bcj->bc...", c_tensor, x)

#             outs.append(out.reshape(x.shape[0], -1))

#         out = jnp.concatenate(outs, axis=-1)

#         return e3nn.IrrepsArray(input.shape[1] * e3nn.Irreps(keep_irrep_out), out)