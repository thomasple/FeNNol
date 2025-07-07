import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict, Union, ClassVar, Optional
import numpy as np
from ...utils.periodic_table import D3_COV_RADII
import dataclasses
from ..misc.encodings import RadialBasis, SpeciesEncoding
from ...utils.spherical_harmonics import generate_spherical_harmonics
from ..misc.e3 import  ChannelMixing
from ..misc.nets import FullyConnectedNet,BlockIndexNet
from ...utils.activations import activation_from_str
from ...utils import AtomicUnits as au
from ...utils.initializers import initializer_from_str

class RaSTER(nn.Module):
    """ Range-Separated Transformer with Equivariant Representations

    FID : RASTER

    """

    _graphs_properties: Dict
    dim: int = 176
    """The dimension of the output embedding."""
    nlayers: int = 2
    """The number of message-passing layers."""
    att_dim: int = 16
    """The dimension of the attention heads."""
    scal_heads: int = 16
    """The number of scalar attention heads."""
    tens_heads: int = 4
    """The number of tensor attention heads."""
    lmax: int = 3
    """The maximum angular momentum to consider."""
    normalize_vec: bool = True
    """Whether to normalize the vector features before computing spherical harmonics."""
    att_activation: str = "identity"
    """The activation function to use for the attention coefficients."""
    activation: str = "swish"
    """The activation function to use for the update network."""
    update_hidden: Sequence[int] = ()
    """The hidden layers for the update network."""
    update_bias: bool = True
    """Whether to use bias in the update network."""
    positional_activation: str = "swish"
    """The activation function to use for the positional embedding network."""
    positional_bias: bool = True
    """Whether to use bias in the positional embedding network."""
    switch_before_net: bool = False
    """Whether to apply the switch function to the radial basis before the edge neural network."""
    ignore_parity: bool = False
    """Whether to ignore the parity of the spherical harmonics when constructing the relative positional encoding."""
    additive_positional: bool = False
    """Whether to use additive relative positional encoding. If False, multiplicative relative positional encoding is used."""
    edge_value: bool = False
    """Whether to use edge values in the attention mechanism."""
    layer_normalization: bool = True
    """Whether to use layer normalization of atomic embeddings."""
    graph_key: str = "graph"
    """ The key in the input dictionary that corresponds to the radial graph."""
    embedding_key: str = "embedding"
    """ The key in the output dictionary that corresponds to the embedding."""
    radial_basis: dict = dataclasses.field(
        default_factory=lambda: {"start": 0.8, "basis": "gaussian", "dim": 16}
    )
    """The dictionary of parameters for radial basis functions. See `fennol.models.misc.encodings.RadialBasis`."""
    species_encoding: str | dict = dataclasses.field(
        default_factory=lambda: {"dim": 16, "trainable": True, "encoding": "random"}
    )
    """The dictionary of parameters for species encoding. See `fennol.models.misc.encodings.SpeciesEncoding`."""
    graph_lode: Optional[str] = None
    """The key in the input dictionary that corresponds to the long-range graph."""
    lmax_lode: int = 0
    """The maximum angular momentum for the long-range features."""
    lode_rshort: Optional[float] = None
    """The short-range cutoff for the long-range features."""
    lode_dshort: float = 2.0
    """The width of the short-range cutoff for the long-range features."""
    lode_extra_powers: Sequence[int] = ()
    """The extra powers to include in the long-range features."""
    a_lode: float = -1.0
    """The damping parameter for the long-range features. If negative, the damping is trainable with initial value abs(a_lode)."""
    block_index_key: Optional[str] = None
    """The key in the input dictionary that corresponds to the block index for the MoE network. If None, a normal neural network is used."""
    lode_channels: int = 1
    """The number of channels for the long-range features."""
    switch_cov_start: float = 0.5
    """The start of close-range covalent switch (in units of covalent radii)."""
    switch_cov_end: float = 0.6
    """The end of close-range covalent switch (in units of covalent radii)."""
    normalize_keys: bool = False
    """Whether to normalize queries and keys in the attention mechanism."""
    keep_all_layers: bool = False
    """Whether to return the stacked scalar embeddings from all message-passing layers."""
    kernel_init: Optional[str] = None
    
    FID: ClassVar[str] = "RASTER"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]

        ## SETUP LAYER NORMALIZATION
        def _layer_norm(x):
            mu = jnp.mean(x, axis=-1, keepdims=True)
            dx = x - mu
            var = jnp.mean(dx**2, axis=-1, keepdims=True)
            sig = (1.0e-6 + var) ** (-0.5)
            return dx * sig

        if self.layer_normalization:
            layer_norm = _layer_norm
        else:
            layer_norm = lambda x: x
        
        if self.normalize_keys:
            ln_qk = _layer_norm
        else:
            ln_qk = lambda x: x

        kernel_init = initializer_from_str(self.kernel_init)

        ## SPECIES ENCODING
        if isinstance(self.species_encoding, str):
            Zi = inputs[self.species_encoding]
        else:
            Zi = SpeciesEncoding(**self.species_encoding)(species)

        ## INITIALIZE SCALAR FEATURES
        xi = layer_norm(nn.Dense(self.dim, use_bias=False,name="species_linear",kernel_init=kernel_init)(Zi))

        # RADIAL GRAPH
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        switch = graph["switch"]
        edge_src = graph["edge_src"]
        edge_dst = graph["edge_dst"]
        vec = (
            graph["vec"] / graph["distances"][:, None]
            if self.normalize_vec
            else graph["vec"]
        )
        ## CLOSE-RANGE SWITCH
        use_switch_cov = False
        if self.switch_cov_end > 0 and self.switch_cov_start > 0:
            use_switch_cov = True
            assert self.switch_cov_start < self.switch_cov_end, f"switch_cov_start {self.switch_cov_start} must be smaller than switch_cov_end {self.switch_cov_end}"
            assert self.switch_cov_start > 0 and self.switch_cov_end < 1, f"switch_cov_start {self.switch_cov_start} and switch_cov_end {self.switch_cov_end} must be between 0 and 1"
            rc = jnp.array(D3_COV_RADII*au.BOHR)[species]
            rcij = rc[edge_src] + rc[edge_dst]
            rstart = rcij * self.switch_cov_start
            rend = rcij * self.switch_cov_end
            switch_short = (distances >= rend) + 0.5*(1-jnp.cos(jnp.pi*(distances - rstart)/(rend-rstart)))*(distances > rstart)*(distances < rend)
            switch = switch * switch_short

        ## COMPUTE SPHERICAL HARMONICS ON EDGES
        Yij = generate_spherical_harmonics(lmax=self.lmax, normalize=False)(vec)[:,None,:]
        nrep = np.array([2 * l + 1 for l in range(self.lmax + 1)])
        ls = np.arange(self.lmax + 1).repeat(nrep)
            
        parity = jnp.array((-1) ** ls[None,None,:])
        if self.ignore_parity:
            parity = -jnp.ones_like(parity)

        ## INITIALIZE TENSOR FEATURES
        Vi = 0. #jnp.zeros((Zi.shape[0],self.tens_heads, Yij.shape[1]))

        # RADIAL BASIS
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        radial_terms = RadialBasis(
            **{
                **self.radial_basis,
                "end": cutoff,
                "name": f"RadialBasis",
            }
        )(distances)
        if self.switch_before_net:
            radial_terms = radial_terms * switch[:, None]
        elif use_switch_cov:
            radial_terms = radial_terms * switch_short[:, None]

        ## INITIALIZE LODE
        do_lode = self.graph_lode is not None
        if do_lode:
            ## LONG-RANGE GRAPH
            graph_lode = inputs[self.graph_lode]
            switch_lode = graph_lode["switch"][:, None]
            edge_src_lr, edge_dst_lr = graph_lode["edge_src"], graph_lode["edge_dst"]
            r = graph_lode["distances"][:, None]
            rc = self._graphs_properties[self.graph_lode]["cutoff"]

            lmax_lr = self.lmax_lode
            equivariant_lode = lmax_lr > 0
            assert lmax_lr >= 0, f"lmax_lode must be >= 0, got {lmax_lr}"
            assert (
                lmax_lr <= self.lmax
            ), f"lmax_lode must be <= lmax for multipole interaction, got {lmax_lr} > {self.lmax}"
            nrep_lr = np.array([2 * l + 1 for l in range(lmax_lr + 1)], dtype=np.int32)
            if equivariant_lode:
                ls_lr = np.arange(lmax_lr + 1)
            else:
                ls_lr = np.array([0])

            ## PARAMETERS FOR THE LR RADIAL BASIS
            nextra_powers = len(self.lode_extra_powers)
            if nextra_powers > 0:
                ls_lr = np.concatenate([self.lode_extra_powers, ls_lr])

            if self.a_lode > 0:
                a = self.a_lode**2
            else:
                a = (
                    self.param(
                        "a_lr",
                        lambda key: jnp.asarray([-self.a_lode] * ls_lr.shape[0])[
                            None, :
                        ],
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

            dim_lr = 1
            if nextra_powers > 0:
                eij_lr_extra = eij_lr[:, :nextra_powers]
                eij_lr = eij_lr[:, nextra_powers:]
                dim_lr += nextra_powers

            if equivariant_lode:
                ## SPHERICAL HARMONICS ON LONG-RANGE GRAPH
                eij_lr = eij_lr.repeat(nrep_lr, axis=-1)
                Yij_lr = generate_spherical_harmonics(lmax=lmax_lr, normalize=False)(
                    graph_lode["vec"] / r
                )
                dim_lr += lmax_lr
                eij_lr = eij_lr * Yij_lr
                del Yij_lr
        

        if self.keep_all_layers:
            xis = []

        ### START MESSAGE PASSING ITERATIONS
        for layer in range(self.nlayers):
            ## GATHER SCALAR EDGE FEATURES
            u = [radial_terms]
            if layer > 0:
                ## edge-tensor contraction
                xij2 = (Vi[edge_dst] + (parity* Vi)[edge_src]) * Yij
                for l in range(self.lmax + 1):
                    u.append((xij2[:,:, l**2 : (l + 1) ** 2]).sum(axis=-1))
            ur = jnp.concatenate(u, axis=-1)

            ## BUILD RELATIVE POSITIONAL ENCODING
            if self.edge_value:
                nout = 2
            else:
                nout = 1
            w = FullyConnectedNet(
                [2 * self.att_dim, nout*self.att_dim],
                activation=self.positional_activation,
                use_bias=self.positional_bias,
                name=f"positional_encoding_{layer}",
            )(ur).reshape(radial_terms.shape[0],nout, self.att_dim)
            if self.edge_value:
                w,vij = jnp.split(w, 2, axis=1)

            nls = self.lmax + 1 if layer == 0 else 2 * (self.lmax + 1)


            ## QUERY, KEY, VALUE
            q = ln_qk(nn.Dense((self.scal_heads + nls*self.tens_heads) * self.att_dim, use_bias=False,name=f"queries_{layer}",kernel_init=kernel_init)(
                xi
            ).reshape(xi.shape[0], self.scal_heads + nls*self.tens_heads, self.att_dim))
            k = nn.Dense((self.scal_heads + nls*self.tens_heads) * self.att_dim, use_bias=False, name=f"keys_{layer}",kernel_init=kernel_init)(
                xi
            ).reshape(xi.shape[0], self.scal_heads + nls*self.tens_heads, self.att_dim)

            v = nn.Dense(self.scal_heads * self.att_dim, use_bias=False, name=f"values_{layer}",kernel_init=kernel_init)(xi).reshape(
                xi.shape[0], self.scal_heads, self.att_dim
            )

            ## ATTENTION COEFFICIENTS
            if self.additive_positional:
                wk = ln_qk(w + k[edge_dst])
            else:
                wk = ln_qk(w * k[edge_dst])

            act = activation_from_str(self.att_activation)
            aij = (
                act((q[edge_src] * wk).sum(axis=-1) / (self.att_dim**0.5))
                * switch[:, None]
            )

            aijl = aij[:, : self.tens_heads*(self.lmax + 1)].reshape(-1,self.tens_heads,self.lmax+1).repeat(nrep, axis=-1)
            if layer > 0:
                aijl1 = aij[:, self.tens_heads*(self.lmax + 1) : self.tens_heads*nls].reshape(-1,self.tens_heads,self.lmax+1).repeat(nrep, axis=-1)
            aij = aij[:, self.tens_heads*nls:, None]

            if self.edge_value:
                ## EDGE VALUES
                if self.additive_positional:
                    vij = vij + v[edge_dst]
                else:
                    vij = vij * v[edge_dst]
            else:
                ## MOVE DEST VALUES TO EDGE
                vij = v[edge_dst]

            ## SCALAR ATTENDED FEATURES
            vai = jax.ops.segment_sum(
                aij * vij,
                edge_src,
                num_segments=xi.shape[0],
            )
            vai = vai.reshape(xi.shape[0], -1)

            ### TENSOR ATTENDED FEATURES
            uij = aijl * Yij
            if layer > 0:
                uij = uij + aijl1 * Vi[edge_dst]
            Vi = Vi + jax.ops.segment_sum(uij, edge_src, num_segments=Zi.shape[0])

            ## SELF SCALAR FEATURES
            si = nn.Dense(self.att_dim, use_bias=False, name=f"self_values_{layer}",kernel_init=kernel_init)(xi)

            components = [si, vai]

            ### CONTRACT TENSOR FEATURES TO BUILD INVARIANTS
            if self.tens_heads == 1:
                Vi2 = Vi**2
            else:
                Vi2 = Vi * ChannelMixing(self.lmax, self.tens_heads, name=f"extract_mixing_{layer}")(Vi)
            for l in range(self.lmax + 1):
                norm = 1.0 / (2 * l + 1)
                components.append(
                    (Vi2[:,:, l**2 : (l + 1) ** 2]).sum(axis=-1) * norm
                )

            ### LODE (~ LONG-RANGE ATTENTION)
            if do_lode and layer == self.nlayers - 1:
                assert self.lode_channels <= self.tens_heads
                zj = nn.Dense(self.lode_channels*dim_lr, use_bias=False, name=f"lode_values_{layer}",kernel_init=kernel_init)(xi).reshape(
                    xi.shape[0], self.lode_channels, dim_lr
                )
                if nextra_powers > 0:
                    zj_extra = zj[:,:, :nextra_powers]
                    zj = zj[:, :, nextra_powers:]
                    xi_lr_extra = jax.ops.segment_sum(
                        eij_lr_extra[:,None,:] * zj_extra[edge_dst_lr],
                        edge_src_lr,
                        species.shape[0],
                    ).reshape(species.shape[0],-1)
                    components.append(xi_lr_extra)
                if equivariant_lode:
                    zj = zj.repeat(nrep_lr, axis=-1)
                Vi_lr = jax.ops.segment_sum(
                    eij_lr[:,None,:] * zj[edge_dst_lr], edge_src_lr, species.shape[0]
                )
                components.append(Vi_lr[:,: , 0])
                if equivariant_lode:
                    Mi_lr = Vi[:,:self.lode_channels, : (lmax_lr + 1) ** 2] * Vi_lr
                    for l in range(1, lmax_lr + 1):
                        norm = 1.0 / (2 * l + 1)
                        components.append(
                            Mi_lr[:, :,l**2 : (l + 1) ** 2].sum(axis=-1)
                            * norm
                        )

            ### CONCATENATE UPDATE COMPONENTS
            components = jnp.concatenate(components, axis=-1)
            ### COMPUTE UPDATE
            if self.block_index_key is not None:
                ## MoE neural network from block index
                block_index = inputs[self.block_index_key]
                updi = BlockIndexNet(
                        output_dim=self.dim + self.tens_heads*(self.lmax + 1),
                        hidden_neurons=self.update_hidden,
                        activation=self.activation,
                        use_bias=self.update_bias,
                        name=f"update_net_{layer}",
                        kernel_init=kernel_init,
                    )((species,components, block_index))
            else:
                updi = FullyConnectedNet(
                        [*self.update_hidden, self.dim + self.tens_heads*(self.lmax + 1)],
                        activation=self.activation,
                        use_bias=self.update_bias,
                        name=f"update_net_{layer}",
                        kernel_init=kernel_init,
                    )(components)
                
            ## UPDATE ATOM FEATURES
            xi = layer_norm(xi + updi[:,:self.dim])
            Vi = Vi * (1 + updi[:,self.dim:]).reshape(-1,self.tens_heads,self.lmax+1).repeat(nrep, axis=-1)
            if self.tens_heads > 1:
                Vi = ChannelMixing(self.lmax, self.tens_heads,name=f"update_mixing_{layer}")(Vi)

            if self.keep_all_layers:
                ## STORE ALL LAYERS
                xis.append(xi)


        output = {**inputs, self.embedding_key: xi, self.embedding_key + "_tensor": Vi}
        if self.keep_all_layers:
            output[self.embedding_key+'_layers'] = jnp.stack(xis,axis=1)
        return output