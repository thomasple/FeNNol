import flax.linen as nn
from typing import Sequence, Callable, Union, Optional, ClassVar, Tuple
from ...utils.periodic_table import PERIODIC_TABLE_REV_IDX, PERIODIC_TABLE
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
from ...utils.activations import activation_from_str, TrainableSiLU
from ...utils.initializers import initializer_from_str, scaled_orthogonal
from flax.core import FrozenDict


class FullyConnectedNet(nn.Module):
    """A fully connected neural network module.

    FID: NEURAL_NET
    """

    neurons: Sequence[int]
    """A sequence of integers representing the dimensions of the network."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use."""
    use_bias: bool = True
    """Whether to use bias in the dense layers."""
    input_key: Optional[str] = None
    """The key of the input tensor."""
    output_key: Optional[str] = None
    """The key of the output tensor."""
    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method to use."""

    FID: ClassVar[str] = "NEURAL_NET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        """Applies the neural network to the given inputs.

        Args:
            inputs (Union[dict, jax.Array]): If input_key is None, a JAX array containing the inputs to the neural network. Else, a dictionary containing the inputs at the key input_key.

        Returns:
            Union[dict, jax.Array]: If output_key is None, the output tensor of the neural network. Else, a dictionary containing the original inputs and the output tensor at the key output_key.
        """
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            x = inputs
        else:
            x = inputs[self.input_key]

        # activation = (
        #     activation_from_str(self.activation)
        #     if isinstance(self.activation, str)
        #     else self.activation
        # )
        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )
        ############################
        if isinstance(self.activation, str) and self.activation.lower() == "swiglu":
            for i, d in enumerate(self.neurons[:-1]):
                y = nn.Dense(
                    d,
                    use_bias=self.use_bias,
                    name=f"Layer_{i+1}",
                    kernel_init=kernel_init,
                )(x)
                z = jax.nn.swish(nn.Dense(
                    d,
                    use_bias=self.use_bias,
                    name=f"Mask_{i+1}",
                    kernel_init=kernel_init,
                )(x))
                x = y * z
        else:
            for i, d in enumerate(self.neurons[:-1]):
                x = nn.Dense(
                    d,
                    use_bias=self.use_bias,
                    name=f"Layer_{i+1}",
                    kernel_init=kernel_init,
                )(x)
                x = activation_from_str(self.activation)(x)
        x = nn.Dense(
            self.neurons[-1],
            use_bias=self.use_bias,
            name=f"Layer_{len(self.neurons)}",
            kernel_init=kernel_init,
        )(x)
        if self.squeeze and x.shape[-1] == 1:
            x = jnp.squeeze(x, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: x} if output_key is not None else x
        return x


class ResMLP(nn.Module):
    """Residual neural network as defined in the SpookyNet paper.
    
    FID: RES_MLP
    """

    use_bias: bool = True
    """Whether to include bias in the linear layers."""
    input_key: Optional[str] = None
    """The key of the input tensor."""
    output_key: Optional[str] = None
    """The key of the output tensor."""

    kernel_init: Union[str, Callable] = "scaled_orthogonal(mode='fan_avg')"
    """The kernel initialization method to use."""
    res_only: bool = False
    """Whether to only apply the residual connection without additional activation and linear layer."""

    FID: ClassVar[str] = "RES_MLP"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            x = inputs
        else:
            x = inputs[self.input_key]

        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )
        ############################
        out = nn.Dense(x.shape[-1], use_bias=self.use_bias, kernel_init=kernel_init)(
            TrainableSiLU()(x)
        )
        out = x + nn.Dense(
            x.shape[-1], use_bias=self.use_bias, kernel_init=nn.initializers.zeros
        )(TrainableSiLU()(out))

        if not self.res_only:
            out = nn.Dense(
                x.shape[-1], use_bias=self.use_bias, kernel_init=kernel_init
            )(TrainableSiLU()(out))
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class FullyResidualNet(nn.Module):
    """A neural network with skip connections at each layer.
    
    FID: SKIP_NET
    """

    dim: int
    """The dimension of the hidden layers."""
    output_dim: int
    """The dimension of the output layer."""
    nlayers: int
    """The number of layers in the network."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use."""
    use_bias: bool = True
    """Whether to include bias terms in the linear layers."""
    input_key: Optional[str] = None
    """The key of the input tensor."""
    output_key: Optional[str] = None
    """The key of the output tensor."""
    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method to use."""

    FID: ClassVar[str] = "SKIP_NET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            x = inputs
        else:
            x = inputs[self.input_key]

        # activation = (
        #     activation_from_str(self.activation)
        #     if isinstance(self.activation, str)
        #     else self.activation
        # )
        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )
        ############################
        if x.shape[-1] != self.dim:
            x = nn.Dense(
                self.dim,
                use_bias=self.use_bias,
                name=f"Reshape",
                kernel_init=kernel_init,
            )(x)

        for i in range(self.nlayers - 1):
            x = x + activation_from_str(self.activation)(
                nn.Dense(
                    self.dim,
                    use_bias=self.use_bias,
                    name=f"Layer_{i+1}",
                    kernel_init=kernel_init,
                )(x)
            )
        x = nn.Dense(
            self.output_dim,
            use_bias=self.use_bias,
            name=f"Layer_{self.nlayers}",
            kernel_init=kernel_init,
        )(x)
        if self.squeeze and x.shape[-1] == 1:
            x = jnp.squeeze(x, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: x} if output_key is not None else x
        return x


class HierarchicalNet(nn.Module):
    """Neural network for a sequence of inputs (in axis=-2) with a decay factor
    
    FID: HIERARCHICAL_NET
    """

    neurons: Sequence[int]
    """A sequence of integers representing the number of neurons in each layer."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use."""
    use_bias: bool = True
    """Whether to include bias terms in the linear layers."""
    input_key: Optional[str] = None
    """The key of the input tensor."""
    output_key: Optional[str] = None
    """The key of the output tensor."""
    decay: float = 0.01
    """The decay factor to scale each element of the sequence."""
    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method to use."""

    FID: ClassVar[str] = "HIERARCHICAL_NET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            x = inputs
        else:
            x = inputs[self.input_key]

        ############################
        networks = nn.vmap(
            FullyConnectedNet,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=-2,
            out_axes=-2,
            kenel_init=self.kernel_init,
        )(self.neurons, self.activation, self.use_bias)

        out = networks(x)
        # scale each layer by a decay factor
        decay = jnp.asarray([self.decay**i for i in range(out.shape[-2])])
        out = out * decay[..., :, None]

        if self.squeeze and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class SpeciesIndexNet(nn.Module):
    """Chemical-species-specific neural network using precomputed species index.

    FID: SPECIES_INDEX_NET

    A neural network that applies a species-specific fully connected network to each atom embedding.
    A species index must be provided to filter the embeddings for each species and apply the corresponding network.
    This index can be obtained using the SPECIES_INDEXER preprocessing module from `fennol.models.preprocessing.SpeciesIndexer`

    """

    output_dim: int
    """The dimension of the output of the fully connected networks."""
    hidden_neurons: Union[dict, FrozenDict, Sequence[int]]
    """The hidden dimensions of the fully connected networks.
        If a dictionary is provided, it should map species names to dimensions.
        If a sequence is provided, the same dimensions will be used for all species."""
    species_order: Optional[Union[str, Sequence[str]]] = None
    """The species for which to build a network. Only required if neurons is not a dictionary."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use in the fully connected networks."""
    use_bias: bool = True
    """Whether to include bias terms in the fully connected networks."""
    input_key: Optional[str] = None
    """The key in the input dictionary that corresponds to the embeddings of the atoms."""
    species_index_key: str = "species_index"
    """The key in the input dictionary that corresponds to the species index of the atoms. See `fennol.models.preprocessing.SpeciesIndexer`"""
    output_key: Optional[str] = None
    """The key in the output dictionary that corresponds to the network's output."""

    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method for the fully connected networks."""
    check_unhandled: bool = True

    FID: ClassVar[str] = "SPECIES_INDEX_NET"

    def setup(self):
        if not (
            isinstance(self.hidden_neurons, dict)
            or isinstance(self.hidden_neurons, FrozenDict)
        ):
            assert (
                self.species_order is not None
            ), "species_order must be provided if hidden_neurons is not a dictionary"
            if isinstance(self.species_order, str):
                species_order = [el.strip() for el in self.species_order.split(",")]
            else:
                species_order = [el for el in self.species_order]
            neurons = {k: self.hidden_neurons for k in species_order}
        else:
            neurons = self.hidden_neurons
            species_order = list(neurons.keys())
        for species in species_order:
            assert (
                species in PERIODIC_TABLE
            ), f"species {species} not found in periodic table"

        self.networks = {
            k: FullyConnectedNet(
                [*neurons[k], self.output_dim],
                self.activation,
                self.use_bias,
                name=k,
                kernel_init=self.kernel_init,
            )
            for k in species_order
        }

    def __call__(
        self, inputs: Union[dict, Tuple[jax.Array, jax.Array]]
    ) -> Union[dict, jax.Array]:

        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            species, embedding, species_index = inputs
        else:
            species, embedding = inputs["species"], inputs[self.input_key]
            species_index = inputs[self.species_index_key]

        assert isinstance(
            species_index, dict
        ), "species_index must be a dictionary for SpeciesIndexNetHet"

        ############################
        # initialization => instantiate all networks
        if self.is_initializing():
            x = jnp.zeros((1, embedding.shape[-1]), dtype=embedding.dtype)
            [net(x) for net in self.networks.values()]

        if self.check_unhandled:
            for b in species_index.keys():
                if b not in self.networks.keys():
                    raise ValueError(f"Species {b} not found in networks. Handled species are {self.networks.keys()}")

        ############################
        outputs = []
        indices = []
        for s, net in self.networks.items():
            if s not in species_index:
                continue
            idx = species_index[s]
            o = net(embedding[idx])
            outputs.append(o)
            indices.append(idx)

        o = jnp.concatenate(outputs, axis=0)
        idx = jnp.concatenate(indices, axis=0)

        out = (
            jnp.zeros((species.shape[0], *o.shape[1:]), dtype=o.dtype)
            .at[idx]
            .set(o, mode="drop")
        )

        if self.squeeze and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class ChemicalNet(nn.Module):
    """optimized Chemical-species-specific neural network.

    FID: CHEMICAL_NET

    A neural network that applies a fully connected network to each atom embedding in a chemical system and selects the output corresponding to the atom's species.
    This is an optimized version of ChemicalNetHet that uses vmap to apply the networks to all atoms at once.
    The optimization is allowed because all networks have the same shape.

    """

    species_order: Union[str, Sequence[str]]
    """The species for which to build a network."""
    neurons: Sequence[int]
    """The dimensions of the fully connected networks."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use in the fully connected networks."""
    use_bias: bool = True
    """Whether to include bias terms in the fully connected networks."""
    input_key: Optional[str] = None
    """The key in the input dictionary that corresponds to the embeddings of the atoms."""
    output_key: Optional[str] = None
    """The key in the output dictionary that corresponds to the network's output."""
    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method for the fully connected networks."""

    FID: ClassVar[str] = "CHEMICAL_NET"

    @nn.compact
    def __call__(
        self, inputs: Union[dict, Tuple[jax.Array, jax.Array]]
    ) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            species, embedding = inputs
        else:
            species, embedding = inputs["species"], inputs[self.input_key]

        ############################
        # build species to network index mapping (static => fixed when jitted)
        rev_idx = PERIODIC_TABLE_REV_IDX
        maxidx = max(rev_idx.values())
        if isinstance(self.species_order, str):
            species_order = [el.strip() for el in self.species_order.split(",")]
        else:
            species_order = [el for el in self.species_order]
        nspecies = len(species_order)
        conv_tensor_ = np.full((maxidx + 2,), -1, dtype=np.int32)
        for i, s in enumerate(species_order):
            conv_tensor_[rev_idx[s]] = i
        conv_tensor = jnp.asarray(conv_tensor_)
        indices = conv_tensor[species]

        ############################
        # build shape-sharing networks using vmap
        networks = nn.vmap(
            FullyConnectedNet,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=0,
        )(self.neurons, self.activation, self.use_bias, kernel_init=self.kernel_init)
        # repeat input along a new axis to compute for all species at once
        x = jnp.broadcast_to(
            embedding[None, :, :], (nspecies, *embedding.shape)
        )

        # apply networks to input and select the output corresponding to the species
        out = jnp.squeeze(
            jnp.take_along_axis(networks(x), indices[None, :, None], axis=0), axis=0
        )

        out = jnp.where((indices >= 0)[:, None], out, 0.0)
        if self.squeeze and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class MOENet(nn.Module):
    """Mixture of Experts neural network.

    FID: MOE_NET

    This class represents a Mixture of Experts neural network. It takes in an input and applies a set of shape-sharing networks
    to the input based on a router. The outputs of the shape-sharing networks are then combined using weights computed by the router.

    """

    neurons: Sequence[int]
    """A sequence of integers representing the number of neurons in each shape-sharing network."""
    num_networks: int
    """The number of shape-sharing networks to create."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use in the shape-sharing networks."""
    use_bias: bool = True
    """Whether to include bias in the shape-sharing networks."""
    input_key: Optional[str] = None
    """The key of the input tensor."""
    output_key: Optional[str] = None
    """The key of the output tensor."""
    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method to use in the shape-sharing networks."""
    router_key: Optional[str] = None
    """The key of the router tensor. If None, the router is assumed to be the same as the input tensor."""

    FID: ClassVar[str] = "MOE_NET"

    @nn.compact
    def __call__(
        self, inputs: Union[dict, Tuple[jax.Array, jax.Array]]
    ) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            if isinstance(inputs, tuple):
                embedding, router = inputs
            else:
                embedding = router = inputs
        else:
            embedding = inputs[self.input_key]
            router = (
                inputs[self.router_key] if self.router_key is not None else embedding
            )

        ############################
        # build shape-sharing networks using vmap
        networks = nn.vmap(
            FullyConnectedNet,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=0,
            out_axes=0,
        )(self.neurons, self.activation, self.use_bias, kernel_init=self.kernel_init)
        # repeat input along a new axis to compute for all networks at once
        x = jnp.repeat(embedding[None, :, :], self.num_networks, axis=0)

        w = nn.softmax(nn.Dense(self.num_networks, name="router")(router), axis=-1)

        out = (networks(x) * w.T[:, :, None]).sum(axis=0)

        if self.squeeze and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out

class ChannelNet(nn.Module):
    """Apply a different neural network to each channel.
    
    FID: CHANNEL_NET
    """

    neurons: Sequence[int]
    """A sequence of integers representing the number of neurons in each shape-sharing network."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use in the shape-sharing networks."""
    use_bias: bool = True
    """Whether to include bias in the shape-sharing networks."""
    input_key: Optional[str] = None
    """The key of the input tensor."""
    output_key: Optional[str] = None
    """The key of the output tensor."""
    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method to use in the shape-sharing networks."""
    channel_axis: int = -2
    """The axis to use as channel. Its length will be the number of shape-sharing networks."""

    FID: ClassVar[str] = "CHANNEL_NET"

    @nn.compact
    def __call__(
        self, inputs: Union[dict, Tuple[jax.Array, jax.Array]]
    ) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            x = inputs
        else:
            x = inputs[self.input_key]

        ############################
        # build shape-sharing networks using vmap
        networks = nn.vmap(
            FullyConnectedNet,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=self.channel_axis,
            out_axes=self.channel_axis,
        )(self.neurons, self.activation, self.use_bias, kernel_init=self.kernel_init)

        out = networks(x)

        if self.squeeze and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class GatedPerceptron(nn.Module):
    """Gated Perceptron neural network.

    FID: GATED_PERCEPTRON

    This class represents a Gated Perceptron neural network model. It applies a gating mechanism
    to the input data and performs linear transformation using a dense layer followed by an activation function.
    """

    dim: int
    """The dimensionality of the output space."""
    use_bias: bool = True
    """Whether to include a bias term in the dense layer."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method to use."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use."""

    input_key: Optional[str] = None
    """The key of the input tensor."""
    output_key: Optional[str] = None
    """The key of the output tensor."""
    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""

    FID: ClassVar[str] = "GATED_PERCEPTRON"

    @nn.compact
    def __call__(self, inputs):
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            x = inputs
        else:
            x = inputs[self.input_key]

        # activation = (
        #     activation_from_str(self.activation)
        #     if isinstance(self.activation, str)
        #     else self.activation
        # )
        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )
        ############################
        gate = jax.nn.sigmoid(
            nn.Dense(self.dim, use_bias=self.use_bias, kernel_init=kernel_init)(x)
        )
        x = gate * activation_from_str(self.activation)(
            nn.Dense(self.dim, use_bias=self.use_bias, kernel_init=kernel_init)(x)
        )

        if self.squeeze and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: x} if output_key is not None else x
        return x


class ZAcNet(nn.Module):
    """ A fully connected neural network module with affine Z-dependent adjustments of activations.
    
    FID: ZACNET
    """

    neurons: Sequence[int]
    """A sequence of integers representing the dimensions of the network."""
    zmax: int = 86
    """The maximum atomic number to consider."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use."""
    use_bias: bool = True
    """Whether to use bias in the dense layers."""
    input_key: Optional[str] = None
    """The key of the input tensor."""
    output_key: Optional[str] = None
    """The key of the output tensor."""
    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method to use."""
    species_key: str = "species"
    """The key of the species tensor."""

    FID: ClassVar[str] = "ZACNET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            species, x = inputs
        else:
            species, x = inputs[self.species_key], inputs[self.input_key]

        # activation = (
        #     activation_from_str(self.activation)
        #     if isinstance(self.activation, str)
        #     else self.activation
        # )
        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )
        ############################
        for i, d in enumerate(self.neurons[:-1]):
            x = nn.Dense(
                d, use_bias=self.use_bias, name=f"Layer_{i+1}", kernel_init=kernel_init
            )(x)
            sig = self.param(
                f"sig_{i+1}",
                lambda key, shape: jnp.ones(shape, dtype=x.dtype),
                (self.zmax + 2, d),
            )[species]
            if self.use_bias:
                b = self.param(
                    f"b_{i+1}",
                    lambda key, shape: jnp.zeros(shape, dtype=x.dtype),
                    (self.zmax + 2, d),
                )[species]
            else:
                b = 0
            x = activation_from_str(self.activation)(sig * x + b)
        x = nn.Dense(
            self.neurons[-1],
            use_bias=self.use_bias,
            name=f"Layer_{len(self.neurons)}",
            kernel_init=kernel_init,
        )(x)
        sig = self.param(
            f"sig_{len(self.neurons)}",
            lambda key, shape: jnp.ones(shape, dtype=x.dtype),
            (self.zmax + 2, self.neurons[-1]),
        )[species]
        if self.use_bias:
            b = self.param(
                f"b_{len(self.neurons)}",
                lambda key, shape: jnp.zeros(shape, dtype=x.dtype),
                (self.zmax + 2, self.neurons[-1]),
            )[species]
        else:
            b = 0
        x = sig * x + b
        if self.squeeze and x.shape[-1] == 1:
            x = jnp.squeeze(x, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: x} if output_key is not None else x
        return x


class ZLoRANet(nn.Module):
    """A fully connected neural network module with Z-dependent low-rank adaptation.
    
    FID: ZLORANET
    """

    neurons: Sequence[int]
    """A sequence of integers representing the dimensions of the network."""
    ranks: Sequence[int]
    """A sequence of integers representing the ranks of the low-rank adaptation at each layer."""
    zmax: int = 86
    """The maximum atomic number to consider."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use."""
    use_bias: bool = True
    """Whether to use bias in the dense layers."""
    input_key: Optional[str] = None
    """The key of the input tensor."""
    output_key: Optional[str] = None
    """The key of the output tensor."""
    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method to use."""
    species_key: str = "species"
    """The key of the species tensor."""

    FID: ClassVar[str] = "ZLORANET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            species, x = inputs
        else:
            species, x = inputs[self.species_key], inputs[self.input_key]

        # activation = (
        #     activation_from_str(self.activation)
        #     if isinstance(self.activation, str)
        #     else self.activation
        # )
        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )
        ############################
        for i, d in enumerate(self.neurons[:-1]):
            xi = nn.Dense(
                d, use_bias=self.use_bias, name=f"Layer_{i+1}", kernel_init=kernel_init
            )(x)
            A = self.param(
                f"A_{i+1}",
                lambda key, shape: jnp.zeros(shape, dtype=x.dtype),
                (self.zmax + 2, self.ranks[i], x.shape[-1]),
            )[species]
            B = self.param(
                f"B_{i+1}",
                lambda key, shape: jnp.zeros(shape, dtype=x.dtype),
                (self.zmax + 2, d, self.ranks[i]),
            )[species]
            Ax = jnp.einsum("zrd,zd->zr", A, x)
            BAx = jnp.einsum("zrd,zd->zr", B, Ax)
            x = activation_from_str(self.activation)(xi + BAx)
        xi = nn.Dense(
            self.neurons[-1],
            use_bias=self.use_bias,
            name=f"Layer_{len(self.neurons)}",
            kernel_init=kernel_init,
        )(x)
        A = self.param(
            f"A_{len(self.neurons)}",
            lambda key, shape: jnp.zeros(shape, dtype=x.dtype),
            (self.zmax + 2, self.ranks[-1], x.shape[-1]),
        )[species]
        B = self.param(
            f"B_{len(self.neurons)}",
            lambda key, shape: jnp.zeros(shape, dtype=x.dtype),
            (self.zmax + 2, self.neurons[-1], self.ranks[-1]),
        )[species]
        Ax = jnp.einsum("zrd,zd->zr", A, x)
        BAx = jnp.einsum("zrd,zd->zr", B, Ax)
        x = xi + BAx
        if self.squeeze and x.shape[-1] == 1:
            x = jnp.squeeze(x, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: x} if output_key is not None else x
        return x


class BlockIndexNet(nn.Module):
    """Chemical-species-specific neural network using precomputed species index.

    FID: BLOCK_INDEX_NET

    A neural network that applies a species-specific fully connected network to each atom embedding.
    A species index must be provided to filter the embeddings for each species and apply the corresponding network.
    This index can be obtained using the SPECIES_INDEXER preprocessing module from `fennol.models.preprocessing.SpeciesIndexer`

    """

    output_dim: int
    """The dimension of the output of the fully connected networks."""
    hidden_neurons: Sequence[int]
    """The hidden dimensions of the fully connected networks.
        If a dictionary is provided, it should map species names to dimensions.
        If a sequence is provided, the same dimensions will be used for all species."""
    used_blocks: Optional[Sequence[str]] = None
    """The blocks to use. If None, all blocks will be used."""
    activation: Union[Callable, str] = "silu"
    """The activation function to use in the fully connected networks."""
    use_bias: bool = True
    """Whether to include bias terms in the fully connected networks."""
    input_key: Optional[str] = None
    """The key in the input dictionary that corresponds to the embeddings of the atoms."""
    block_index_key: str = "block_index"
    """The key in the input dictionary that corresponds to the block index of the atoms. See `fennol.models.preprocessing.BlockIndexer`"""
    output_key: Optional[str] = None
    """The key in the output dictionary that corresponds to the network's output."""

    squeeze: bool = False
    """Whether to remove the last axis of the output tensor if it is of dimension 1."""
    kernel_init: Union[str, Callable] = "lecun_normal()"
    """The kernel initialization method for the fully connected networks."""
    # check_unhandled: bool = True

    FID: ClassVar[str] = "BLOCK_INDEX_NET"

    # def setup(self):
    #     all_blocks = CHEMICAL_BLOCKS_NAMES
    #     if self.used_blocks is None:
    #         used_blocks = all_blocks
    #     else:
    #         used_blocks = []
    #         for b in self.used_blocks:
    #             b_=str(b).strip().upper()
    #             if b_ not in all_blocks:
    #                 raise ValueError(f"Block {b} not found in {all_blocks}")
    #             used_blocks.append(b_)
    #     used_blocks = set(used_blocks)
    #     self._used_blocks = used_blocks

    #     if not (
    #         isinstance(self.hidden_neurons, dict)
    #         or isinstance(self.hidden_neurons, FrozenDict)
    #     ):
    #         neurons = {k: self.hidden_neurons for k in used_blocks}
    #     else:
    #         neurons = {}
    #         for b in self.hidden_neurons.keys():
    #             b_=str(b).strip().upper()
    #             if b_ not in all_blocks:
    #                 raise ValueError(f"Block {b} does not exist.  Available blocks are {all_blocks}")
    #             neurons[b_] = self.hidden_neurons[b]
    #         used_blocks = set(neurons.keys())
    #         if used_blocks != self._used_blocks and self.used_blocks is not None:
    #             print(
    #                 f"Warning: hidden neurons definitions do not match specified used_blocks {self.used_blocks}. Using blocks defined in hidden_neurons.")
    #         self._used_blocks = used_blocks

    #     self.networks = {
    #         k: FullyConnectedNet(
    #             [*neurons[k], self.output_dim],
    #             self.activation,
    #             self.use_bias,
    #             name=k,
    #             kernel_init=self.kernel_init,
    #         )
    #         for k in self._used_blocks
    #     }

    @nn.compact
    def __call__(
        self, inputs: Union[dict, Tuple[jax.Array, jax.Array]]
    ) -> Union[dict, jax.Array]:

        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            species, embedding, block_index = inputs
        else:
            species, embedding = inputs["species"], inputs[self.input_key]
            block_index = inputs[self.block_index_key]

        assert isinstance(
            block_index, dict
        ), "block_index must be a dictionary for BlockIndexNet"

        networks = {
            k: FullyConnectedNet(
                [*self.hidden_neurons, self.output_dim],
                self.activation,
                self.use_bias,
                name=k,
                kernel_init=self.kernel_init,
            )
            for k in block_index.keys()
        }

        ############################
        # initialization => instantiate all networks
        if self.is_initializing():
            x = jnp.zeros((1, embedding.shape[-1]), dtype=embedding.dtype)
            [net(x) for net in networks.values()]

        # if self.check_unhandled:
        #     for b in block_index.keys():
        #         if b not in networks.keys():
        #             raise ValueError(f"Block {b} not found in networks. Available blocks are {self.networks.keys()}")

        ############################
        outputs = []
        indices = []
        for s, net in networks.items():
            if s not in block_index:
                continue
            if block_index[s] is None:
                continue
            idx = block_index[s]
            o = net(embedding[idx])
            outputs.append(o)
            indices.append(idx)

        o = jnp.concatenate(outputs, axis=0)
        idx = jnp.concatenate(indices, axis=0)

        out = (
            jnp.zeros((species.shape[0], *o.shape[1:]), dtype=o.dtype)
            .at[idx]
            .set(o, mode="drop")
        )

        if self.squeeze and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out