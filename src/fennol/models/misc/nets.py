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

    Parameters:
        neurons (Sequence[int]): A sequence of integers representing the dimensions of the network.
        activation (Union[Callable, str], optional): The activation function to use. Defaults to nn.silu.
        use_bias (bool, optional): Whether to use bias in the dense layers. Defaults to True.
        input_key (Optional[str], optional): The key to use to access the input tensor. Defaults to None.
        output_key (Optional[str], optional): The key to use to access the output tensor. Defaults to None.
        squeeze (bool, optional): Whether to squeeze the output tensor if its shape is (..., 1). Defaults to False.
        kernel_init (Union[str, Callable], optional): The kernel initialization method to use. Defaults to nn.linear.default_kernel_init.

    """

    neurons: Sequence[int]
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = "lecun_normal()"

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

        activation = (
            activation_from_str(self.activation)
            if isinstance(self.activation, str)
            else self.activation
        )
        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )
        ############################
        for i, d in enumerate(self.neurons[:-1]):
            x = nn.Dense(
                d,
                use_bias=self.use_bias,
                name=f"Layer_{i+1}",
                kernel_init=kernel_init,
                # precision=jax.lax.Precision.HIGH,
            )(x)
            x = activation(x)
        x = nn.Dense(
            self.neurons[-1],
            use_bias=self.use_bias,
            name=f"Layer_{len(self.neurons)}",
            kernel_init=kernel_init,
            # precision=jax.lax.Precision.HIGH,
        )(x)
        if self.squeeze and x.shape[-1] == 1:
            x = jnp.squeeze(x, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: x} if output_key is not None else x
        return x


class ResMLP(nn.Module):
    """Residual neural network as defined in SpookyNet paper

    Parameters:
        use_bias (bool): Whether to include bias in the linear layers. Default is True.
        input_key (Optional[str]): Key to access the input data from a dictionary. If None, assumes inputs are not a dictionary. Default is None.
        output_key (Optional[str]): Key to store the output data in the dictionary. If None, uses the name of the module. Default is None.
        kernel_init (Union[str, Callable]): Initialization method for the kernel weights. Can be a string representing a built-in initializer or a custom initialization function. Default is scaled_orthogonal(mode="fan_avg").
        res_only (bool): Whether to only apply the residual connection without additional linear layers. Default is False.

    """

    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    kernel_init: Union[str, Callable] = "scaled_orthogonal(mode='fan_avg')"
    res_only: bool = False

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

    Parameters:
        dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output layer.
        nlayers (int): The number of layers in the network.
        activation (Union[Callable, str]): The activation function to use. Can be a callable or a string representing the name of the activation function.
        use_bias (bool): Whether to include bias terms in the linear layers.
        input_key (Optional[str]): The key to access the input data from a dictionary, if the input is a dictionary.
        output_key (Optional[str]): The key to store the output data in a dictionary, if the output is a dictionary.
        squeeze (bool): Whether to squeeze the output if it has shape (..., 1).
        kernel_init (Union[str, Callable]): The initializer for the linear layer weights. Can be a callable or a string representing the name of the initializer.

    """

    dim: int
    output_dim: int
    nlayers: int
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = "lecun_normal()"

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

        activation = (
            activation_from_str(self.activation)
            if isinstance(self.activation, str)
            else self.activation
        )
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
            x = x + activation(
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

    Parameters:
        neurons (Sequence[int]): A sequence of integers representing the number of neurons in each layer.
        activation (Union[Callable, str], optional): Activation function to use. Defaults to nn.silu.
        use_bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
        input_key (Optional[str], optional): Key to access the input data if it is a dictionary. Defaults to None.
        output_key (Optional[str], optional): Key to store the output data if it is a dictionary. Defaults to None.
        decay (float, optional): Decay factor to scale each layer. Defaults to 0.01.
        squeeze (bool, optional): Whether to squeeze the output if it has shape (..., 1). Defaults to False.
        kernel_init (Union[str, Callable], optional): Initialization method for the linear layers. Defaults to nn.linear.default_kernel_init.

    """

    neurons: Sequence[int]
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    decay: float = 0.01
    squeeze: bool = False
    kernel_init: Union[str, Callable] = "lecun_normal()"

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

    A neural network that applies a species-specific fully connected network to each atom embedding.
    A species index must be provided to filter the embeddings for each species and apply the corresponding network.
    This index can be obtained using the SPECIES_INDEXER preprocessing module.

    Parameters:
        output_dim (int): The dimension of the output of the fully connected networks. It is the same for all species.
        hidden_neurons (Union[dict, Sequence[int]]): The hidden dimensions of the fully connected networks for each species.
            If a dictionary is provided, it should map species names to dimensions.
            If a sequence is provided, the same dimensions will be used for all species.
        species_order (Sequence[str]): The species for which to build a network. Only required if neurons is not a dictionary.
        activation (Callable, optional): The activation function to use in the fully connected networks. Defaults to nn.silu.
        use_bias (bool, optional): Whether to include bias terms in the fully connected networks. Defaults to True.
        input_key (str, optional): The key in the input dictionary that corresponds to the embeddings of the atoms.
            If None, the input is assumed to be a tuple (species,embeddings). Defaults to None.
        species_index_key str: The key in the input dictionary that corresponds to the species index of the atoms.
            This index can be obtained via the SPECIES_INDEXER preprocessing module. Defaults to "species_index".
        output_key (str, optional): The key in the output dictionary that corresponds to the network's output.
            If None, the output is returned directly. Defaults to None.
        squeeze (bool, optional): Whether to squeeze the output if it has shape (batch_size, 1). Defaults to False.
        kernel_init (Union[str, Callable], optional): The kernel initialization method for the fully connected networks. Defaults to nn.linear.default_kernel_init.
    """

    output_dim: int
    hidden_neurons: Union[dict, FrozenDict, Sequence[int]]
    species_order: None | str | Sequence[str] = None
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    species_index_key: str = "species_index"
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = "lecun_normal()"

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

    A neural network that applies a fully connected network to each atom embedding in a chemical system and selects the output corresponding to the atom's species.
    This is an optimized version of ChemicalNetHet that uses vmap to apply the networks to all atoms at once.
    The optimization is allowed because all networks have the same shape.

    Parameters:
        species_order (Sequence[str]): The species for which to build a network.
        neurons (Sequence[int]): The dimensions of the fully connected networks.
        activation (Callable, optional): The activation function to use in the fully connected networks. Defaults to nn.silu.
        use_bias (bool, optional): Whether to include bias terms in the fully connected networks. Defaults to True.
        input_key (str, optional): The key in the input dictionary that corresponds to the embeddings of the atoms.
            If None, the input is assumed to be a tuple (species,embeddings). Defaults to None.
        output_key (str, optional): The key in the output dictionary that corresponds to the network's output.
            If None, the output is returned directly. Defaults to None.
        squeeze (bool, optional): Whether to squeeze the output if it has shape (batch_size, 1). Defaults to False.
        kernel_init (Union[str, Callable], optional): The kernel initialization method for the fully connected networks. Defaults to nn.linear.default_kernel_init.
    """

    species_order: str | Sequence[str]
    neurons: Sequence[int]
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = "lecun_normal()"

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

    This class represents a Mixture of Experts neural network. It takes in an input and applies a set of shape-sharing networks
    to the input based on a router. The outputs of the shape-sharing networks are then combined using weights computed by the router.

    Parameters:
        neurons (Sequence[int]): A sequence of integers representing the number of neurons in each shape-sharing network.
        num_networks (int): The number of shape-sharing networks to create.
        activation (Union[Callable, str], optional): The activation function to use in the shape-sharing networks. Defaults to nn.silu.
        use_bias (bool, optional): Whether to include bias in the shape-sharing networks. Defaults to True.
        input_key (Optional[str], optional): The key to access the input from the inputs dictionary. If None, the input is assumed to be the same as the router. Defaults to None.
        output_key (Optional[str], optional): The key to store the output in the outputs dictionary. If None, the output is stored with the name of the MOENet instance. Defaults to None.
        squeeze (bool, optional): Whether to squeeze the output if it has a shape of (batch_size, 1). Defaults to False.
        kernel_init (Union[str, Callable], optional): The kernel initialization function to use in the shape-sharing networks. Defaults to nn.linear.default_kernel_init.
        router_key (Optional[str], optional): The key to access the router from the inputs dictionary. If None, the router is assumed to be the same as the input. Defaults to None.

    """

    neurons: Sequence[int]
    num_networks: int
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = "lecun_normal()"
    router_key: Optional[str] = None

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
    """Apply a neural network to each channel.

    Parameters:
        neurons (Sequence[int]): A sequence of integers representing the number of neurons in each shape-sharing network.
        num_networks (int): The number of shape-sharing networks to create.
        activation (Union[Callable, str], optional): The activation function to use in the shape-sharing networks. Defaults to nn.silu.
        use_bias (bool, optional): Whether to include bias in the shape-sharing networks. Defaults to True.
        input_key (Optional[str], optional): The key to access the input from the inputs dictionary. If None, the input is assumed to be the same as the router. Defaults to None.
        output_key (Optional[str], optional): The key to store the output in the outputs dictionary. If None, the output is stored with the name of the MOENet instance. Defaults to None.
        squeeze (bool, optional): Whether to squeeze the output if it has a shape of (batch_size, 1). Defaults to False.
        kernel_init (Union[str, Callable], optional): The kernel initialization function to use in the shape-sharing networks. Defaults to nn.linear.default_kernel_init.
        router_key (Optional[str], optional): The key to access the router from the inputs dictionary. If None, the router is assumed to be the same as the input. Defaults to None.

    """

    neurons: Sequence[int]
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = "lecun_normal()"
    channel_axis: int = -2

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

    This class represents a Gated Perceptron neural network model. It applies a gating mechanism
    to the input data and performs linear transformation using a dense layer followed by an activation function.

    Parameters:
        dim (int): The dimensionality of the output space.
        use_bias (bool, optional): Whether to include a bias term in the dense layer. Defaults to True.
        kernel_init (Union[str, Callable], optional): The initializer for the kernel weights of the dense layer.
            It can be a string representing a predefined initializer or a custom initializer function.
            Defaults to nn.linear.default_kernel_init.
        activation (Union[Callable, str], optional): The activation function to be applied after the dense layer.
            It can be a string representing a predefined activation function or a custom activation function.
            Defaults to nn.silu.
        input_key (Optional[str], optional): The key to access the input data if it is a dictionary.
            If None, assumes the input data is not a dictionary. Defaults to None.
        output_key (Optional[str], optional): The key to store the output data in the dictionary if input_key is not None.
            If None, uses the name of the module as the output key. Defaults to None.
        squeeze (bool, optional): Whether to squeeze the output tensor if its last dimension is 1. Defaults to False.

    """

    dim: int
    use_bias: bool = True
    kernel_init: Union[str, Callable] = "lecun_normal()"
    activation: Union[Callable, str] = nn.silu
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False

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

        activation = (
            activation_from_str(self.activation)
            if isinstance(self.activation, str)
            else self.activation
        )
        kernel_init = (
            initializer_from_str(self.kernel_init)
            if isinstance(self.kernel_init, str)
            else self.kernel_init
        )
        ############################
        gate = jax.nn.sigmoid(
            nn.Dense(self.dim, use_bias=self.use_bias, kernel_init=kernel_init)(x)
        )
        x = gate * activation(
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
    """
    A fully connected neural network module with affine Z-dependent adjustments of activations.

    Parameters:
        neurons (Sequence[int]): A sequence of integers representing the dimensions of the network.
        zmax (int): The maximum atomic number to consider.
        activation (Union[Callable, str], optional): The activation function to use. Defaults to nn.silu.
        use_bias (bool, optional): Whether to use bias in the dense layers. Defaults to True.
        input_key (Optional[str], optional): The key to use to access the input tensor. Defaults to None.
        output_key (Optional[str], optional): The key to use to access the output tensor. Defaults to None.
        squeeze (bool, optional): Whether to squeeze the output tensor if its shape is (..., 1). Defaults to False.
        kernel_init (Union[str, Callable], optional): The kernel initialization method to use. Defaults to nn.linear.default_kernel_init.
    """

    neurons: Sequence[int]
    zmax: int = 86
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = "lecun_normal()"

    FID: ClassVar[str] = "ZACNET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            species, x = inputs
        else:
            species, x = inputs["species"], inputs[self.input_key]

        activation = (
            activation_from_str(self.activation)
            if isinstance(self.activation, str)
            else self.activation
        )
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
            x = activation(sig * x + b)
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
    """
    A fully connected neural network module with Z-dependent low-rank adaptation.

    Parameters:
        neurons (Sequence[int]): A sequence of integers representing the dimensions of the network.
        ranks (Sequence[int]): A sequence of integers representing the ranks of the low-rank adaptation matrices.
        zmax (int): The maximum atomic number to consider.
        activation (Union[Callable, str], optional): The activation function to use. Defaults to nn.silu.
        use_bias (bool, optional): Whether to use bias in the dense layers. Defaults to True.
        input_key (Optional[str], optional): The key to use to access the input tensor. Defaults to None.
        output_key (Optional[str], optional): The key to use to access the output tensor. Defaults to None.
        squeeze (bool, optional): Whether to squeeze the output tensor if its shape is (..., 1). Defaults to False.
        kernel_init (Union[str, Callable], optional): The kernel initialization method to use. Defaults to nn.linear.default_kernel_init.
    """

    neurons: Sequence[int]
    ranks: Sequence[int]
    zmax: int = 86
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = "lecun_normal()"

    FID: ClassVar[str] = "ZLORANET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.input_key is None:
            assert not isinstance(
                inputs, dict
            ), "input key must be provided if inputs is a dictionary"
            species, x = inputs
        else:
            species, x = inputs["species"], inputs[self.input_key]

        activation = (
            activation_from_str(self.activation)
            if isinstance(self.activation, str)
            else self.activation
        )
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
            x = activation(xi + BAx)
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
