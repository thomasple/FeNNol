import flax.linen as nn
from typing import Sequence, Callable, Union
from ...utils.periodic_table import PERIODIC_TABLE
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
from typing import Optional, Tuple
from ...utils.activations import activation_from_str, TrainableSiLU
from ...utils.initializers import initializer_from_str, scaled_orthogonal


class FullyConnectedNet(nn.Module):
    """
    A fully connected neural network module.

    Args:
        neurons (Sequence[int]): A sequence of integers representing the dimensions of the network.
        activation (Callable, optional): The activation function to use. Defaults to nn.silu.
        use_bias (bool, optional): Whether to use bias in the dense layers. Defaults to True.
        input_key (Optional[str], optional): The key to use to access the input tensor. Defaults to None.
        output_key (Optional[str], optional): The key to use to access the output tensor. Defaults to None.
    """

    neurons: Sequence[int]
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = nn.linear.default_kernel_init

    FID: str = "NEURAL_NET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        """
        Applies the neural network to the given inputs.

        Args:
        - inputs: A dictionary or JAX array containing the inputs to the neural network.

        Returns:
        - A dictionary or JAX array containing the output of the neural network.
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
    """
    Residual neural network as defined in SpookyNet paper
    """

    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    kernel_init: Union[str, Callable] = scaled_orthogonal(mode="fan_avg")
    res_only: bool = False

    FID: str = "RES_MLP"

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
    """
    A fully connected neural network module with residual connections.

    Args:
        neurons (Sequence[int]): A sequence of integers representing the dimensions of the network.
        activation (Callable, optional): The activation function to use. Defaults to nn.silu.
        use_bias (bool, optional): Whether to use bias in the dense layers. Defaults to True.
        input_key (Optional[str], optional): The key to use to access the input tensor. Defaults to None.
        output_key (Optional[str], optional): The key to use to access the output tensor. Defaults to None.
    """

    dim: int
    output_dim: int
    nlayers: int
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = nn.linear.default_kernel_init

    FID: str = "SKIP_NET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        """
        Applies the neural network to the given inputs.

        Args:
        - inputs: A dictionary or JAX array containing the inputs to the neural network.

        Returns:
        - A dictionary or JAX array containing the output of the neural network.
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
    """
    Neural network for a sequence of inputs (in axis=-2) with a decay factor
    """

    neurons: Sequence[int]
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    decay: float = 0.01
    squeeze: bool = False
    kernel_init: Union[str, Callable] = nn.linear.default_kernel_init

    FID: str = "HIERARCHICAL_NET"

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


class ChemicalNetHet(nn.Module):
    """
    A neural network that applies a fully connected network to each atom in a chemical system,
    and selects the output corresponding to the atom's species.

    Args:
        species_order (Sequence[str]): The species for which to build a network.
        neurons (Union[dict, Sequence[int]]): The dimensions of the fully connected networks for each species.
            If a dictionary is provided, it should map species names to dimensions.
            If a sequence is provided, the same dimensions will be used for all species.
        activation (Callable, optional): The activation function to use in the fully connected networks. Defaults to nn.silu.
        use_bias (bool, optional): Whether to include bias terms in the fully connected networks. Defaults to True.
        input_key (str, optional): The key in the input dictionary that corresponds to the embeddings of the atoms.
            If None, the input is assumed to be a tuple (species,embeddings). Defaults to None.
        output_key (str, optional): The key in the output dictionary that corresponds to the network's output.
            If None, the output is returned directly. Defaults to None.
    """

    species_order: Sequence[str]
    neurons: Union[dict, Sequence[int]]
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = nn.linear.default_kernel_init

    FID: str = "CHEMICAL_NET_HET"

    def setup(self):
        idx_map = {s: i for i, s in enumerate(PERIODIC_TABLE)}
        if not isinstance(self.neurons, dict):
            neurons = {k: self.neurons for k in self.species_order}
        else:
            neurons = self.neurons
        self.networks = {
            idx_map[k]: FullyConnectedNet(
                neurons[k],
                self.activation,
                self.use_bias,
                name=k,
                kernel_init=self.kernel_init,
            )
            for k in self.species_order
        }

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
        # apply each network to all the atoms and select the output corresponding to the species
        # (brute force because shapes need to be fixed, coulb be improved by passing
        #   a fixed size list of atoms for each species)
        out = jnp.zeros((species.shape[0], self.neurons[-1]), dtype=embedding.dtype)
        for s, net in self.networks.items():
            out += jnp.where((species == s)[:, None], net(embedding), 0.0)
        if self.squeeze and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        ############################

        if self.input_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class ChemicalNet(nn.Module):
    """
    A neural network that applies a fully connected network to each atom in a chemical system,
    and selects the output corresponding to the atom's species.
    This is an optimized version of ChemicalNetHet that uses vmap to apply the networks to all atoms at once.
    The optimization is allowed because all networks have the same shape.

    Args:
        species_order (Sequence[str]): The species for which to build a network.
        neurons (Sequence[int]): The dimensions of the fully connected networks.
        activation (Callable, optional): The activation function to use in the fully connected networks. Defaults to nn.silu.
        use_bias (bool, optional): Whether to include bias terms in the fully connected networks. Defaults to True.
        input_key (str, optional): The key in the input dictionary that corresponds to the embeddings of the atoms.
            If None, the input is assumed to be a tuple (species,embeddings). Defaults to None.
        output_key (str, optional): The key in the output dictionary that corresponds to the network's output.
            If None, the output is returned directly. Defaults to None.
    """

    species_order: Sequence[str]
    neurons: Sequence[int]
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = nn.linear.default_kernel_init

    FID: str = "CHEMICAL_NET"

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
        rev_idx = {s: k for k, s in enumerate(PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        nspecies = len(self.species_order)
        conv_tensor_ = np.full((maxidx + 2,), -1, dtype=np.int32)
        for i, s in enumerate(self.species_order):
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
        x = jnp.repeat(embedding[None, :, :], len(self.species_order), axis=0)

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
    neurons: Sequence[int]
    num_networks: int
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = nn.linear.default_kernel_init
    router_key: Optional[str] = None

    FID: str = "MOE_NET"

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
        # repeat input along a new axis to compute for all species at once
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


class GatedPerceptron(nn.Module):
    dim: int
    use_bias: bool = True
    kernel_init: Union[str, Callable] = nn.linear.default_kernel_init
    activation: Union[Callable, str] = nn.silu
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False

    FID: str = "GATED_PERCEPTRON"

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

    Args:
        neurons (Sequence[int]): A sequence of integers representing the dimensions of the network.
        activation (Callable, optional): The activation function to use. Defaults to nn.silu.
        use_bias (bool, optional): Whether to use bias in the dense layers. Defaults to True.
        input_key (Optional[str], optional): The key to use to access the input tensor. Defaults to None.
        output_key (Optional[str], optional): The key to use to access the output tensor. Defaults to None.
    """

    neurons: Sequence[int]
    zmax: int = 86
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = nn.linear.default_kernel_init

    FID: str = "ZACNET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        """
        Applies the neural network to the given inputs.

        Args:
        - inputs: A dictionary or JAX array containing the inputs to the neural network.

        Returns:
        - A dictionary or JAX array containing the output of the neural network.
        """
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

    Args:
        neurons (Sequence[int]): A sequence of integers representing the dimensions of the network.
        activation (Callable, optional): The activation function to use. Defaults to nn.silu.
        use_bias (bool, optional): Whether to use bias in the dense layers. Defaults to True.
        input_key (Optional[str], optional): The key to use to access the input tensor. Defaults to None.
        output_key (Optional[str], optional): The key to use to access the output tensor. Defaults to None.
    """

    neurons: Sequence[int]
    ranks: Sequence[int]
    zmax: int = 86
    activation: Union[Callable, str] = nn.silu
    use_bias: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    squeeze: bool = False
    kernel_init: Union[str, Callable] = nn.linear.default_kernel_init

    FID: str = "ZLORANET"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        """
        Applies the neural network to the given inputs.

        Args:
        - inputs: A dictionary or JAX array containing the inputs to the neural network.

        Returns:
        - A dictionary or JAX array containing the output of the neural network.
        """
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
