import flax.linen as nn
from typing import Any, Sequence, Callable, Union, ClassVar, Optional, Dict, List
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
from ...utils.activations import activation_from_str
from ...utils.periodic_table import (
    CHEMICAL_PROPERTIES,
    PERIODIC_TABLE,
    PERIODIC_TABLE_REV_IDX,
)


def apply_switch(x: jax.Array, switch: jax.Array):
    """@private Multiply a switch array to an array of values."""
    shape = x.shape
    return (
        jnp.expand_dims(x, axis=-1).reshape(*switch.shape, -1) * switch[..., None]
    ).reshape(shape)


class ApplySwitch(nn.Module):
    """Multiply an edge array by a switch array.

    FID: APPLY_SWITCH
    """

    key: str
    """The key of the input array."""
    switch_key: Optional[str] = None
    """The key of the switch array."""
    graph_key: Optional[str] = None
    """The key of the graph containing the switch."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the input key is used."""

    FID: ClassVar[str] = "APPLY_SWITCH"

    @nn.compact
    def __call__(self, inputs) -> Any:
        if self.graph_key is not None:
            graph = inputs[self.graph_key]
            switch = graph["switch"]
        elif self.switch_key is not None:
            switch = inputs[self.switch_key]
        else:
            raise ValueError("Either graph_key or switch_key must be specified")

        x = inputs[self.key]
        output = apply_switch(x, switch)
        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: output}


class AtomToEdge(nn.Module):
    """Map atom-wise values to edge-wise values.

    FID: ATOM_TO_EDGE

    By default, we map the destination atom value to the edge. This can be changed by setting `use_source` to True.
    """

    _graphs_properties: Dict
    key: str
    """The key of the input atom-wise array."""
    output_key: Optional[str] = None
    """The key of the output edge-wise array. If None, the input key is used."""
    graph_key: str = "graph"
    """The key of the graph containing the edges."""
    switch: bool = False
    """Whether to apply a switch to the edge values."""
    switch_key: Optional[str] = None
    """The key of the switch array. If None, the switch is taken from the graph."""
    use_source: bool = False
    """Whether to use the source atom value instead of the destination atom value."""

    FID: ClassVar[str] = "ATOM_TO_EDGE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        graph = inputs[self.graph_key]
        nat = inputs["species"].shape[0]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]

        x = inputs[self.key]
        if self.use_source:
            x_edge = x[edge_src]
        else:
            x_edge = x[edge_dst]

        if self.switch:
            switch = (
                graph["switch"] if self.switch_key is None else inputs[self.switch_key]
            )
            x_edge = apply_switch(x_edge, switch)

        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: x_edge}


class ScatterEdges(nn.Module):
    """Reduce an edge array to atoms by summing over neighbors.

    FID: SCATTER_EDGES
    """

    _graphs_properties: Dict
    key: str
    """The key of the input edge-wise array."""
    output_key: Optional[str] = None
    """The key of the output atom-wise array. If None, the input key is used."""
    graph_key: str = "graph"
    """The key of the graph containing the edges."""
    switch: bool = False
    """Whether to apply a switch to the edge values before summing."""
    switch_key: Optional[str] = None
    """The key of the switch array. If None, the switch is taken from the graph."""
    antisymmetric: bool = False

    FID: ClassVar[str] = "SCATTER_EDGES"

    @nn.compact
    def __call__(self, inputs) -> Any:
        graph = inputs[self.graph_key]
        nat = inputs["species"].shape[0]
        x = inputs[self.key]

        if self.switch:
            switch = (
                graph["switch"] if self.switch_key is None else inputs[self.switch_key]
            )
            x = apply_switch(x, switch)

        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        output = jax.ops.segment_sum(
            x, edge_src, nat
        )  # jnp.zeros((nat, *x.shape[1:])).at[edge_src].add(x,mode="drop")
        if self.antisymmetric:
            output = output - jax.ops.segment_sum(x, edge_dst, nat)

        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: output}


class EdgeConcatenate(nn.Module):
    """Concatenate the source and destination atom values of an edge.

    FID: EDGE_CONCATENATE
    """

    _graphs_properties: Dict
    key: str
    """The key of the input atom-wise array."""
    output_key: Optional[str] = None
    """The key of the output edge-wise array. If None, the input key is used."""
    graph_key: str = "graph"
    """The key of the graph containing the edges."""
    switch: bool = False
    """Whether to apply a switch to the edge values."""
    switch_key: Optional[str] = None
    """The key of the switch array. If None, the switch is taken from the graph."""
    axis: int = -1
    """The axis along which to concatenate the atom values."""

    FID: ClassVar[str] = "EDGE_CONCATENATE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        nat = inputs["species"].shape[0]
        xi = inputs[self.key]

        assert self._graphs_properties[self.graph_key][
            "directed"
        ], "EdgeConcatenate only works for directed graphs"
        assert xi.shape[0] == nat, "Shape mismatch, xi.shape[0] != nat"

        xij = jnp.concatenate([xi[edge_src], xi[edge_dst]], axis=self.axis)

        if self.switch:
            switch = (
                graph["switch"] if self.switch_key is None else inputs[self.switch_key]
            )
            xij = apply_switch(xij, switch)

        output_key = self.name if self.output_key is None else self.output_key
        return {**inputs, output_key: xij}


class ScatterSystem(nn.Module):
    """Reduce an atom-wise array to a system-wise array by summing over atoms (in the batch).

    FID: SCATTER_SYSTEM
    """

    key: str
    """The key of the input atom-wise array."""
    output_key: Optional[str] = None
    """The key of the output system-wise array. If None, the input key is used."""
    average: bool = False
    """Wether to divide by the number of atoms in the system."""

    FID: ClassVar[str] = "SCATTER_SYSTEM"

    @nn.compact
    def __call__(self, inputs) -> Any:
        batch_index = inputs["batch_index"]
        x = inputs[self.key]
        assert (
            x.shape[0] == batch_index.shape[0]
        ), f"Shape mismatch {x.shape[0]} != {batch_index.shape[0]}"
        nsys = inputs["natoms"].shape[0]
        if self.average:
            shape = [batch_index.shape[0]] + (x.ndim - 1) * [1]
            x = x / inputs["natoms"][batch_index].reshape(shape)

        output = jax.ops.segment_sum(x, batch_index, nsys)

        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: output}


class SystemToAtoms(nn.Module):
    """Broadcast a system-wise array to an atom-wise array.

    FID: SYSTEM_TO_ATOMS
    """

    key: str
    """The key of the input system-wise array."""
    output_key: Optional[str] = None
    """The key of the output atom-wise array. If None, the input key is used."""

    FID: ClassVar[str] = "SYSTEM_TO_ATOMS"

    @nn.compact
    def __call__(self, inputs) -> Any:
        batch_index = inputs["batch_index"]
        x = inputs[self.key]
        output = x[batch_index]

        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: output}


class SumAxis(nn.Module):
    """Sum an array along an axis.

    FID: SUM_AXIS
    """

    key: str
    """The key of the input array."""
    axis: Union[None, int, Sequence[int]] = None
    """The axis along which to sum the array."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the input key is used."""
    norm: Optional[str] = None
    """Normalization of the sum. Can be 'dim', 'sqrt', or 'none'."""

    FID: ClassVar[str] = "SUM_AXIS"

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        output = jnp.sum(x, axis=self.axis)
        if self.norm is not None:
            norm = self.norm.lower()
            if norm == "dim":
                dim = np.prod(x.shape[self.axis])
                output = output / dim
            elif norm == "sqrt":
                dim = np.prod(x.shape[self.axis])
                output = output / dim**0.5
            elif norm == "none":
                pass
            else:
                raise ValueError(f"Unknown norm {norm}")
        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: output}


class Split(nn.Module):
    """Split an array along an axis.

    FID: SPLIT
    """

    key: str
    """The key of the input array."""
    output_keys: Sequence[str]
    """The keys of the output arrays."""
    axis: int = -1
    """The axis along which to split the array."""
    sizes: Union[int, Sequence[int]] = 1
    """The sizes of the splits."""
    squeeze: bool = True
    """Whether to remove the axis in the output if the size is 1."""

    FID: ClassVar[str] = "SPLIT"

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]

        if isinstance(self.sizes, int):
            split_size = [self.sizes] * len(self.output_keys)
        else:
            split_size = self.sizes
        if len(split_size) == len(self.output_keys):
            assert (
                sum(split_size) == x.shape[self.axis]
            ), f"Split sizes {split_size} do not match input shape"
            split_size = split_size[:-1]
        assert (
            len(split_size) == len(self.output_keys) - 1
        ), f"Wrong number of split sizes {split_size} for {len(self.output_keys)} outputs"
        split_indices = np.cumsum(split_size)
        outs = {}

        for k, v in zip(self.output_keys, jnp.split(x, split_indices, axis=self.axis)):
            outs[k] = (
                jnp.squeeze(v, axis=self.axis)
                if self.squeeze and v.shape[self.axis] == 1
                else v
            )

        return {**inputs, **outs}


class Concatenate(nn.Module):
    """Concatenate a list of arrays along an axis.

    FID: CONCATENATE
    """

    keys: Sequence[str]
    """The keys of the input arrays."""
    axis: int = -1
    """The axis along which to concatenate the arrays."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the name of the module is used."""

    FID: ClassVar[str] = "CONCATENATE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        output = jnp.concatenate([inputs[k] for k in self.keys], axis=self.axis)
        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class Activation(nn.Module):
    """Apply an element-wise activation function to an array.

    FID: ACTIVATION
    """

    key: str
    """The key of the input array."""
    activation: Union[Callable, str]
    """The activation function or its name."""
    scale_out: float = 1.0
    """Output scaling factor."""
    shift_out: float = 0.0
    """Output shift."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the input key is used."""

    FID: ClassVar[str] = "ACTIVATION"

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        activation = (
            activation_from_str(self.activation)
            if isinstance(self.activation, str)
            else self.activation
        )
        output = self.scale_out * activation(x) + self.shift_out
        output_key = self.output_key if self.output_key is not None else self.key
        return {**inputs, output_key: output}


class Scale(nn.Module):
    """Scale an array by a constant factor.

    FID: SCALE
    """

    key: str
    """The key of the input array."""
    scale: float
    """The (initial) scaling factor."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the input key is used."""
    trainable: bool = False
    """Whether the scaling factor is trainable."""

    FID: ClassVar[str] = "SCALE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]

        if self.trainable:
            scale = self.param("scale", lambda rng: jnp.asarray(self.scale))
        else:
            scale = self.scale

        output = scale * x
        output_key = self.output_key if self.output_key is not None else self.key
        return {**inputs, output_key: output}


class Add(nn.Module):
    """Add together a list of arrays.

    FID: ADD
    """

    keys: Sequence[str]
    """The keys of the input arrays."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the name of the module is used."""

    FID: ClassVar[str] = "ADD"

    @nn.compact
    def __call__(self, inputs) -> Any:
        output = 0
        for k in self.keys:
            output = output + inputs[k]

        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class Multiply(nn.Module):
    """Element-wise-multiply together a list of arrays.

    FID: MULTIPLY
    """

    keys: Sequence[str]
    """The keys of the input arrays."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the name of the module is used."""

    FID: ClassVar[str] = "MULTIPLY"

    @nn.compact
    def __call__(self, inputs) -> Any:
        output = 1
        for k in self.keys:
            output = output * inputs[k]

        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class Transpose(nn.Module):
    """Transpose an array.

    FID: TRANSPOSE
    """

    key: str
    """The key of the input array."""
    axes: Sequence[int]
    """The permutation of the axes. See `jax.numpy.transpose` for more details."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the input key is used."""

    FID: ClassVar[str] = "TRANSPOSE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        output = jnp.transpose(inputs[self.key], axes=self.axes)
        output_key = self.output_key if self.output_key is not None else self.key
        return {**inputs, output_key: output}


class Reshape(nn.Module):
    """Reshape an array.

    FID: RESHAPE
    """

    key: str
    """The key of the input array."""
    shape: Sequence[Union[int,str]]
    """The shape of the output array."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the input key is used."""

    FID: ClassVar[str] = "RESHAPE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        shape = []
        for s in self.shape:
            if isinstance(s,int):
                shape.append(s)
                continue

            if isinstance(s,str):
                s_=s.lower().strip()
                if s_ in ["natoms" ,"nat","natom","n_atoms","atoms"]:
                    shape.append(inputs["species"].shape[0])
                    continue
                
                if s_ in ["nsys","nbatch","nsystems","n_sys","n_systems","n_batch"]:
                    shape.append(inputs["natoms"].shape[0])
                    continue
                
                s_ = s.strip().split("[")
                key = s_[0]
                if key in inputs:
                    axis = int(s_[1].split("]")[0])
                    shape.append(inputs[key].shape[axis])
                    continue

            raise ValueError(f"Error parsing shape component {s}")

        output = jnp.reshape(inputs[self.key], shape)
        output_key = self.output_key if self.output_key is not None else self.key
        return {**inputs, output_key: output}


class ChemicalConstant(nn.Module):
    """Map atomic species to a constant value.

    FID: CHEMICAL_CONSTANT
    """

    value: Union[str, List[float], float, Dict]
    """The constant value or a dictionary of values for each element."""
    output_key: Optional[str] = None
    """The key of the output array. If None, the name of the module is used."""
    trainable: bool = False
    """Whether the constant is trainable."""

    FID: ClassVar[str] = "CHEMICAL_CONSTANT"

    @nn.compact
    def __call__(self, inputs) -> Any:
        if isinstance(self.value, str):
            constant = CHEMICAL_PROPERTIES[self.value.upper()]
        elif isinstance(self.value, list) or isinstance(self.value, tuple):
            constant = list(self.value)
        elif isinstance(self.value, float):
            constant = [self.value] * len(PERIODIC_TABLE)
        elif hasattr(self.value, "items"):
            constant = [0.0] * len(PERIODIC_TABLE)
            for k, v in self.value.items():
                constant[PERIODIC_TABLE_REV_IDX[k]] = v
        else:
            raise ValueError(f"Unknown constant type {type(self.value)}")

        if self.trainable:
            constant = self.param(
                "constant", lambda rng: jnp.asarray(constant, dtype=jnp.float32)
            )
        else:
            constant = jnp.asarray(constant, dtype=jnp.float32)
        output = constant[inputs["species"]]
        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class SwitchFunction(nn.Module):
    """Compute a switch array from an array of distances and a cutoff.

    FID: SWITCH_FUNCTION
    """

    cutoff: Optional[float] = None
    """The cutoff distance. If None, the cutoff is taken from the graph."""
    switch_start: float = 0.0
    """The proportion of the cutoff distance at which the switch function starts."""
    graph_key: Optional[str] = "graph"
    """The key of the graph containing the distances and edge mask."""
    output_key: Optional[str] = None
    """The key of the output switch array. If None, it is added to the graph."""
    switch_type: str = "cosine"
    """The type of switch function. Can be 'cosine', 'polynomial', or 'exponential'."""
    p: Optional[float] = None
    """ The parameter of the switch function. If None, it is fixed to the default for each `switch_type`."""
    trainable: bool = False
    """Whether the switch parameter is trainable."""

    FID: ClassVar[str] = "SWITCH_FUNCTION"

    @nn.compact
    def __call__(self, inputs) -> Any:
        if self.graph_key is not None:
            graph = inputs[self.graph_key]
            distances, edge_mask = graph["distances"], graph["edge_mask"]
            if self.cutoff is not None:
                edge_mask = jnp.logical_and(edge_mask, (distances < self.cutoff))
                cutoff = self.cutoff
            else:
                cutoff = graph["cutoff"]
        else:
            # distances = inputs
            if len(inputs) == 3:
                distances, edge_mask, cutoff = inputs
            else:
                distances, edge_mask = inputs
                assert (
                    self.cutoff is not None
                ), "cutoff must be specified if no graph is given"
                # edge_mask = distances < self.cutoff
                cutoff = self.cutoff

        if self.switch_start > 1.0e-5:
            assert (
                self.switch_start < 1.0
            ), "switch_start is a proportion of cutoff and must be smaller than 1."
            cutoff_in = self.switch_start * cutoff
            x = distances - cutoff_in
            end = cutoff - cutoff_in
        else:
            x = distances
            end = cutoff

        switch_type = self.switch_type.lower()
        if switch_type == "cosine":
            p = self.p if self.p is not None else 1.0
            if self.trainable:
                p = self.param("p", lambda rng: jnp.asarray(p, dtype=jnp.float32))
            switch = (0.5 * jnp.cos(x * (jnp.pi / end)) + 0.5) ** p

        elif switch_type == "polynomial":
            p = self.p if self.p is not None else 3.0
            if self.trainable:
                p = self.param("p", lambda rng: jnp.asarray(p, dtype=jnp.float32))
            d = x / end
            switch = (
                1.0
                - 0.5 * (p + 1) * (p + 2) * d**p
                + p * (p + 2) * d ** (p + 1)
                - 0.5 * p * (p + 1) * d ** (p + 2)
            )

        elif switch_type == "exponential":
            p = self.p if self.p is not None else 1.0
            if self.trainable:
                p = self.param("p", lambda rng: jnp.asarray(p, dtype=jnp.float32))
            r2 = x**2
            c2 = end**2
            switch = jnp.exp(-p * r2 / (c2 - r2))
        
        elif switch_type == "hard":
            switch = jnp.where(distances < cutoff, 1.0, 0.0)
        else:
            raise ValueError(f"Unknown switch function {switch_type}")

        if self.switch_start > 1.0e-5:
            switch = jnp.where(distances < cutoff_in, 1.0, switch)

        switch = jnp.where(edge_mask, switch, 0.0)

        if self.graph_key is not None:
            if self.output_key is not None:
                return {**inputs, self.output_key: switch}
            else:
                return {**inputs, self.graph_key: {**graph, "switch": switch}}
        else:
            return switch  # , edge_mask
