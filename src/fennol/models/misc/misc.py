import flax.linen as nn
from typing import Any, Sequence, Callable, Union
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
from typing import Optional, Tuple, Dict, List, Union
from ...utils.activations import activation_from_str
from ...utils.periodic_table import (
    CHEMICAL_PROPERTIES,
    PERIODIC_TABLE,
    PERIODIC_TABLE_REV_IDX,
)


def apply_switch(x: jax.Array, switch: jax.Array):
    shape = x.shape
    return (
        jnp.expand_dims(x, axis=-1).reshape(*switch.shape, -1) * switch[..., None]
    ).reshape(shape)


class ApplySwitch(nn.Module):
    key: str
    switch_key: Optional[str] = None
    graph_key: Optional[str] = None
    output_key: Optional[str] = None

    FID: str = "APPLY_SWITCH"

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


class ScatterEdges(nn.Module):
    _graphs_properties: Dict
    key: str
    output_key: Optional[str] = None
    graph_key: str = "graph"
    switch: bool = False
    switch_key: Optional[str] = None

    FID: str = "SCATTER_EDGES"

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
        output = jax.ops.segment_sum(x,edge_src,nat) #jnp.zeros((nat, *x.shape[1:])).at[edge_src].add(x,mode="drop")
        if not self._graphs_properties[self.graph_key]["directed"]:
            output = output + jax.ops.segment_sum(x,edge_dst,nat)

        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: output}

class EdgeConcatenate(nn.Module):
    _graphs_properties: Dict
    key: str
    output_key: Optional[str] = None
    graph_key: str = "graph"
    switch: bool = False
    switch_key: Optional[str] = None
    axis: int = -1
    
    FID: str = "EDGE_CONCATENATE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        nat = inputs["species"].shape[0]
        xi = inputs[self.key]

        assert self._graphs_properties[self.graph_key]["directed"], "EdgeConcatenate only works for directed graphs"
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
    key: str
    output_key: Optional[str] = None

    FID: str = "SCATTER_SYSTEM"

    @nn.compact
    def __call__(self, inputs) -> Any:
        batch_index = inputs["batch_index"]
        x = inputs[self.key]
        assert (
            x.shape[0] == batch_index.shape[0]
        ), f"Shape mismatch {x.shape[0]} != {batch_index.shape[0]}"
        nsys = inputs["natoms"].shape[0]
        output = jax.ops.segment_sum(x, batch_index, nsys)

        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: output}


class SumAxis(nn.Module):
    key: str
    axis: Union[None, int, Sequence[int]] = None
    output_key: Optional[str] = None
    norm: Optional[str] = None

    FID: str = "SUM_AXIS"

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        output = jnp.sum(x, axis=self.axis)
        if self.norm is not None:
            norm=self.norm.lower()
            if norm == "dim":
                output = output / x.shape[self.axis]
            elif norm == "sqrt":
                output = output / np.sqrt(x.shape[self.axis])
            elif norm == "none":
                pass
            else:
                raise ValueError(f"Unknown norm {norm}")
        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: output}


class Split(nn.Module):
    key: str
    output_keys: Sequence[str]
    axis: int = -1
    sizes: Union[int, Sequence[int]] = 1
    squeeze: bool = True

    FID: str = "SPLIT"

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
    keys: Sequence[str]
    axis: int = -1
    output_key: Optional[str] = None

    FID: str = "CONCATENATE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        output = jnp.concatenate([inputs[k] for k in self.keys], axis=self.axis)
        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class Activation(nn.Module):
    key: str
    activation: Union[Callable, str]
    scale_out: float = 1.0
    shift_out: float = 0.0
    output_key: Optional[str] = None

    FID: str = "ACTIVATION"

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
    key: str
    scale: float
    output_key: Optional[str] = None
    trainable: bool = False

    FID: str = "SCALE"

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
    keys: Sequence[str]
    output_key: Optional[str] = None

    FID: str = "ADD"

    @nn.compact
    def __call__(self, inputs) -> Any:
        output = 0
        for k in self.keys:
            output = output + inputs[k]

        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class Multiply(nn.Module):
    keys: Sequence[str]
    output_key: Optional[str] = None

    FID: str = "MULTIPLY"

    @nn.compact
    def __call__(self, inputs) -> Any:
        output = 1
        for k in self.keys:
            output = output * inputs[k]

        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}

class Transpose(nn.Module):
    key: str
    axes: Sequence[int] 
    output_key: Optional[str] = None

    FID: str = "TRANSPOSE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        output = jnp.transpose(inputs[self.key], axes=self.axes)
        output_key = self.output_key if self.output_key is not None else self.key
        return {**inputs, output_key: output}

class Reshape(nn.Module):
    key: str
    shape: Sequence[int]
    output_key: Optional[str] = None

    FID: str = "RESHAPE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        output = jnp.reshape(inputs[self.key], self.shape)
        output_key = self.output_key if self.output_key is not None else self.key
        return {**inputs, output_key: output}

class ChemicalConstant(nn.Module):
    value: Union[str, List[float], float, Dict]
    output_key: Optional[str] = None
    trainable: bool = False

    FID: str = "CHEMICAL_CONSTANT"


    @nn.compact
    def __call__(self, inputs) -> Any:
        if isinstance(self.value, str):
            constant = CHEMICAL_PROPERTIES[self.value.upper()]
        elif isinstance(self.value, list):
            constant = self.value
        elif isinstance(self.value, float):
            constant = [self.value] * len(PERIODIC_TABLE)
        elif hasattr(self.value,'items' ): 
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
    cutoff: Optional[float] = None
    switch_start: float = 0.0
    graph_key: Optional[str] = "graph"
    output_key: Optional[str] = None
    switch_type: str = "cosine"
    p: Optional[float] = None
    trainable: bool = False

    FID: str = "SWITCH_FUNCTION"


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
            return switch #, edge_mask
