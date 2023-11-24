import flax.linen as nn
from typing import Any, Sequence, Callable, Union
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
from typing import Optional, Tuple, Dict
from ..utils.activations import activation_from_str


def apply_switch(x: jax.Array, switch: jax.Array):
    shape = x.shape
    return (
        jnp.expand_dims(x, axis=-1).reshape(shape[0], -1) * switch[:, None]
    ).reshape(shape)


class ScatterEdges(nn.Module):
    _graphs_properties: Dict
    key: str
    key_out: Optional[str] = None
    graph_key: str = "graph"
    switch: bool = False

    @nn.compact
    def __call__(self, inputs) -> Any:
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        nat = inputs["species"].shape[0]
        x = inputs[self.key]

        if self.switch:
            x = apply_switch(x, graph["switch"])

        if self._graphs_properties[self.graph_key]["directed"]:
            output = jnp.zeros((nat, *x.shape[1:])).at[edge_src].add(x)
        else:
            output = (
                jnp.zeros((nat, *x.shape[1:])).at[edge_src].add(x).at[edge_dst].add(x)
            )
        key_out = self.key if self.key_out is None else self.key_out
        return {**inputs, key_out: output}

class SumAxis(nn.Module):
    key: str
    axis: Union[None, int, Sequence[int]] = None
    key_out: Optional[str] = None

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        output = jnp.sum(x, axis=self.axis)
        key_out = self.key if self.key_out is None else self.key_out
        return {**inputs, key_out: output}

class Split(nn.Module):
    key: str
    keys_out: Sequence[str]
    axis: int = -1
    split_sizes:Union[int,Sequence[int]]=1
    squeeze: bool = True

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]

        if isinstance(self.split_sizes,int):
            split_size = [self.split_sizes]*len(self.keys_out)
        else:
            split_size = self.split_sizes
        if len(split_size) == len(self.keys_out):
            assert sum(split_size) == x.shape[self.axis], f"Split sizes {split_size} do not match input shape"
            split_size = split_size[:-1]
        assert len(split_size) == len(self.keys_out)-1, f"Wrong number of split sizes {split_size} for {len(self.keys_out)} outputs"
        split_indices = np.cumsum(split_size)
        outs={}
        
        for k,v in zip(self.keys_out,jnp.split(x,split_indices,axis=self.axis)):
            outs[k] = jnp.squeeze(v,axis=self.axis) if self.squeeze and v.shape[self.axis]==1 else v
        
        return {**inputs, **outs}

class Activation(nn.Module):
    key: str
    activation: Union[Callable, str] = nn.silu
    key_out: Optional[str] = None

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        activation = (
            activation_from_str(self.activation)
            if isinstance(self.activation, str)
            else self.activation
        )
        output = activation(x)
        key_out = self.key_out if self.key_out is not None else self.key
        return {**inputs, key_out: output}

class Scale(nn.Module):
    key: str
    scale: float
    key_out: Optional[str] = None

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        
        output = self.scale*x
        key_out = self.key_out if self.key_out is not None else self.key
        return {**inputs, key_out: output}
        


MISC = {
    "SCATTER_EDGES": ScatterEdges,
    "SUM_AXIS": SumAxis,
    "SPLIT": Split,
    "ACTIVATION": Activation,
    "SCALE": Scale,
}