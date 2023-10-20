import flax.linen as nn
from typing import Any, Sequence, Callable, Union
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
from typing import Optional, Tuple, Dict


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

MISC = {
    "SCATTER_EDGES": ScatterEdges,
}