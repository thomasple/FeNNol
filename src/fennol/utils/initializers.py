import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable,Union

def scaled_orthogonal(
    scale=1.0, mode="fan_avg", in_axis=-2, out_axis=-1, dtype=jnp.float_
):
    assert mode in ["fan_in", "fan_out", "fan_avg"]
    init_ortho = nn.initializers.orthogonal(
        scale=scale, column_axis=out_axis, dtype=dtype
    )
    if mode == "fan_in":

        def init(key, shape, dtype=jnp.float32):
            return init_ortho(key, shape, dtype=dtype) * (shape[in_axis] ** -0.5)

    elif mode == "fan_out":

        def init(key, shape, dtype=jnp.float32):
            return init_ortho(key, shape, dtype=dtype) * (shape[out_axis] ** -0.5)

    else:

        def init(key, shape, dtype=jnp.float32):
            return init_ortho(key, shape, dtype=dtype) * (
                (shape[in_axis] + shape[out_axis]) ** -0.5
            )

    return init


def initializer_from_str(name: Union[str,Callable])->Callable:
    if callable(name):
        return name
    if not isinstance(name, str):
        raise ValueError(f"Invalid initializer {name}")
    return eval(
        name,
        {"__builtins__": None},
        {
            **nn.initializers.__dict__,
            "scaled_orthogonal": scaled_orthogonal,
        },
    )
