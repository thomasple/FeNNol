import jax
import jax.numpy as jnp
from functools import partial
import math


@jax.jit
def aptx(x):
    return (1.0 + jax.nn.tanh(x)) * x


@partial(jax.jit, static_argnums=(1,))
def leaky_celu(x, alpha=0.1):
    return alpha * x + (1.0 - alpha) * (jax.nn.softplus(x) - math.log(2.0))


@jax.jit
def tssr(x):
    ax = jnp.abs(x)
    mask = ax <= 1.0
    axx = jnp.where(mask, 1.0, ax)
    return jnp.where(mask, x, jnp.sign(x) * (2 * axx**0.5 - 1))


@jax.jit
def tssr2(x):
    ax = jnp.abs(x)
    mask = ax <= 1.0
    axx = jnp.where(mask, 1.0, ax)
    return jnp.sign(x) * jnp.where(mask, 1.25 * ax - 0.25 * ax**3, axx**0.5)


def activation_from_str(activation: str):
    if activation is None or activation.lower() == "none":
        return lambda x: x
    return eval(
        activation,
        {"__builtins__": None},
        {
            **jax.nn.__dict__,
            "partial": partial,
            "leaky_celu": leaky_celu,
            "aptx": aptx,
            "tssr": tssr,
            "tssr2": tssr2,
        },
    )
