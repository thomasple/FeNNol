import jax
import jax.numpy as jnp
from functools import partial
import math
import flax.linen as nn
from typing import Union,Callable


class TrainableSiLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        a = self.param("alpha", lambda k, s: jnp.ones(s), (1,x.shape[-1]))
        b = self.param("beta", lambda k, s: 1.702 * jnp.ones(s), (1,x.shape[-1]))
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        return (a * x * jax.nn.sigmoid(b * x)).reshape(shape)


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


@jax.jit
def pow(x, a):
    return x**a


def chain(*activations):
    @jax.jit
    def act(x):
        for a in activations:
            x = a(x)
        return x

    return act


def activation_from_str(activation: Union[str,Callable,None])->Callable:
    if activation is None:
        return lambda x: x
    if not isinstance(activation, str):
        return activation
    if activation.lower() in ["none" ,"linear","identity"]:
        return lambda x: x
    return eval(
        activation,
        {"__builtins__": None},
        {
            **jax.nn.__dict__,
            **jax.numpy.__dict__,
            **jax.__dict__,
            "chain": chain,
            "pow": pow,
            "partial": partial,
            "leaky_celu": leaky_celu,
            "aptx": aptx,
            "tssr": tssr,
            "tssr2": tssr2,
        },
    )
