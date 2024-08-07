import jax
import jax.numpy as jnp
from functools import partial
import math
import flax.linen as nn
from typing import Union, Callable


class TrainableSiLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        a = self.param("alpha", lambda k, s: jnp.ones(s), (1, x.shape[-1]))
        b = self.param("beta", lambda k, s: 1.702 * jnp.ones(s), (1, x.shape[-1]))
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        return (a * x * jax.nn.sigmoid(b * x)).reshape(shape)


class TrainableCELU(nn.Module):
    alpha: float = 0.1

    @nn.compact
    def __call__(self, x):
        a = self.alpha * (
            1
            + jax.nn.celu(
                self.param(
                    "alpha",
                    lambda k, s: jnp.zeros(s),
                    (
                        1,
                        x.shape[-1],
                    ),
                ),
                alpha=1.0,
            )
        )
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        return jax.nn.celu(x, a).reshape(shape)


class TrainableLeakyCELU(nn.Module):
    alpha: float = 0.05
    beta: float = 1.0

    @nn.compact
    def __call__(self, x):
        a = self.alpha + self.param(
            "alpha",
            lambda k, s: jnp.zeros(s),
            (
                1,
                x.shape[-1],
            ),
        )
        b = self.beta * (
            1
            + jax.nn.celu(
                self.param(
                    "beta",
                    lambda k, s: jnp.zeros(s),
                    (
                        1,
                        x.shape[-1],
                    ),
                ),
                alpha=1.0,
            )
        )
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        return leaky_celu(x, a, b).reshape(shape)


@jax.jit
def aptx(x):
    return (1.0 + jax.nn.tanh(x)) * x


@partial(jax.jit, static_argnums=(1,))
def leaky_celu(x, alpha=0.1, beta=1.0):
    return alpha * x + ((1.0 - alpha) / beta) * (
        jax.nn.softplus(beta * x) - math.log(2.0)
    )


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
def tssr3(x):
    ax = jnp.abs(x)
    mask = ax <= 1.0
    axx = jnp.where(mask, 1.0, ax)
    ax2 = ax * ax
    dax2 = ax - ax2
    poly = 2.1875 * dax2 + ax2 * (ax + 0.3125 * dax2)
    return jnp.sign(x) * jnp.where(mask, poly, axx**0.5)


@jax.jit
def pow(x, a):
    return x**a


@jax.jit
def ssp(x):
    return jnp.logaddexp(x + math.log(0.5), math.log(0.5))


@jax.jit
def smooth_floor(x, eps=0.99):
    return (
        x
        - 0.5
        - jnp.atan(
            -eps * jnp.sin((-2 * jnp.pi) * x) / (eps * jnp.cos((2 * jnp.pi) * x) - 1.0)
        )
        / jnp.pi
    )


@jax.jit
def smooth_round(x, eps=0.99):
    return (
        x
        - jnp.atan(
            -eps
            * jnp.sin(-2 * jnp.pi * (x - 0.5))
            / (eps * jnp.cos(2 * jnp.pi * (x - 0.5)) - 1.0)
        )
        / jnp.pi
    )


def chain(*activations):
    # @jax.jit
    def act(x):
        for a in activations:
            x = a(x)
        return x

    return act


def activation_from_str(activation: Union[str, Callable, None]) -> Callable:
    if activation is None:
        return lambda x: x
    if callable(activation):
        return activation
    if not isinstance(activation, str):
        raise ValueError(f"Invalid activation {activation}")
    if activation.lower() in ["none", "linear", "identity"]:
        return lambda x: x
    try:
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
                "tssr3": tssr3,
                "ssp": ssp,
                "smooth_floor": smooth_floor,
                "smooth_round": smooth_round,
                "TrainableSiLU": TrainableSiLU,
                "TrainableCELU": TrainableCELU,
            },
        )
    except Exception as e:
        raise ValueError(
            f"The following exception was raised while parsing the activation function {activation} : {e}"
        )
