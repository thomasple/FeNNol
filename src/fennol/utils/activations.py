import jax
import jax.numpy as jnp
from functools import partial
import math
import flax.linen as nn
import numpy as np
from typing import Union, Callable


class TrainableSiLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        a = self.param("alpha", lambda k, s: jnp.ones(s), (1, x.shape[-1]))
        b = self.param("beta", lambda k, s: 1.702 * jnp.ones(s), (1, x.shape[-1]))
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        return (a * x * jax.nn.sigmoid(b * x)).reshape(shape)

class DyT(nn.Module):
    alpha0: float = 0.5

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        dim = shape[-1]
        a = self.param("scale", lambda k, s: jnp.ones(s,dtype=x.dtype), (1, dim))
        b = self.param("bias", lambda k, s: jnp.zeros(s,dtype=x.dtype), (1, dim))
        alpha = self.param("alpha", lambda k: jnp.asarray(self.alpha0,dtype=x.dtype))
        x = x.reshape(-1, dim)
        return (a*jnp.tanh(alpha*x) + b).reshape(shape)

class DynAct(nn.Module):
    activation : Union[str, Callable]
    alpha0: float = 1.
    channelwise: bool = False

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        dim = shape[-1]
        a = self.param("scale", lambda k, s: jnp.ones(s,dtype=x.dtype), (1, dim))
        b = self.param("bias", lambda k, s: jnp.zeros(s,dtype=x.dtype), (1, dim))

        shape_alpha = (1,dim) if self.channelwise else (1,1)
        alpha = self.param("alpha", lambda k,s: self.alpha0*jnp.ones(s,dtype=x.dtype), shape_alpha)

        x = x.reshape(-1, dim)
        act = activation_from_str(self.activation)
        return (a*act(alpha*x) + b).reshape(shape)

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


class FourierActivation(nn.Module):
    nmax: int = 4

    @nn.compact
    def __call__(self, x):

        out = jax.nn.swish(x)
        nfeatures = x.shape[-1]
        n = 2 * np.pi * np.arange(1, self.nmax + 1)
        shape = [1] * x.ndim + [self.nmax]
        n = n.reshape(shape)
        x = jnp.expand_dims(x, axis=-1)
        cx = jnp.cos(n * x)
        sx = jnp.sin(n * x)
        x = jnp.concatenate((cx, sx), axis=-1)

        shape = [1] * (out.ndim - 1) + [nfeatures]
        b = self.param("b", jax.nn.initializers.zeros, shape)

        w = self.param(
            "w", jax.nn.initializers.normal(stddev=0.1), (*shape, 2 * self.nmax)
        )

        return out + (x * w).sum(axis=-1) + b


class GausianBasisActivation(nn.Module):
    nbasis: int = 10
    xmin: float = -1.0
    xmax: float = 3.0

    @nn.compact
    def __call__(self, x):

        nfeatures = x.shape[-1]
        shape = [1] * (x.ndim - 1) + [nfeatures]
        b = self.param("b", jax.nn.initializers.zeros, shape)
        w0 = self.param("w0", jax.nn.initializers.ones, shape)
        out = jax.nn.swish(w0*x)

        shape+= [self.nbasis]
        x0 = self.param(
            "x0",
            lambda key: jnp.asarray(
                np.linspace(self.xmin, self.xmax, self.nbasis)[None, :]
                .repeat(nfeatures, axis=0)
                .reshape(shape),
                dtype=x.dtype,
            ),
        )

        sigma0 = np.abs(self.xmax - self.xmin)/self.nbasis
        alphas = (0.5**0.5)*self.param(
            "sigmas",
            lambda key: jnp.full(shape, 1./sigma0, dtype=x.dtype),
        )

        w = self.param(
            "w", jax.nn.initializers.normal(stddev=0.1), shape
        )

        ex = jnp.exp(-((x[...,None] - x0) * alphas)**2)

        return out + (ex * w).sum(axis=-1) + b

def safe_sqrt(x,eps=1.e-5):
    return jnp.sqrt(jnp.clip(x, min=eps))

@jax.jit
def aptx(x):
    return (1.0 + jax.nn.tanh(x)) * x


@jax.jit
def serf(x):
    return x * jax.scipy.special.erf(jax.nn.softplus(x))


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


def normalize_activation(
    phi: Callable[[float], float], return_scale=False
) -> Callable[[float], float]:
    r"""Normalize a function, :math:`\psi(x)=\phi(x)/c` where :math:`c` is the normalization constant such that

    .. math::

        \int_{-\infty}^{\infty} \psi(x)^2 \frac{e^{-x^2/2}}{\sqrt{2\pi}} dx = 1

    ! Adapted from e3nn_jax !
    """
    with jax.ensure_compile_time_eval():
        # k = jax.random.PRNGKey(0)
        # x = jax.random.normal(k, (1_000_000,))
        n = 1_000_001
        x = jnp.sqrt(2) * jax.scipy.special.erfinv(jnp.linspace(-1.0, 1.0, n + 2)[1:-1])
        c = jnp.mean(phi(x) ** 2) ** 0.5
        c = c.item()

        if jnp.allclose(c, 1.0):
            rho = phi
        else:

            def rho(x):
                return phi(x) / c

        if return_scale:
            return rho, 1.0 / c
        return rho


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
                "FourierActivation": FourierActivation,
                "GaussianBasis": GausianBasisActivation,
                "normalize": normalize_activation,
                "DyT": DyT,
                "DynAct": DynAct,
            },
        )
    except Exception as e:
        raise ValueError(
            f"The following exception was raised while parsing the activation function {activation} : {e}"
        )
