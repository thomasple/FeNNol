import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Union
import math
import numpy as np


class SpeciesEncoding(nn.Module):
    """
    A module that encodes species information.
    For now, the only encoding is a random vector for each species.

    Args:
        dim (int): The dimensionality of the output encoding.
        zmax (int): The maximum atomic number to encode.
        output_key (Optional[str]): The key to use for the output in the returned dictionary.

    Returns:
        jax.Array or dict: The encoded species information.
    """

    dim: int = 16
    zmax: int = 50
    output_key: Optional[str] = None

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        species = inputs["species"] if isinstance(inputs, dict) else inputs

        ############################
        conv_tensor = self.param(
            "conv_tensor",
            lambda key, shape: jax.nn.standardize(jax.random.normal(key, shape)),
            (self.zmax, self.dim),
        )
        out = conv_tensor[species]
        ############################

        if isinstance(inputs, dict):
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class RadialBasis(nn.Module):
    """
    A module that computes a radial embedding of distances.
    For now, the only embedding is the Bessel embedding used for example in Allegro.

    Args:
        end (float): The maximum distance to consider.
        start (float, optional): The minimum distance to consider (default: 0.0).
        dim (int, optional): The number of dimensions of the embedding (default: 8).
        graph_key (str, optional): The key to use to extract the distances from a graph input (default: None).
            If None, the input is either a graph (dict) or a JAX array containing distances.
        output_key (str, optional): The key to use to store the embedding in the output dictionary (default: None).
            If None, the embedding directly returned.

    Returns:
        jax.Array or dict: The radial embedding of the distances.
    """

    end: float
    start: float = 0.0
    dim: int = 8
    graph_key: Optional[str] = None
    output_key: Optional[str] = None
    basis: str = "bessel"
    trainable: bool = False

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.graph_key is not None:
            x = inputs[self.graph_key]["distances"]
        else:
            x = inputs["distances"] if isinstance(inputs, dict) else inputs

        basis = self.basis.lower()
        ############################
        if basis == "bessel":
            c = self.end - self.start
            x = x[:, None] - self.start

            if self.trainable:
                bessel_roots = self.param(
                    "bessel_roots",
                    lambda key, dim: jnp.asarray(
                        np.arange(1, dim + 1, dtype=x.dtype)[None, :] * (math.pi / c)
                    ),
                    self.dim,
                )
            else:
                bessel_roots = jnp.asarray(
                    np.arange(1, self.dim + 1, dtype=x.dtype)[None, :] * (math.pi / c)
                )

            out = (2.0 / c) ** 0.5 * jnp.sin(x * bessel_roots) / x

        elif basis == "gaussian":
            if self.trainable:
                roots = self.param(
                    "radial_centers",
                    lambda key, dim, start, end: jnp.linspace(
                        start, end, dim + 1, dtype=x.dtype
                    )[None, :-1],
                    self.dim,
                    self.start,
                    self.end,
                )
                eta = self.param(
                    "radial_etas",
                    lambda key, dim, start, end: jnp.full(
                        dim,
                        dim / (end - start),
                        dtype=x.dtype,
                    )[None, :],
                    self.dim,
                    self.start,
                    self.end,
                )

            else:
                roots = jnp.asarray(
                    np.linspace(self.start, self.end, self.dim + 1)[None, :-1],
                    dtype=x.dtype,
                )
                eta = jnp.asarray(
                    np.full(self.dim, self.dim / (self.end - self.start))[None, :],
                    dtype=x.dtype,
                )

            x2 = (eta * (x[:, None] - roots)) ** 2
            out = jnp.exp(-x2)

        elif basis == "gaussian_rinv":
            rinv_high = 1.0 / max(self.start, 0.1)
            rinv_low = 1.0 / (0.8 * self.end)

            if self.trainable:
                roots = self.param(
                    "radial_centers",
                    lambda key, dim, rinv_low, rinv_high: jnp.linspace(
                        rinv_low, rinv_high, dim, dtype=x.dtype
                    )[None, :],
                    self.dim,
                    rinv_low,
                    rinv_high,
                )
                sigmas = self.param(
                    "radial_sigmas",
                    lambda key, dim, rinv_low, rinv_high: jnp.full(
                        dim,
                        2**0.5 / (2 * dim * rinv_low),
                        dtype=x.dtype,
                    )[None, :],
                    self.dim,
                    rinv_low,
                    rinv_high,
                )
            else:
                roots = jnp.asarray(
                    np.linspace(rinv_low, rinv_high, self.dim, dtype=x.dtype)[None, :]
                )
                sigmas = jnp.asarray(
                    np.full(
                        self.dim,
                        2**0.5 / (2 * self.dim * rinv_low),
                    )[None, :],
                    dtype=x.dtype,
                )

            rinv = 1.0 / x

            out = jnp.exp(-((rinv[..., None] - roots) ** 2) / sigmas**2)

        elif basis == "fourier":
            if self.trainable:
                roots = self.param(
                    "roots",
                    lambda key, dim: jnp.arange(dim, dtype=x.dtype)[None, :] * math.pi,
                    self.dim,
                )
            else:
                roots = jnp.asarray(
                    np.arange(self.dim)[None, :] * math.pi, dtype=x.dtype
                )
            x = (x[:, None] - self.start) / (self.end - self.start)
            norm = 1.0 / (0.25 + 0.5 * self.dim) ** 0.5
            out = norm * jnp.cos(x * roots)

        elif basis == "spooky":
            norms = []
            for k in range(self.dim):
                norms.append(math.comb(self.dim - 1, k))
            norms = jnp.asarray(np.array(norms)[None, :], dtype=x.dtype)
            gamma = 0.5 / self.end
            if self.trainable:
                gamma = jnp.abs(
                    self.param("gamma", lambda key: jnp.asarray(gamma, dtype=x.dtype))
                )
            e = jnp.exp(-gamma * x)[:, None]
            k = jnp.asarray(np.arange(self.dim, dtype=x.dtype)[None, :])
            b = e**k * (1 - e) ** (self.dim - 1 - k)
            out = b * norms
        else:
            raise NotImplementedError(f"Unknown radial basis {basis}.")
        ############################

        if self.graph_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


ENCODINGS = {
    "RADIAL_ENCODING": RadialBasis,
    "SPECIES_ENCODING": SpeciesEncoding,
}
