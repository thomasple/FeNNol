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

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.graph_key is not None:
            x = inputs[self.graph_key]["distances"]
        else:
            x = inputs["distances"] if isinstance(inputs, dict) else inputs
        

        ############################
        c = self.end - self.start
        x = x[:, None] - self.start
        bessel_roots = jnp.asarray(
            np.arange(1, self.dim + 1, dtype=x.dtype)[None, :] * (math.pi / c)
        )

        out = (2.0 / c) ** 0.5 * jnp.sin(x * bessel_roots) / x
        ############################

        if self.graph_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out} if output_key is not None else out
        return out


ENCODINGS={
    "RADIAL_ENCODING": RadialBasis,
    "SPECIES_ENCODING": SpeciesEncoding,
}
