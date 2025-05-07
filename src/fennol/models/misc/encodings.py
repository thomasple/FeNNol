import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Union, List, Sequence, ClassVar, Dict
import math
import dataclasses
import numpy as np
from .nets import FullyConnectedNet
from ...utils import AtomicUnits as au
from functools import partial
from ...utils.periodic_table import (
    PERIODIC_TABLE_REV_IDX,
    PERIODIC_TABLE,
    EL_STRUCT,
    VALENCE_STRUCTURE,
    XENONPY_PROPS,
    SJS_COORDINATES,
    PERIODIC_COORDINATES,
    ATOMIC_IONIZATION_ENERGY,
    POLARIZABILITIES,
    D3_COV_RADII,
    VDW_RADII,
    OXIDATION_STATES,
)


class SpeciesEncoding(nn.Module):
    """A module that encodes chemical species information.

    FID: SPECIES_ENCODING
    """

    encoding: str = "random"
    """ The encoding to use. Can be one of "one_hot", "occupation", "electronic_structure", "properties", "sjs_coordinates", "random". 
        Multiple encodings can be concatenated using the "+" separator.
    """
    dim: int = 16
    """ The dimension of the encoding if not fixed by design."""
    zmax: int = 86
    """ The maximum atomic number to encode."""
    output_key: Optional[str] = None
    """ The key to use for the output in the returned dictionary."""

    species_order: Optional[Union[str, Sequence[str]]] = None
    """ The order of the species to use for the encoding. Only used for "onehot" encoding.
         If None, we encode all elements up to `zmax`."""
    trainable: bool = False
    """ Whether the encoding is trainable or fixed. Does not apply to "random" encoding which is always trainable."""
    extra_params: Dict = dataclasses.field(default_factory=dict)
    """ Dictionary of extra parameters for the basis."""

    FID: ClassVar[str] = "SPECIES_ENCODING"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:

        zmax = self.zmax
        if zmax <= 0 or zmax > len(PERIODIC_TABLE):
            zmax = len(PERIODIC_TABLE)

        zmaxpad = zmax + 2

        encoding = self.encoding.lower().strip()
        encodings = encoding.split("+")
        ############################
        conv_tensors = []

        if "one_hot" in encodings or "onehot" in encodings:
            if self.species_order is None:
                conv_tensor = np.eye(zmax)
                conv_tensor = np.concatenate(
                    [np.zeros((1, zmax)), conv_tensor, np.zeros((1, zmax))], axis=0
                )
            else:
                if isinstance(self.species_order, str):
                    species_order = [el.strip() for el in self.species_order.split(",")]
                else:
                    species_order = [el for el in self.species_order]
                conv_tensor = np.zeros((zmaxpad, len(species_order)))
                for i, s in enumerate(species_order):
                    conv_tensor[PERIODIC_TABLE_REV_IDX[s], i] = 1

            conv_tensors.append(conv_tensor)

        if "occupation" in encodings:
            conv_tensor = np.zeros((zmaxpad, (zmax + 1) // 2))
            for i in range(1, zmax + 1):
                conv_tensor[i, : i // 2] = 1
                if i % 2 == 1:
                    conv_tensor[i, i // 2] = 0.5

            conv_tensors.append(conv_tensor)

        if "electronic_structure" in encodings:
            Z = np.arange(1, zmax + 1).reshape(-1, 1)
            Zref = [zmax]
            e_struct = np.array(EL_STRUCT[1 : zmax + 1])
            eref = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 14, 10, 6]
            v_struct = np.array(VALENCE_STRUCTURE[1 : zmax + 1])
            vref = [2, 6, 10, 14]
            if zmax <= 86:
                e_struct = e_struct[:, :15]
                eref = eref[:15]
            ref = np.array(Zref + eref + vref)
            conv_tensor = np.concatenate([Z, e_struct, v_struct], axis=1)
            conv_tensor = conv_tensor / ref[None, :]
            dim = conv_tensor.shape[1]
            conv_tensor = np.concatenate(
                [np.zeros((1, dim)), conv_tensor, np.zeros((1, dim))], axis=0
            )

            conv_tensors.append(conv_tensor)
        
        if "valence_structure" in encodings:
            vref = np.array([2, 6, 10, 14])
            conv_tensor = np.array(VALENCE_STRUCTURE[1 : zmax + 1])/vref[None,:]
            dim = conv_tensor.shape[1]
            conv_tensor = np.concatenate(
                [np.zeros((1, dim)), conv_tensor, np.zeros((1, dim))], axis=0
            )
            conv_tensors.append(conv_tensor)
        
        if "period_onehot" in encodings:
            periods = PERIODIC_COORDINATES[1 : zmax + 1, 0]-1
            conv_tensor = np.eye(7)[periods]
            dim = conv_tensor.shape[1]
            conv_tensor = np.concatenate(
                [np.zeros((1, dim)), conv_tensor, np.zeros((1, dim))], axis=0
            )
            conv_tensors.append(conv_tensor)
        
        if "period" in encodings:
            conv_tensor = (PERIODIC_COORDINATES[1 : zmax + 1, 0]/7.)[:,None]
            dim = conv_tensor.shape[1]
            conv_tensor = np.concatenate(
                [np.zeros((1, dim)), conv_tensor, np.zeros((1, dim))], axis=0
            )
            conv_tensors.append(conv_tensor)
        
        if "period_positional" in encodings:
            row = PERIODIC_COORDINATES[1 : zmax + 1, 0]
            drow = self.extra_params.get("drow", default=5)
            nrow = self.extra_params.get("nrow", default=100.0)

            conv_tensor = positional_encoding_static(row, drow, nrow)
            dim = conv_tensor.shape[1]
            conv_tensor = np.concatenate(
                [np.zeros((1, dim)), conv_tensor, np.zeros((1, dim))], axis=0
            )
            conv_tensors.append(conv_tensor)


        if "properties" in encodings:
            props = np.array(XENONPY_PROPS)[1:-1]
            assert (
                self.zmax <= props.shape[0]
            ), f"zmax > {props.shape[0]} not supported for xenonpy properties"
            conv_tensor = props[1 : zmax + 1]
            mean = np.mean(props, axis=0)
            std = np.std(props, axis=0)
            conv_tensor = (conv_tensor - mean[None, :]) / std[None, :]
            dim = conv_tensor.shape[1]
            conv_tensor = np.concatenate(
                [np.zeros((1, dim)), conv_tensor, np.zeros((1, dim))], axis=0
            )
            conv_tensors.append(conv_tensor)
        
        
        if "valence_properties" in encodings:
            assert zmax <= 86, "Valence properties only available for zmax <= 86"
            Z = np.arange(1, zmax + 1).reshape(-1, 1)
            Zref = [zmax]
            Zinv = 1.0 / Z
            Zinvref = [1.0]
            v_struct = np.array(VALENCE_STRUCTURE[1 : zmax + 1])
            vref = [2, 6, 10, 14]
            ionization = np.array(ATOMIC_IONIZATION_ENERGY[1 : zmax + 1]).reshape(-1, 1)
            ionizationref = [0.5]
            polariz = np.array(POLARIZABILITIES[1 : zmax + 1]).reshape(-1, 1)
            polarizref = [np.median(polariz)]

            cov = np.array(D3_COV_RADII[1 : zmax + 1]).reshape(-1, 1)
            covref = [np.median(cov)]

            vdw = np.array(VDW_RADII[1 : zmax + 1]).reshape(-1, 1)
            vdwref = [np.median(vdw)]

            ref = np.array(
                Zref + Zinvref + ionizationref + polarizref + covref + vdwref + vref
            )
            conv_tensor = np.concatenate(
                [Z, Zinv, ionization, polariz, cov, vdw, v_struct], axis=1
            )

            conv_tensor = conv_tensor / ref[None, :]
            dim = conv_tensor.shape[1]
            conv_tensor = np.concatenate(
                [np.zeros((1, dim)), conv_tensor, np.zeros((1, dim))], axis=0
            )
            conv_tensors.append(conv_tensor)

        if "sjs_coordinates" in encodings:
            coords = np.array(SJS_COORDINATES)[1:-1]
            conv_tensor = coords[1 : zmax + 1]
            mean = np.mean(coords, axis=0)
            std = np.std(coords, axis=0)
            conv_tensor = (conv_tensor - mean[None, :]) / std[None, :]
            dim = conv_tensor.shape[1]
            conv_tensor = np.concatenate(
                [np.zeros((1, dim)), conv_tensor, np.zeros((1, dim))], axis=0
            )
            conv_tensors.append(conv_tensor)

        if "positional" in encodings:
            coords = np.array(PERIODIC_COORDINATES)[1 : zmax + 1]
            row, col = coords[:, 0], coords[:, 1]
            drow = self.extra_params.get("drow", default=5)
            dcol = self.dim - drow
            nrow = self.extra_params.get("nrow", default=100.0)
            ncol = self.extra_params.get("ncol", default=1000.0)

            erow = positional_encoding_static(row, drow, nrow)
            ecol = positional_encoding_static(col, dcol, ncol)
            conv_tensor = np.concatenate([erow, ecol], axis=-1)
            dim = conv_tensor.shape[1]
            conv_tensor = np.concatenate(
                [np.zeros((1, dim)), conv_tensor, np.zeros((1, dim))], axis=0
            )
            conv_tensors.append(conv_tensor)

        if "oxidation" in encodings:
            states_set = sorted(set(sum(OXIDATION_STATES, [])))
            nstates = len(states_set)
            state_dict = {s: i for i, s in enumerate(states_set)}
            conv_tensor = np.zeros((zmaxpad, nstates))
            for i, states in enumerate(OXIDATION_STATES[1 : zmax + 1]):
                for s in states:
                    conv_tensor[i, state_dict[s]] = 1
            conv_tensors.append(conv_tensor)

        if len(conv_tensors) > 0:
            conv_tensor = np.concatenate(conv_tensors, axis=1)
            if self.trainable:
                conv_tensor = self.param(
                    "conv_tensor",
                    lambda key: jnp.asarray(conv_tensor, dtype=jnp.float32),
                )
            else:
                conv_tensor = jnp.asarray(conv_tensor, dtype=jnp.float32)
            conv_tensors = [conv_tensor]
        else:
            conv_tensors = []

        if "random" in encodings:
            rand_encoding = self.param(
                "rand_encoding",
                lambda key, shape: jax.nn.standardize(
                    jax.random.normal(key, shape, dtype=jnp.float32)
                ),
                (zmaxpad, self.dim),
            )
            conv_tensors.append(rand_encoding)
        
        if "zeros" in encodings:
            zeros_encoding = self.param(
                "zeros_encoding",
                lambda key, shape: jnp.zeros(shape, dtype=jnp.float32),
                (zmaxpad, self.dim),
            )
            conv_tensors.append(zeros_encoding)

        if "randint" in encodings:
            rand_encoding = self.param(
                "randint_encoding",
                lambda key, shape: jax.random.randint(key, shape, 0, 2).astype(
                    jnp.float32
                ),
                (zmaxpad, self.dim),
            )
            conv_tensors.append(rand_encoding)

        if "randtri" in encodings:
            rand_encoding = self.param(
                "randtri_encoding",
                lambda key, shape: (jax.random.randint(key, shape, 0, 3) - 1).astype(
                    jnp.float32
                ),
                (zmaxpad, self.dim),
            )
            conv_tensors.append(rand_encoding)

        assert len(conv_tensors) > 0, f"No encoding recognized in '{self.encoding}'"

        conv_tensor = jnp.concatenate(conv_tensors, axis=1)

        species = inputs["species"] if isinstance(inputs, dict) else inputs
        out = conv_tensor[species]
        ############################

        if isinstance(inputs, dict):
            output_key = self.name if self.output_key is None else self.output_key
            out = out.astype(inputs["coordinates"].dtype)
            return {**inputs, output_key: out} if output_key is not None else out
        return out


class RadialBasis(nn.Module):
    """Computes a radial encoding of distances.

    FID: RADIAL_BASIS
    """

    end: float
    """ The maximum distance to consider."""
    start: float = 0.0
    """ The minimum distance to consider."""
    dim: int = 8
    """ The dimension of the basis."""
    graph_key: Optional[str] = None
    """ The key of the graph in the inputs."""
    output_key: Optional[str] = None
    """ The key to use for the output in the returned dictionary."""
    basis: str = "bessel"
    """ The basis to use. Can be one of "bessel", "gaussian", "gaussian_rinv", "fourier", "spooky"."""
    trainable: bool = False
    """ Whether the basis parameters are trainable or fixed."""
    enforce_positive: bool = False
    """ Whether to enforce distance-start to be positive"""
    gamma: float = 1.0 / (2 * au.BOHR)
    """ The gamma parameter for the "spooky" basis."""
    n_levels: int = 10
    """ The number of levels for the "levels" basis."""
    alt_bessel_norm: bool = False
    """ If True, use the (2/(end-start))**0.5 normalization for the bessel basis."""
    extra_params: Dict = dataclasses.field(default_factory=dict)
    """ Dictionary of extra parameters for the basis."""

    FID: ClassVar[str] = "RADIAL_BASIS"

    @nn.compact
    def __call__(self, inputs: Union[dict, jax.Array]) -> Union[dict, jax.Array]:
        if self.graph_key is not None:
            x = inputs[self.graph_key]["distances"]
        else:
            x = inputs["distances"] if isinstance(inputs, dict) else inputs

        shape = x.shape
        x = x.reshape(-1)

        basis = self.basis.lower()
        ############################
        if basis == "bessel":
            c = self.end - self.start
            x = x[:, None] - self.start
            # if self.enforce_positive:
            #     x = jax.nn.softplus(x)

            if self.trainable:
                bessel_roots = self.param(
                    "bessel_roots",
                    lambda key, dim: jnp.asarray(
                        np.arange(1, dim + 1, dtype=x.dtype)[None, :] * (math.pi / c)
                    ),
                    self.dim,
                )
                norm = 1.0 / jnp.max(
                    bessel_roots
                )  # (2.0 / c) ** 0.5 /jnp.max(bessel_roots)
            else:
                bessel_roots = jnp.asarray(
                    np.arange(1, self.dim + 1, dtype=x.dtype)[None, :] * (math.pi / c)
                )
                norm = 1.0 / (
                    self.dim * math.pi / c
                )  # (2.0 / c) ** 0.5/(self.dim*math.pi/c)
            if self.alt_bessel_norm:
                norm = (2.0 / c) ** 0.5
            out = norm * jnp.sin(x * bessel_roots) / x

            if self.enforce_positive:
                out = jnp.where(x > 0, out * (1.0 - jnp.exp(-(x**2))), 0.0)

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

            x = x[:, None]
            x2 = (eta * (x - roots)) ** 2
            out = jnp.exp(-x2)
            if self.enforce_positive:
                out = jnp.where(
                    x > self.start,
                    out * (1.0 - jnp.exp(-10 * (x - self.start) ** 2)),
                    0.0,
                )

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

            rinv = 1.0 / x[:, None]

            out = jnp.exp(-((rinv - roots) ** 2) / sigmas**2)

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
            c = self.end - self.start
            x = x[:, None] - self.start
            # if self.enforce_positive:
            #     x = jax.nn.softplus(x)
            norm = 1.0 / (0.25 + 0.5 * self.dim) ** 0.5
            out = norm * jnp.cos(x * roots / c)
            if self.enforce_positive:
                out = jnp.where(x > 0, out, norm)

        elif basis == "spooky":

            gamma = self.gamma
            if self.trainable:
                gamma = jnp.abs(
                    self.param("gamma", lambda key: jnp.asarray(gamma, dtype=x.dtype))
                )

            if self.enforce_positive:
                x = jnp.where(x - self.start > 1.0e-3, x - self.start, 1.0e-3)[:, None]
                dim = self.dim
            else:
                x = x[:, None] - self.start
                dim = self.dim - 1

            norms = []
            for k in range(self.dim):
                norms.append(math.comb(dim, k))
            norms = jnp.asarray(np.array(norms)[None, :], dtype=x.dtype)

            e = jnp.exp(-gamma * x)
            k = jnp.asarray(np.arange(self.dim, dtype=x.dtype)[None, :])
            b = e**k * (1 - e) ** (dim - k)
            out = b * e * norms
            if self.enforce_positive:
                out = jnp.where(x > 1.0e-3, out * (1.0 - jnp.exp(-(x**2))), 0.0)
            # logfac = np.zeros(self.dim)
            # for i in range(2, self.dim):
            #     logfac[i] = logfac[i - 1] + np.log(i)
            # k = np.arange(self.dim)
            # n = self.dim - 1 - k
            # logbin = jnp.asarray((logfac[-1] - logfac[k] - logfac[n])[None,:], dtype=x.dtype)
            # n = jnp.asarray(n[None,:], dtype=x.dtype)
            # k = jnp.asarray(k[None,:], dtype=x.dtype)

            # gamma = 1.0 / (2 * au.BOHR)
            # if self.trainable:
            #     gamma = jnp.abs(
            #         self.param("gamma", lambda key: jnp.asarray(gamma, dtype=x.dtype))
            #     )
            # gammar = (-gamma * x)[:,None]
            # x = logbin + n * gammar + k * jnp.log(-jnp.expm1(gammar))
            # out = jnp.exp(x)*jnp.exp(gammar)
        elif basis == "levels":
            assert self.n_levels >= 2, "Number of levels must be >= 2."

            def initialize_levels(key):
                key0, key1, key_phi = jax.random.split(key, 3)
                level0 = jax.random.randint(key0, (self.dim,), 0, 2)
                level1 = jax.random.randint(key1, (self.dim,), 0, 2)
                # level0 = jax.random.normal(key0, (self.dim,), dtype=jnp.float32)
                # level1 = jax.random.normal(key1, (self.dim,), dtype=jnp.float32)
                phi = jax.random.uniform(key_phi, (self.dim,), dtype=jnp.float32)
                levels = [level0]
                for l in range(2, self.n_levels - 1):
                    tau = float(self.n_levels - l) / float(self.n_levels - 1)
                    phil = phi < tau
                    level = jnp.where(phil, level0, level1)
                    levels.append(level)
                levels.append(level1)
                return jnp.stack(levels).astype(jnp.float32)

            levels = self.param("levels", initialize_levels)
            # levels = self.param(
            #     "levels",
            #     lambda key, shape: jax.random.normal(key, shape, dtype=jnp.float32),
            #     (self.n_levels,self.dim),
            # )

            flevel = (x - self.start) / (self.end - self.start) * (self.n_levels - 1)
            ilevel = jnp.floor(flevel).astype(jnp.int32)
            ilevel1 = jnp.clip(ilevel + 1, 0, self.n_levels - 1)
            ilevel = jnp.clip(ilevel, 0, self.n_levels - 1)

            dx = flevel - ilevel
            w = 0.5 * (1 + jnp.cos(jnp.pi * dx))[:, None]

            ## interpolate between level vectors
            v1 = levels[ilevel]
            v2 = levels[ilevel1]
            out = v1 * w + v2 * (1 - w)
        elif basis == "finite_support":
            flevel = (x - self.start) / (self.end - self.start) * (self.dim + 1)
            ilevel = jnp.floor(flevel).astype(jnp.int32)
            ilevel1 = jnp.clip(ilevel + 1, 0, self.dim + 1)
            ilevel = jnp.clip(ilevel, 0, self.dim + 1)

            dx = flevel - ilevel
            w = 0.5 * (1 + jnp.cos(jnp.pi * dx))

            ilevelflat = ilevel + jnp.arange(x.shape[0]) * (self.dim + 2)
            ilevel1flat = ilevel1 + jnp.arange(x.shape[0]) * (self.dim + 2)

            out = (
                jnp.zeros((x.shape[0] * (self.dim + 2)), dtype=x.dtype)
                .at[ilevelflat]
                .set(w)
                .at[ilevel1flat]
                .set(1 - w)
                .reshape(-1, self.dim + 2)[:, 1:-1]
            )

        elif basis == "exp_lr" or basis == "exp":
            zeta = self.extra_params.get("zeta", default=2.0)
            s = self.extra_params.get("s", default=0.5)
            n = np.arange(self.dim)
            # if self.trainable:
            # zeta = jnp.abs(
            #     self.param("zeta", lambda key: jnp.asarray(zeta, dtype=x.dtype))
            # )
            # s = jnp.abs(self.param("s", lambda key: jnp.asarray(s, dtype=x.dtype)))

            a = zeta * s**n
            xx = np.linspace(0, self.end, 10000)

            if self.start > 0:
                a1 = np.minimum(1.0 / a, 1.0)
                switchstart = jnp.where(
                    x[:, None] < self.end,
                    1
                    - (
                        0.5
                        + 0.5
                        * jnp.cos(
                            np.pi * (x[:, None] - self.start) / (self.end - self.start)
                        )
                    )
                    ** a1[None, :],
                    1,
                ) * (x[:, None] > self.start)
                switchstartxx = (
                    1
                    - (
                        0.5
                        + 0.5
                        * np.cos(
                            np.pi * (xx[:, None] - self.start) / (self.end - self.start)
                        )
                    )
                    ** a1[None, :]
                ) * (xx[:, None] > self.start)

            else:
                switchstart = 1.0
                switchstartxx = 1.0

            norm = 1.0 / np.trapz(
                switchstartxx * np.exp(-a[None, :] * xx[:, None]), xx, axis=0
            )
            # norm = 1./np.max(switchstartxx*np.exp(-a[None, :] * xx[:, None]), axis=0)

            if self.trainable:
                a = jnp.abs(self.param("exponent", lambda key: jnp.asarray(a)))

            out = switchstart * jnp.exp(-a[None, :] * x[:, None]) * norm[None, :]

        elif basis == "neural_net" or basis == "nn":
            neurons = [*self.extra_params.get(
                "hidden_neurons", default=[2 * self.dim]
            ), self.dim]
            activation = self.extra_params.get("activation", default="swish")
            use_bias = self.extra_params.get("use_bias", default=True)

            out = FullyConnectedNet(
                neurons, activation=activation, use_bias=use_bias, squeeze=False
            )(x[:, None])
        elif basis == "damped_coulomb":
            l = np.arange(self.dim)[None, :]
            a = self.extra_params.get("a", default=1.0)
            if self.trainable:
                a = self.param("a", lambda key: jnp.asarray([a]*self.dim)[None,:])
            a2=a**2
            x = x[:, None] - self.start
            end = self.end - self.start
            x21 = a2 + x**2
            R21 = a2 + end**2
            out = (
                1.0 / x21 ** (0.5 * (l + 1))
                - 1.0 / (R21 ** (0.5 * (l + 1)))
                + (x - end) * ((l + 1) * end / (R21 ** (0.5 * (l + 3))))
            ) * (x < end)

        elif basis.startswith("spherical_bessel_"):
            l = int(basis.split("_")[-1])
            out = generate_spherical_jn_basis(self.dim, self.end,[l])(x)
        else:
            raise NotImplementedError(f"Unknown radial basis {basis}.")
        ############################

        out = out.reshape((*shape, self.dim))

        if self.graph_key is not None:
            output_key = self.name if self.output_key is None else self.output_key
            return {**inputs, output_key: out}
        return out


def positional_encoding_static(t, d: int, n: float = 10000.0):
    if d % 2 == 0:
        k = np.arange(d // 2)
    else:
        k = np.arange((d + 1) // 2)
    wk = np.asarray(1.0 / (n ** (2 * k / d)))
    wkt = wk[None, :] * t[:, None]
    out = np.concatenate([np.cos(wkt), np.sin(wkt)], axis=-1)
    if d % 2 == 1:
        out = out[:, :-1]
    return out


@partial(jax.jit, static_argnums=(1, 2), inline=True)
def positional_encoding(t, d: int, n: float = 10000.0):
    if d % 2 == 0:
        k = np.arange(d // 2)
    else:
        k = np.arange((d + 1) // 2)
    wk = jnp.asarray(1.0 / (n ** (2 * k / d)))
    wkt = wk[None, :] * t[:, None]
    out = jnp.concatenate([jnp.cos(wkt), jnp.sin(wkt)], axis=-1)
    if d % 2 == 1:
        out = out[:, :-1]
    return out


def generate_spherical_jn_basis(dim:int, rc:float, ls:Union[int,Sequence[int]]=[0], print_code:bool=False, jit:bool=False):
    from sympy import Symbol, jn, expand_func
    from scipy.special import spherical_jn
    from sympy import jn_zeros
    import scipy.integrate as integrate

    if isinstance(ls, int):
        ls = list(range(ls + 1))
    zl = [Symbol(f"xz[...,{l}]") for l in ls]
    zn = np.array([jn_zeros(l, dim) for l in ls], dtype=float).T
    znrc = zn / rc
    norms = np.zeros((dim, len(ls)), dtype=float)
    for il,l in enumerate(ls):
        for i in range(dim):
            norms[i, il] = (
                integrate.quad(lambda x: (spherical_jn(l, x) * x) ** 2, 0, zn[i, il])[0]
                / znrc[i, il] ** 3
            ) ** (-0.5)

    fn_str = f"""def spherical_jn_basis_(x):
    from jax.numpy import cos,sin
    
    znrc = jnp.array({znrc.tolist()},dtype=x.dtype)
    norms = jnp.array({norms.tolist()},dtype=x.dtype)
    xshape = x.shape
    x = x.reshape(-1)
    xz = x[:,None,None]*znrc[None,:,:]

    jns = jnp.stack([
  """
    for il,l in enumerate(ls):
        fn_str += f"      {expand_func(jn(l, zl[il]))},\n"
    fn_str += f"""    ],axis=-1)
    return (norms[None,:,:]*jns).reshape(*xshape,{dim},{len(ls)})
  """

    if print_code:
        print(fn_str)
    exec(fn_str)
    jn_basis = locals()["spherical_jn_basis_"]
    if jit:
        jn_basis = jax.jit(jn_basis)
    return jn_basis
