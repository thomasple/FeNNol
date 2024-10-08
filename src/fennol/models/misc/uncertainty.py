import flax.linen as nn
from typing import Any, Sequence, Callable, Union, ClassVar,Optional
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
import dataclasses
from ...utils.periodic_table import PERIODIC_TABLE


class EnsembleStatistics(nn.Module):
    """Computes the mean and variance of an ensemble.
    
    FID: ENSEMBLE_STAT
    """

    key: str
    """The key to access the input data from the `inputs` dictionary."""
    axis: int = -1
    """The axis along which to compute the mean and variance."""
    shuffle_ensemble: bool = False
    """Whether to shuffle the ensemble."""
    weights_key: Optional[str] = None
    mean_key: Optional[str] = None

    FID: ClassVar[str] = "ENSEMBLE_STAT"

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        if self.weights_key is not None:
            weights = inputs[self.weights_key]
        else:
            weights = jnp.ones_like(x)
        weights = weights/jnp.sum(weights,axis=self.axis,keepdims=True)
        mean = jnp.sum(x*weights, axis=self.axis,keepdims=True)

        nsamples = x.shape[self.axis]
        if nsamples == 1:
            var = jnp.zeros_like(mean)
        else:
            # var = jnp.var(x, axis=self.axis, ddof=1)
            var = jnp.sum(weights*(x-mean)**2,axis=self.axis)*(nsamples/(nsamples-1.))
        
        mean = jnp.squeeze(mean,axis=self.axis)
        output = {**inputs, self.key + "_mean": mean, self.key + "_var": var}

        if self.mean_key is not None:
            output[self.mean_key] = mean

        training = "training" in inputs.get("flags", {})
        if self.shuffle_ensemble and training and "rng_key" in inputs:
            key, subkey = jax.random.split(inputs["rng_key"])
            x = jax.random.permutation(subkey, x, axis=self.axis, independent=True)
            output[self.key] = x
            output["rng_key"] = key
        return output


class EnsembleShift(nn.Module):
    """Shifts the mean of an ensemble to match a reference tensor.
    
    FID: ENSEMBLE_SHIFT
    """

    key: str
    """The key to access the input data from the `inputs` dictionary."""
    ref_key: str
    """The key to access the reference data from the `inputs` dictionary."""
    axis: int = -1
    """The axis of the ensemble."""
    output_key: Optional[str] = None
    """The key of the output ensemble. If None, the input key will be used."""


    FID: ClassVar[str] = "ENSEMBLE_SHIFT"

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        ref = inputs[self.ref_key].reshape(mean.shape)
        x = x - mean + ref
        output_key = self.key if self.output_key is None else self.output_key
        return {**inputs, output_key: x}


class ConstrainEvidence(nn.Module):
    """ Constrain the parameters of an evidential model

    FID: CONSTRAIN_EVIDENCE

    ### References
    - Amini et al, Deep Evidential Regression, NeurIPS 2020 (https://arxiv.org/abs/1910.02600)
    - Meinert et al, Multivariate Deep Evidential Regression, (https://arxiv.org/pdf/2104.06135.pdf)

    """
    key: str
    """The key to access the input data from the `inputs` dictionary."""
    output_key: Optional[str] = None
    """The key to use for the constrained paramters in the `output` dictionary."""
    beta_scale: Union[str, float] = 1.0
    """The scale of the beta parameter."""
    alpha_init: float = 2.0
    """The initial value of the alpha parameter."""
    nu_init: float = 1.0
    """The initial value of the nu parameter."""
    chemical_shift: Optional[float] = None
    """The initial chemical shift of evidence."""
    trainable_beta: bool = False
    """Whether the beta parameter is trainable."""
    constant_beta: bool = False
    """Whether the beta parameter is a constant."""
    trainable_alpha: bool = False
    """Whether the alpha parameter is trainable."""
    constant_alpha: bool = False
    """Whether the alpha parameter is a constant."""
    trainable_nu: bool = False
    """Whether the nu parameter is trainable."""
    constant_nu: bool = False
    """Whether the nu parameter is a constant."""
    nualpha_coupling: Optional[float] = None
    """The coupling constant between nu and alpha."""
    graph_key: Optional[str] = None
    """The key to access the graph data from the `inputs` dictionary. 
        Only used to obtain an environment-dependent chemical shift."""
    self_weight: float = 10.0
    """The weight of the self interaction in environment-dependent chemical shift."""
    # target_dim: Optional[int] = None

    FID: ClassVar[str] = "CONSTRAIN_EVIDENCE"

    @nn.compact
    def __call__(self, inputs) -> Any:
        x = inputs[self.key]
        dim = 3
        assert x.shape[-1] == dim, f"Dimension {-1} must be {dim}, got {x.shape[-1]}"
        nui, alphai, betai = jnp.split(x, 3, axis=-1)

        if self.chemical_shift is not None:
            # modify nu evidence depending on the chemical species 
            #   this is useful for predicting high uncertainties for unknown species
            #   in known geometries
            nu_shift = jnp.abs(
                self.param(
                    "nu_shift",
                    lambda key, shape: jnp.ones(shape) * self.chemical_shift,
                    (len(PERIODIC_TABLE),),
                )
            )[inputs["species"]]
            if self.graph_key is not None:
                graph = inputs[self.graph_key]
                edge_dst = graph["edge_dst"]
                edge_src = graph["edge_src"]
                switch = graph["switch"]
                nushift_neigh = jax.ops.segment_sum(
                    nu_shift[edge_dst] * switch, edge_src, nu_shift.shape[0]
                )
                norm = self.self_weight + jax.ops.segment_sum(
                    switch, edge_src, nu_shift.shape[0]
                )
                nu_shift = (self.self_weight * nu_shift + nushift_neigh) / norm
            nu_shift = nu_shift[:, None]
        else:
            nu_shift = 1.0

        if self.nualpha_coupling is None:
            # couple nu and alpha to remove overparameterization
            #  of the evidential model (see Meinert et al)
            if self.constant_alpha:
                if self.trainable_alpha:
                    alphai = 1 + jnp.abs(
                        self.param(
                            "alpha",
                            lambda key: jnp.asarray(self.alpha_init - 1),
                        )
                    ) * jnp.ones_like(alphai)
                else:
                    assert self.alpha_init > 1, "alpha_init must be >1"
                    alphai = self.alpha_init * jnp.ones_like(alphai)
            elif self.constant_nu:
                alphai = 1 + (self.alpha_init - 1) * nu_shift * jax.nn.softplus(alphai)
            else:
                alphai = 1 + (self.alpha_init - 1) * jax.nn.softplus(alphai)

            if self.constant_nu:
                if self.trainable_nu:
                    nui = 1.0e-5 + jnp.abs(
                        self.param(
                            "nu",
                            lambda key: jnp.asarray(self.nu_init),
                        )
                    ) * jnp.ones_like(nui)
                else:
                    nui = self.nu_init * jnp.ones_like(nui)
            else:
                nui = 1.0e-5 + self.nu_init * nu_shift * jax.nn.softplus(nui)
        else:
            alphai = 1 + nu_shift * jax.nn.softplus(alphai)
            if self.trainable_nu:
                nualpha_coupling = jnp.abs(
                    self.param(
                        "nualpha_coupling",
                        lambda key: jnp.asarray(self.nualpha_coupling),
                    )
                )
            else:
                nualpha_coupling = self.nualpha_coupling
            nui = nualpha_coupling * 2 * alphai

        if self.constant_beta:
            if self.trainable_beta:
                betai = (
                    jax.nn.softplus(
                        self.param(
                            "beta",
                            lambda key: jnp.asarray(0.0),
                        )
                    )
                    / np.log(2.0)
                ) * jnp.ones_like(alphai)
            else:
                betai = jnp.ones_like(alphai)
        else:
            betai = jax.nn.softplus(betai)

        betai = self.beta_scale * betai

        output = jnp.concatenate([nui, alphai, betai], axis=-1)
        # if self.target_dim is not None:
        #     output = output[...,None,:].repeat(self.target_dim, axis=-2)

        output_key = self.key if self.output_key is None else self.output_key
        out = {
            **inputs,
            output_key: output,
            output_key + "_var": jnp.squeeze(betai / (nui * (alphai - 1)), axis=-1),
            output_key + "_aleatoric": jnp.squeeze(betai / (alphai - 1), axis=-1),
            output_key
            + "_wst2": jnp.squeeze(betai * (1 + nui) / (alphai * nui), axis=-1),
        }
        return out
