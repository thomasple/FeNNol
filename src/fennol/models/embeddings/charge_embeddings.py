import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict, Union, ClassVar, Optional
import numpy as np
from ...utils.periodic_table import (
    VALENCE_ELECTRONS,
)

class ChargeHypothesis(nn.Module):
    """Embedding with total charge constraint vi Multiple Neural Charge Equilibration.

    FID: CHARGE_HYPOTHESIS

    """

    embedding_key: str
    """Key for the embedding in the inputs that is used to predict an 'electron affinity' weight"""
    output_key: Union[str, None] = None
    """Key for the charges in the outputs"""
    total_charge_key: str = "total_charge"
    """Key for the total charge in the inputs"""
    ncharges: int = 10
    """The number of charge hypothesis"""
    mode: str = "qeq"
    """Charge distribution mode. Only 'qeq' available for now."""
    squeeze: bool = True

    FID: ClassVar[str] = "CHARGE_HYPOTHESIS"

    @nn.compact
    def __call__(self, inputs):
        embedding = inputs[self.embedding_key].astype(inputs["coordinates"].dtype)

        wi = jax.nn.softplus(
            nn.Dense(self.ncharges, use_bias=True, name="wi")(embedding)
        )

        batch_index = inputs["batch_index"]
        nsys = inputs["natoms"].shape[0]
        wtot = jax.ops.segment_sum(wi, batch_index, nsys)

        Qtot = (
            inputs[self.total_charge_key].astype(wi.dtype)
            if self.total_charge_key in inputs
            else jnp.zeros(nsys, dtype=wi.dtype)
        )
        if Qtot.ndim == 0:
            Qtot = Qtot * jnp.ones(nsys, dtype=wi.dtype)

        
        qtilde = nn.Dense(self.ncharges, use_bias=True, name="qi")(embedding)
        qtot = jax.ops.segment_sum(qtilde, batch_index, nsys)
        dq = Qtot[:, None] - qtot
        f = (dq / wtot)[batch_index]
        q = qtilde + wi * f

        if self.squeeze and self.ncharges == 1:
            q = jnp.squeeze(q, axis=-1)

        output_key = self.output_key if self.output_key is not None else self.name
        return {
            **inputs,
            output_key: q,
        }