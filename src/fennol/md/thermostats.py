import numpy as np
from ..utils.atomic_units import AtomicUnits as au  # CM1,THZ,BOHR,MPROT
import flax.linen as nn
import jax
import jax.numpy as jnp
import math


def get_thermostat(thermostat_name,dt,mass, gamma=None, kT=None):

    if thermostat_name in ["LGV", "LANGEVIN"]:
        if kT is None or gamma is None:
            raise ValueError("kT and gamma must be specified for Langevin thermostat")
        a1 = math.exp(-gamma * dt)
        a2 = jnp.asarray(((1 - a1 * a1) * kT / mass) ** 0.5)

        def thermostat(vel, rng_key):
            rng_key,noise_key = jax.random.split(rng_key)
            noise = jax.random.normal(noise_key, vel.shape)
            vel = a1 * vel + a2 * noise
            return vel, rng_key
    elif thermostat_name in [
            "GD",
            "DESCENT",
            "GRADIENT_DESCENT",
            "MIN",
            "MINIMIZE",
        ]:
        a1 = math.exp(-gamma * dt)
        
        def thermostat(vel, rng_key):
            return a1 * vel, rng_key
    elif thermostat_name in ["NVE", "NONE"]:
        thermostat = lambda x,r:(x,r)
    else:
        raise ValueError(f"Unknown thermostat {thermostat_name}")

    return thermostat
