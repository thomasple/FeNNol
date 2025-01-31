import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import optax

from ..utils.atomic_units import AtomicUnits as au  # CM1,THZ,BOHR,MPROT
from ..utils import Counter
from ..utils.deconvolution import (
    deconvolute_spectrum,
    kernel_lorentz_pot,
    kernel_lorentz,
)


def get_barostat(
    thermostat, simulation_parameters, dt, system_data, fprec, rng_key=None
):
    state = {}

    barostat_name = str(simulation_parameters.get("barostat", "NONE")).upper()

    kT = system_data.get("kT", None)
    assert kT is not None, "kT must be specified for NPT/NPH simulations"
    target_pressure = simulation_parameters.get("target_pressure")
    assert (
        target_pressure is not None
    ), "target_pressure must be specified for NPT/NPH simulations"
    target_pressure = target_pressure / au.BOHR**3

    nbeads = system_data.get("nbeads", None)
    estimate_pressure = True

    pbc_data = system_data["pbc"]

    if barostat_name in ["LGV", "LANGEVIN"]:
        assert rng_key is not None, "rng_key must be provided for QTB barostat"
        rng_key, v_key = jax.random.split(rng_key)
        gamma = simulation_parameters.get("gamma_piston", 20.0 / au.THZ) / au.FS
        tau_piston = simulation_parameters.get("tau_piston", 200.0 / au.FS) * au.FS
        nat = len(system_data["species"])
        masspiston = 3*nat*kT*tau_piston**2
        print(f"# LANGEVIN barostat with piston mass={masspiston:.1e} Ha.fs^2")
        a1 = math.exp(-gamma * dt)
        a2 = ((1 - a1 * a1) * kT / masspiston) ** 0.5

        extvol = jnp.asarray(pbc_data["volume"])
        vextvol = (
            jax.random.normal(v_key, (1,), dtype=extvol.dtype)
            * (kT / masspiston) ** 0.5
        )
        state["extvol"] = extvol
        state["vextvol"] = vextvol
        state["rng_key"] = rng_key

        def barostat(x, vel, system):
            if nbeads is not None:
                x,eigx = x[0],x[1:]
                vel,eigv = vel[0],vel[1:]
            barostat_state = system["barostat"]
            extvol = barostat_state["extvol"]
            vextvol = barostat_state["vextvol"]

            # apply B
            pV = (2 * system["ek"] - jnp.trace(system["virial"])) / 3.0
            aextvol = ((pV - extvol * target_pressure) + kT) * (3./ masspiston)
            vextvol = vextvol + dt * aextvol

            # apply A
            scale1 = jnp.exp((0.5 * dt) * vextvol)
            vel = vel / scale1

            # apply O
            rng_key, noise_key = jax.random.split(barostat_state["rng_key"])
            if nbeads is not None:
                eigv, thermostat_state = thermostat(jnp.concatenate((vel[None], eigv),axis=0), system["thermostat"])
                vel, eigv = eigv[0],eigv[1:]
            else:
                vel, thermostat_state = thermostat(vel, system["thermostat"])
            noise = jax.random.normal(noise_key, (1,), dtype=vextvol.dtype)
            vextvol = a1 * vextvol + a2 * noise

            # apply A
            scale2 = jnp.exp((0.5 * dt) * vextvol)
            vel = vel / scale2

            x = x * (scale1 * scale2)
            extvol = extvol * (scale1 * scale2) ** 3
            cell = system["cell"] * (scale1 * scale2)
            if nbeads is not None:
                x = jnp.concatenate((x[None], eigx), axis=0)
                vel = jnp.concatenate((vel[None], eigv), axis=0)

            piston_temperature = au.KELVIN*masspiston * vextvol ** 2
            barostat_state = {
                **barostat_state,
                "rng_key": rng_key,
                "vextvol": vextvol,
                "extvol": extvol,
                "piston_temperature": piston_temperature,
            }
            return (
                x,
                vel,
                {
                    "barostat": barostat_state,
                    "cell": cell,
                    "thermostat": thermostat_state,
                },
            )


    elif barostat_name in ["NONE"]:
        estimate_pressure = False

        def barostat(x, vel, system):
            vel, thermostat_state = thermostat(vel, system["thermostat"])
            return x, vel, {**system, "thermostat": thermostat_state}

    else:
        raise ValueError(f"Unknown barostat {barostat_name}")

    return barostat, estimate_pressure, state
