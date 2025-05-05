import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import optax
from enum import Enum

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
    if barostat_name != "NONE":
        assert (
            target_pressure is not None
        ), "target_pressure must be specified for NPT/NPH simulations"
        target_pressure = target_pressure / au.BOHR**3

    nbeads = system_data.get("nbeads", None)
    variable_cell = True

    anisotropic = simulation_parameters.get("aniso_barostat", False)
    
    isotropic = not anisotropic

    pbc_data = system_data["pbc"]

    if barostat_name in ["LGV", "LANGEVIN"]:
        assert rng_key is not None, "rng_key must be provided for QTB barostat"
        gamma = simulation_parameters.get("gamma_piston", 20.0 / au.THZ) / au.FS
        tau_piston = simulation_parameters.get("tau_piston", 200.0 / au.FS) * au.FS
        nat = system_data["nat"]
        masspiston = 3 * nat * kT * tau_piston**2
        print(f"# LANGEVIN barostat with piston mass={masspiston:.1e} Ha.fs^2")
        a1 = math.exp(-gamma * dt)
        a2 = ((1 - a1 * a1) * kT / masspiston) ** 0.5

        start_barostat = simulation_parameters.get("start_barostat", 0.0)*au.FS
        istart_barostat = int(round(start_barostat / dt))
        if istart_barostat > 0:
            print(
                f"# LANGEVIN barostat will start at {start_barostat/1000:.3f} ps ({istart_barostat} steps)"
            )
        else:
            istart_barostat = 0

        rng_key, v_key = jax.random.split(rng_key)
        if anisotropic:
            extvol = pbc_data["cell"]
            vextvol = (
                jax.random.normal(v_key, (3, 3), dtype=extvol.dtype)
                * (kT / masspiston) ** 0.5
            )
            vextvol = 0.5 * (vextvol + vextvol.T)

            aniso_mask = simulation_parameters.get(
                "aniso_mask", [True, True, True, True, True, True]
            )
            assert len(aniso_mask) == 6, "aniso_mask must have 6 elements"
            aniso_mask = np.array(aniso_mask, dtype=bool).astype(np.int32)
            ndof_piston = np.sum(aniso_mask)
            # xx   yy   zz    xy   xz   yz
            aniso_mask = np.array(
                [
                    [aniso_mask[0], aniso_mask[3], aniso_mask[4]],
                    [aniso_mask[3], aniso_mask[1], aniso_mask[5]],
                    [aniso_mask[4], aniso_mask[5], aniso_mask[2]],
                ]
            ,dtype=np.int32)
        else:
            extvol = jnp.asarray(pbc_data["volume"])
            vextvol = (
                jax.random.normal(v_key, (1,), dtype=extvol.dtype)
                * (kT / masspiston) ** 0.5
            )
            ndof_piston = 1.

        state["extvol"] = extvol
        state["vextvol"] = vextvol
        state["rng_key"] = rng_key
        state["istep"] = 0

        def barostat(x, vel, system):
            if nbeads is not None:
                x, eigx = x[0], x[1:]
                vel, eigv = vel[0], vel[1:]
            barostat_state = system["barostat"]
            extvol = barostat_state["extvol"]
            vextvol = barostat_state["vextvol"]
            cell = system["cell"]
            volume = jnp.abs(jnp.linalg.det(cell))

            istep = barostat_state["istep"] + 1
            dt_bar = dt * (istep >= istart_barostat)


            # apply B
            pV = 2 * (system["ek_tensor"] + jnp.trace(system["ek_tensor"])*jnp.eye(3)/(3*x.shape[0])) - system["virial"]
            if isotropic:
                dpV = jnp.trace(pV) - 3*volume * target_pressure
            else:
                dpV = 0.5 * (pV + pV.T) - volume * target_pressure * jnp.eye(3) 

            vextvol = vextvol + (dt_bar/masspiston) * dpV

            # apply A
            if isotropic:
                scale1 = jnp.exp((0.5 * dt_bar*(1+1./x.shape[0])) * vextvol)
                vel = vel / scale1
            else:
                vextvol = aniso_mask * vextvol
                l, O = jnp.linalg.eigh(vextvol + jnp.trace(vextvol) * jnp.eye(3)/(3*x.shape[0]))
                Dv = jnp.diag(jnp.exp(-0.5 * dt_bar * l))
                Dx = jnp.diag(jnp.exp(0.5 * dt_bar * l))
                scalev = O @ Dv @ O.T
                scale1 = O @ Dx @ O.T
                vel = vel @ scalev

            # apply O
            if nbeads is not None:
                eigv, thermostat_state = thermostat(
                    jnp.concatenate((vel[None], eigv), axis=0), system["thermostat"]
                )
                vel, eigv = eigv[0], eigv[1:]
            else:
                vel, thermostat_state = thermostat(vel, system["thermostat"])
            rng_key, noise_key = jax.random.split(barostat_state["rng_key"])

            if isotropic:
                noise = jax.random.normal(noise_key, (1,), dtype=vextvol.dtype)
            else:
                noise = jax.random.normal(noise_key, (3, 3), dtype=vextvol.dtype)
                noise = 0.5 * (noise + noise.T)

            vextvol = a1 * vextvol + a2 * noise

            # apply A
            if isotropic:
                scale2 = jnp.exp((0.5 * dt_bar*(1+1./x.shape[0])) * vextvol)
                # vel = vel * scale1
                vel = vel / scale2
                x = x * (scale1 * scale2)
                extvol = extvol * (scale1 * scale2) ** 3
                cell = cell * (scale1 * scale2)
            else:
                vextvol = aniso_mask * vextvol
                l, O = jnp.linalg.eigh(vextvol + jnp.trace(vextvol) * jnp.eye(3)/(3*x.shape[0]))
                Dv = jnp.diag(jnp.exp(-0.5 * dt_bar * l))
                Dx = jnp.diag(jnp.exp(0.5 * dt_bar * l))
                scalev = O @ Dv @ O.T
                scale = scale1 @ (O @ Dx @ O.T)
                vel = vel @ scalev
                x = x @ scale
                extvol = extvol @ scale
                cell = extvol

            if nbeads is not None:
                x = jnp.concatenate((x[None], eigx), axis=0)
                vel = jnp.concatenate((vel[None], eigv), axis=0)

            piston_temperature = (au.KELVIN * masspiston/ndof_piston) * jnp.sum(vextvol**2)
            barostat_state = {
                **barostat_state,
                "istep": istep,
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
        variable_cell = False

        def barostat(x, vel, system):
            vel, thermostat_state = thermostat(vel, system["thermostat"])
            return x, vel, {**system, "thermostat": thermostat_state}

    else:
        raise ValueError(f"Unknown barostat {barostat_name}")

    return barostat, variable_cell, state
