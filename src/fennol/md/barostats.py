import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import optax
from enum import Enum

from .utils import us


def get_barostat(
    thermostat,
    simulation_parameters,
    dt,
    system_data,
    fprec,
    rng_key=None,
    restart_data={},
):
    state = {}

    barostat_name = str(simulation_parameters.get("barostat", "NONE")).upper()
    """@keyword[fennol_md] barostat
    Type of barostat for pressure control (NONE, LGV, LANGEVIN).
    Default: NONE
    """

    kT = system_data.get("kT", None)
    assert kT is not None, "kT must be specified for NPT/NPH simulations"
    target_pressure = simulation_parameters.get("target_pressure")
    """@keyword[fennol_md] target_pressure
    Target pressure for NPT ensemble simulations.
    Required for barostat != NONE
    """
    if barostat_name != "NONE":
        assert (
            target_pressure is not None
        ), "target_pressure must be specified for NPT/NPH simulations"

    nbeads = system_data.get("nbeads", None)
    variable_cell = True

    anisotropic = simulation_parameters.get("aniso_barostat", False)
    """@keyword[fennol_md] aniso_barostat
    Use anisotropic barostat allowing independent cell parameter scaling.
    Default: False
    """

    isotropic = not anisotropic

    pbc_data = system_data["pbc"]
    start_barostat = simulation_parameters.get("start_barostat", 0.0)
    """@keyword[fennol_md] start_barostat
    Time delay before starting barostat pressure coupling.
    Default: 0.0
    """
    start_time = restart_data.get("simulation_time_ps", 0.0) / us.PS
    start_barostat = max(0.0, start_barostat - start_time)
    istart_barostat = int(round(start_barostat / dt))
    if istart_barostat > 0 and barostat_name not in ["NONE"]:
        print(
            f"# BAROSTAT will start at {start_barostat*us.PS:.3f} ps ({istart_barostat} steps)"
        )
    else:
        istart_barostat = 0

    if barostat_name in ["LGV", "LANGEVIN"]:
        assert rng_key is not None, "rng_key must be provided for QTB barostat"
        gamma = simulation_parameters.get("gamma_piston", 20.0 / us.THZ)
        """@keyword[fennol_md] gamma_piston
        Piston friction coefficient for Langevin barostat.
        Default: 20.0 ps^-1
        """
        tau_piston = simulation_parameters.get("tau_piston", 200.0 / us.FS)
        """@keyword[fennol_md] tau_piston
        Piston time constant for barostat coupling.
        Default: 200.0 fs
        """
        nat = system_data["nat"]
        masspiston = 3 * nat * kT * tau_piston**2
        print(
            f"# LANGEVIN barostat with piston mass={masspiston*us.get_multiplier('KCALPERMOL*PS^{2}'):.1e} kcal/mol.ps^2"
        )
        a1 = math.exp(-gamma * dt)
        a2 = ((1 - a1 * a1) * kT / masspiston) ** 0.5

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
            """@keyword[fennol_md] aniso_mask
            Mask for anisotropic barostat degrees of freedom [xx, yy, zz, xy, xz, yz].
            Default: [True, True, True, True, True, True]
            """
            assert len(aniso_mask) == 6, "aniso_mask must have 6 elements"
            aniso_mask = np.array(aniso_mask, dtype=bool).astype(np.int32)
            ndof_piston = np.sum(aniso_mask)
            # xx   yy   zz    xy   xz   yz
            aniso_mask = np.array(
                [
                    [aniso_mask[0], aniso_mask[3], aniso_mask[4]],
                    [aniso_mask[3], aniso_mask[1], aniso_mask[5]],
                    [aniso_mask[4], aniso_mask[5], aniso_mask[2]],
                ],
                dtype=np.int32,
            )
        else:
            extvol = jnp.asarray(pbc_data["volume"])
            vextvol = (
                jax.random.normal(v_key, (1,), dtype=extvol.dtype)
                * (kT / masspiston) ** 0.5
            )
            ndof_piston = 1.0

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
            # pV = 2 * (system["ek_tensor"] + jnp.trace(system["ek_tensor"])*jnp.eye(3)/(3*x.shape[0])) - system["virial"]
            ek = system["ek_c"] if nbeads is not None else system["ek"]
            Pres = (
                system["pressure_tensor"]
                + ek * jnp.array(np.eye(3) * (2 / (3 * x.shape[0]))) / volume
            )
            if isotropic:
                dPres = jnp.trace(Pres) - 3 * target_pressure
            else:
                dPres = 0.5 * (Pres + Pres.T) - jnp.array(target_pressure * np.eye(3))

            vextvol = vextvol + ((dt_bar / masspiston) * volume) * dPres

            # apply A
            if isotropic:
                vdt2 = 0.5 * dt_bar * vextvol
                scalev = jnp.exp(-vdt2 * (1 + 1.0 / x.shape[0]))
                vel = vel * scalev
                scale1 = jnp.exp(vdt2)
            else:
                vextvol = aniso_mask * vextvol
                vdt2 = 0.5 * dt_bar * vextvol
                l, O = jnp.linalg.eigh(vdt2)
                lcorr = jnp.trace(vdt2) / (3 * x.shape[0])
                Dv = jnp.diag(jnp.exp(-(l + lcorr)))
                Dx = jnp.diag(jnp.exp(l))
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
                vdt2 = 0.5 * dt_bar * vextvol
                scalev = jnp.exp(-vdt2 * (1 + 1.0 / x.shape[0]))
                vel = vel * scalev
                scale2 = jnp.exp(vdt2)
                x = x * (scale1 * scale2)
                extvol = extvol * (scale1 * scale2) ** 3
                cell = cell * (scale1 * scale2)
            else:
                vextvol = aniso_mask * vextvol
                vdt2 = 0.5 * dt_bar * vextvol
                l, O = jnp.linalg.eigh(vdt2)
                lcorr = jnp.trace(vdt2) / (3 * x.shape[0])
                Dv = jnp.diag(jnp.exp(-(l + lcorr)))
                Dx = jnp.diag(jnp.exp(l))
                scalev = O @ Dv @ O.T
                scale = scale1 @ (O @ Dx @ O.T)

                extvol = extvol @ scale

                # ensure cell is lower triangular
                extvol, rotation_matrix = tril_cell_(extvol)
                cell = extvol
                vextvol = rotation_matrix.T @ vextvol @ rotation_matrix

                # scale and rotate positions and velocities
                vel = vel @ (scalev @ rotation_matrix)
                x = x @ (scale @ rotation_matrix)

            if nbeads is not None:
                if not isotropic:
                    eigx = eigx @ rotation_matrix
                    eigv = eigv @ rotation_matrix
                x = jnp.concatenate((x[None], eigx), axis=0)
                vel = jnp.concatenate((vel[None], eigv), axis=0)

            piston_temperature = (us.KELVIN * masspiston / ndof_piston) * jnp.sum(
                vextvol**2
            )
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
                    **system,
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


def tril_cell_(cell):
    cell = jnp.asarray(cell, dtype=float).reshape(3, 3)
    a = jnp.linalg.norm(cell[0])
    b = jnp.linalg.norm(cell[1])
    c = jnp.linalg.norm(cell[2])
    cos_alpha = jnp.dot(cell[1], cell[2]) / (b * c)
    cos_beta = jnp.dot(cell[0], cell[2]) / (a * c)
    cos_gamma = jnp.dot(cell[0], cell[1]) / (a * b)
    cell_tril = cell_lengths_angles_to_tril(a, b, c, cos_alpha, cos_beta, cos_gamma)
    rotation = cell_tril @ jnp.linalg.inv(cell)
    return cell_tril, rotation


def cell_lengths_angles_to_tril(a, b, c, cos_alpha, cos_beta, cos_gamma):
    sin_gamma = jnp.sqrt(1.0 - cos_gamma * cos_gamma)

    # Build the cell vectors
    va = a * jnp.array([1, 0, 0])
    vb = b * jnp.array([cos_gamma, sin_gamma, 0])
    cx = cos_beta
    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1.0 - cx * cx - cy * cy
    cz = jnp.sqrt(cz_sqr)
    vc = c * jnp.array([cx, cy, cz])

    return jnp.vstack((va, vb, vc))
