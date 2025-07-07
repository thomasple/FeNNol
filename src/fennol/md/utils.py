import numpy as np
import jax
import jax.numpy as jnp
import pickle
from ..utils import AtomicUnits as au
from functools import partial

def get_restart_file(system_data):
    restart_file = system_data["name"] + ".dyn.restart"
    return restart_file


def load_dynamics_restart(system_data):
    with open(get_restart_file(system_data), "rb") as f:
        restart_data = pickle.load(f)

    assert (
        restart_data["nat"] == system_data["nat"]
    ), f"Restart file does not match system data 'nat'"
    assert restart_data.get("nbeads", 0) == system_data.get(
        "nbeads", 0
    ), f"Restart file does not match system data 'nbeads'"
    assert restart_data.get("nreplicas", 0) == system_data.get(
        "nreplicas", 0
    ), f"Restart file does not match system data 'nreplicas'"
    assert np.all(
        restart_data["species"] == system_data["species"]
    ), f"Restart file does not match system data 'species'"

    return restart_data


def save_dynamics_restart(system_data, conformation, dyn_state, system):
    restart_data = {
        "nat": system_data["nat"],
        "nbeads": system_data.get("nbeads", 0),
        "nreplicas": system_data.get("nreplicas", 0),
        "species": system_data["species"],
        "coordinates": conformation["coordinates"],
        "vel": system["vel"],
        "preproc_state": dyn_state["preproc_state"],
        "simulation_time_ps": dyn_state["start_time_ps"] + (dyn_state["dt"]*1e-3)* dyn_state["istep"],
    }
    if "cells" in conformation:
        restart_data["cells"] = conformation["cells"]

    

    restart_file = get_restart_file(system_data)
    with open(restart_file, "wb") as f:
        pickle.dump(restart_data, f)


@partial(jax.jit,static_argnames=["wrap_groups"])
def wrapbox(x, cell, reciprocal_cell, wrap_groups = None):
    q = x @ reciprocal_cell
    if wrap_groups is not None:
        for (group, indices) in wrap_groups:
            if group == "__other":
                q = q.at[indices].add(-jnp.floor(q[indices]))
            else:
                com = jnp.mean(q[indices], axis=0)
                shift = -jnp.floor(com)[None, :]
                q = q.at[indices].add(shift)
    else:
        q = q - jnp.floor(q)
    return q @ cell


def test_pressure_fd(system_data, conformation, model, verbose=True):
    model_energy_unit = au.get_multiplier(model.energy_unit)
    nat = system_data["nat"]
    volume = system_data["pbc"]["volume"]
    coordinates = conformation["coordinates"]
    cell = conformation["cells"][0]
    # temper = 2 * ek / (3.0 * nat) * au.KELVIN
    ek = 1.5 * nat * system_data["kT"]
    Pkin = (2 * au.KBAR) * ek / ((3.0 / au.BOHR**3) * volume)
    e, f, vir_t, _ = model._energy_and_forces_and_virial(model.variables, conformation)
    KBAR = au.KBAR / model_energy_unit
    Pvir = -(np.trace(vir_t[0]) * KBAR) / ((3.0 / au.BOHR**3) * volume)
    vstep = volume * 0.000001
    scalep = ((volume + vstep) / volume) ** (1.0 / 3.0)
    cellp = cell * scalep
    reciprocal_cell = np.linalg.inv(cellp)
    sysp = model.preprocess(
        **{
            **conformation,
            "coordinates": coordinates * scalep,
            "cells": cellp[None, :, :],
            "reciprocal_cells": reciprocal_cell[None, :, :],
        }
    )
    ep, _ = model._total_energy(model.variables, sysp)
    scalem = ((volume - vstep) / volume) ** (1.0 / 3.0)
    cellm = cell * scalem
    reciprocal_cell = np.linalg.inv(cellm)
    sysm = model.preprocess(
        **{
            **conformation,
            "coordinates": coordinates * scalem,
            "cells": cellm[None, :, :],
            "reciprocal_cells": reciprocal_cell[None, :, :],
        }
    )
    em, _ = model._total_energy(model.variables, sysm)
    Pvir_fd = -(ep[0] * KBAR - em[0] * KBAR) / (2.0 * vstep / au.BOHR**3)
    if verbose:
        print(
            f"# Initial pressure: {Pkin+Pvir:.3f} (virial); {Pkin+Pvir_fd:.3f} (finite difference) ; Pkin: {Pkin:.3f} ; Pvir: {Pvir:.3f} ; Pvir_fd: {Pvir_fd:.3f}"
        )
    return Pkin, Pvir, Pvir_fd


def optimize_fire2(
    x0,
    ef_func,
    atol=1e-4,
    dt=0.002,
    logoutput=True,
    Nmax=10000,
    keep_every=-1,
    max_disp=None,
):
    """Fast inertial relaxation engine (FIRE)
    adapted from https://github.com/elvissoares/PyFIRE
    """
    # global variables
    alpha0 = 0.1
    Ndelay = 5
    finc = 1.1
    fdec = 0.5
    fa = 0.99
    Nnegmax = Nmax // 5

    error = 10 * atol
    dtmax = dt * 10.0
    dtmin = 0.02 * dt
    alpha = alpha0
    Npos = 0
    Nneg = 0

    nat = x0.shape[0]
    x = x0.copy()
    e, F = ef_func(x)
    V = -0.5 * dt * F  # initial velocity
    P=0.
    maxF = np.max(np.abs(F))
    rmsF = np.sqrt(3 * np.mean(F**2))
    error = rmsF
    if error < atol:
        return x

    if logoutput:
        print(
            f"{'#Step':>10} {'Energy':>15} {'RMS Force':>15} {'MAX Force':>15} {'dt':>15} {'Power':>15}"
        )
    if logoutput:
        print(f"{0:10d} {e:15.5f} {rmsF:15.5f} {maxF:15.5f} {dt:15.5f} {0:15.5f}")

    if keep_every > 0:
        x_keep = [x.copy()]

    if max_disp is not None:
        assert max_disp > 0, "max_disp must be positive"

    success = False
    for i in range(Nmax):

        V = V + dt * F
        V = (1 - alpha) * V + alpha * F * np.linalg.norm(V) / np.linalg.norm(F)
        if max_disp is not None:
            V = np.clip(V, -max_disp / dt, max_disp / dt)
        x = x + dt * V
        e, F = ef_func(x)

        maxF = np.max(np.abs(F))
        rmsF = np.sqrt(3 * np.mean(F**2))
        error = rmsF

        if logoutput:
            print(f"{i+1:10d} {e:15.5f} {rmsF:15.5f} {maxF:15.5f} {dt:15.5f} {P:15.5f}")

        if error <= atol:
            success = True
            break
        
        P = (F * V).sum()  # dissipated power

        if P > 0:
            Npos = Npos + 1
            Nneg = 0
            if Npos > Ndelay:
                dt = min(dt * finc, dtmax)
                alpha = alpha * fa
        else:
            Npos = 0
            Nneg = Nneg + 1
            if Nneg > Nnegmax:
                break
            if i >= Ndelay:
                dt = max(dt * fdec, dtmin)
                alpha = alpha0
            x = x - 0.5 * dt * V
            V = np.zeros(x.shape)
        
        if keep_every > 0 and (i + 1) % keep_every == 0:
            x_keep.append(x.copy())

    if keep_every > 0:
        x_keep.append(x.copy())
        return x, success, x_keep
    return x, success

