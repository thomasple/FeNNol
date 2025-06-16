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
