import sys, os, io
import argparse
import time
from pathlib import Path
import math

import numpy as np
from typing import Optional, Callable
from collections import defaultdict
from functools import partial
import jax
import jax.numpy as jnp

from flax.core import freeze, unfreeze

from ..utils.io import last_xyz_frame


from ..models import FENNIX

from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .thermostats import get_thermostat

from copy import deepcopy
from .initial import initialize_system


def initialize_dynamics(
    simulation_parameters, system_data, conformation, model, fprec, rng_key
):
    step, update_conformation, dyn_state, thermostat_state, vel = initialize_integrator(
        simulation_parameters, system_data, model, fprec, rng_key
    )
    ### initialize system
    system = initialize_system(
        conformation,
        vel,
        model,
        system_data,
        fprec,
    )
    system["thermostat"] = thermostat_state
    return step, update_conformation, dyn_state, system


def initialize_integrator(simulation_parameters, system_data, model, fprec, rng_key):
    dt = simulation_parameters.get("dt") * au.FS
    dt2 = 0.5 * dt
    nbeads = system_data.get("nbeads", None)

    mass = system_data["mass"]
    nat = system_data["nat"]
    dt2m = jnp.asarray(dt2 / mass[:, None], dtype=fprec)
    if nbeads is not None:
        dt2m = dt2m[None, :, :]

    dyn_state = {
        "istep": 0,
        "dt": dt,
        "pimd": nbeads is not None,
    }

    model_energy_unit = au.get_multiplier(model.energy_unit)

    # initialize thermostat
    thermostat, thermostat_post, thermostat_state, vel, dyn_state["thermostat_name"] = (
        get_thermostat(simulation_parameters, dt, system_data, fprec, rng_key)
    )

    do_thermostat_post = thermostat_post is not None
    if do_thermostat_post:
        thermostat_post, post_state = thermostat_post
        dyn_state["thermostat_post_state"] = post_state

    pbc_data = system_data.get("pbc", None)
    if pbc_data is not None:
        pscale = pbc_data["pscale"]
        estimate_pressure = pbc_data["estimate_pressure"]
    else:
        pscale = 1.0
        estimate_pressure = False
        
    print("# Estimate pressure: ", estimate_pressure)

    dyn_state["estimate_pressure"] = estimate_pressure

    dyn_state["print_timings"] = simulation_parameters.get("print_timings", False)
    if dyn_state["print_timings"]:
        dyn_state["timings"] = defaultdict(lambda: 0.0)

    ### NBLIST
    nblist_verbose = simulation_parameters.get("nblist_verbose", False)
    nblist_stride = int(simulation_parameters.get("nblist_stride", -1))
    nblist_warmup_time = simulation_parameters.get("nblist_warmup_time", -1.0) * au.FS
    nblist_warmup = int(nblist_warmup_time / dt) if nblist_warmup_time > 0 else 0
    nblist_skin = simulation_parameters.get("nblist_skin", -1.0)
    if nblist_skin > 0:
        if nblist_stride <= 0:
            ## reference skin parameters at 300K (from Tinker-HP)
            ##   => skin of 2 A gives you 40 fs without complete rebuild
            t_ref = 40.0  # FS
            nblist_skin_ref = 2.0  # A
            nblist_stride = int(math.floor(nblist_skin / nblist_skin_ref * t_ref / dt))
        print(
            f"# nblist_skin: {nblist_skin:.2f} A, nblist_stride: {nblist_stride} steps, nblist_warmup: {nblist_warmup} steps"
        )

    if nblist_skin <= 0:
        nblist_stride = 1

    dyn_state["nblist_countdown"] = 0
    dyn_state["print_skin_activation"] = nblist_warmup > 0

    ### DEFINE INTEGRATION FUNCTIONS

    if nbeads is not None:
        ### RING POLYMER INTEGRATOR
        cay_correction = simulation_parameters.get("cay_correction", True)
        omk = system_data["omk"]
        eigmat = system_data["eigmat"]
        cayfact = 1.0 / (4.0 + (dt * omk[1:, None, None]) ** 2) ** 0.5
        if cay_correction:
            axx = jnp.asarray(2 * cayfact)
            axv = jnp.asarray(dt * cayfact)
            avx = jnp.asarray(-dt * cayfact * omk[1:, None, None] ** 2)
        else:
            axx = jnp.asarray(np.cos(omk[1:, None, None] * dt2))
            axv = jnp.asarray(np.sin(omk[1:, None, None] * dt2) / omk[1:, None, None])
            avx = jnp.asarray(-omk[1:, None, None] * np.sin(omk[1:, None, None] * dt2))


        @jax.jit
        def update_conformation(eigx):
            """ update bead coordinates from ring polymer normal modes"""
            return jnp.einsum("in,n...->i...", eigmat, eigx).reshape(
                nbeads * nat, 3
            ) * (nbeads**0.5)

        @jax.jit
        def coords_to_eig(x):
            """ update normal modes from bead coordinates"""
            return jnp.einsum("in,i...->n...", eigmat, x.reshape(nbeads, nat, 3)) * (
                1.0 / nbeads**0.5
            )
            
        def halfstep_free_polymer(eigx0, eigv0):
            """ update coordinates and velocities of a free ring polymer for a half time step"""
            eigx_c = eigx0[0] + dt2 * eigv0[0]
            eigv_c = eigv0[0]
            eigx = eigx0[1:] * axx + eigv0[1:] * axv
            eigv = eigx0[1:] * avx + eigv0[1:] * axx

            return (
                jnp.concatenate((eigx_c[None], eigx), axis=0),
                jnp.concatenate((eigv_c[None], eigv), axis=0),
            )
        
        @jax.jit
        def stepA(system):
            eigx = system["coordinates"]
            eigv = system["vel"] + dt2m * system["forces"]
            eigx, eigv = halfstep_free_polymer(eigx, eigv)
            eigv, thermosat_state = thermostat(eigv, system["thermostat"])
            eigx, eigv = halfstep_free_polymer(eigx, eigv)

            return {
                **system,
                "coordinates": eigx,
                "vel": eigv,
                "thermostat": thermosat_state,
            }
        
        @jax.jit
        def update_forces(system, conformation):
            if estimate_pressure:
                epot, f, vir_t, _ = model._energy_and_forces_and_virial(
                    model.variables, conformation
                )
                epot = epot / model_energy_unit
                f = f / model_energy_unit
                vir_t = vir_t / model_energy_unit
                return {
                    **system,
                    "forces": coords_to_eig(f),
                    "epot": jnp.mean(epot),
                    "virial": jnp.trace(jnp.mean(vir_t, axis=0)),
                }
            else:
                epot, f, _ = model._energy_and_forces(model.variables, conformation)
                epot = epot / model_energy_unit
                f = f / model_energy_unit
                return {**system, "forces": coords_to_eig(f), "epot": jnp.mean(epot)}
        
        @jax.jit
        def stepB(system):
            eigv = system["vel"] + dt2m * system["forces"]

            ek_c = 0.5 * jnp.sum(mass[:, None] * eigv[0] ** 2)
            ek = ek_c - 0.5 * jnp.sum(system["coordinates"][1:] * system["forces"][1:])
            system = {**system, "vel": eigv, "ek": ek, "ek_c": ek_c}

            if estimate_pressure:
                vir = system["virial"]
                Pkin = (2 * pscale) * ek
                Pvir = (-pscale) * vir
                system["pressure"] = Pkin + Pvir

            return system

    else:
        ### CLASSICAL MD INTEGRATOR
        @jax.jit
        def update_conformation(coordinates):
            return coordinates

        @jax.jit
        def stepA(system):
            v = system["vel"]
            f = system["forces"]
            x = system["coordinates"]

            v = v + f * dt2m
            x = x + dt2 * v
            v, state_th = thermostat(v, system["thermostat"])
            x = x + dt2 * v

            return {**system, "coordinates": x, "vel": v, "thermostat": state_th}

        @jax.jit
        def update_forces(system, conformation):
            if estimate_pressure:
                epot, f, vir_t, _ = model._energy_and_forces_and_virial(
                    model.variables, conformation
                )
                epot = epot/ model_energy_unit
                f = f / model_energy_unit
                vir_t = vir_t / model_energy_unit
                return {
                    **system,
                    "forces": f,
                    "epot": epot[0],
                    "virial": jnp.trace(vir_t[0]),
                }
            else:
                epot, f, _ = model._energy_and_forces(model.variables, conformation)
                epot = epot / model_energy_unit
                f = f / model_energy_unit
                return {**system, "forces": f, "epot": epot[0]}

        @jax.jit
        def stepB(system):
            v = system["vel"]
            f = system["forces"]
            state_th = system["thermostat"]

            v = v + f * dt2m
            ek = 0.5 * jnp.sum(mass[:, None] * v**2) / state_th.get("corr_kin", 1.0)
            system = {
                **system,
                "vel": v,
                "ek": ek,
            }

            if estimate_pressure:
                vir = system["virial"]
                Pkin = (2 * pscale) * ek
                Pvir = (-pscale) * vir
                system["pressure"] = Pkin + Pvir

            return system

    ### DEFINE STEP FUNCTION COMMON TO CLASSICAL AND PIMD
    def step(
        istep, dyn_state, system, conformation, preproc_state, force_preprocess=False
    ):
        tstep0 = time.time()
        print_timings = "timings" in dyn_state

        dyn_state = {
            **dyn_state,
            "istep": dyn_state["istep"] + 1,
        }
        if print_timings:
            prev_timings = dyn_state["timings"]
            timings = defaultdict(lambda: 0.0)
            timings.update(prev_timings)

        ## take a half step (update positions, nblist and half velocities)
        system = stepA(system)

        if print_timings:
            system["coordinates"].block_until_ready()
            timings["Integrator"] += time.time() - tstep0
            tstep0 = time.time()

        ### update conformation and graphs
        nblist_countdown = dyn_state["nblist_countdown"]
        if nblist_countdown <= 0 or force_preprocess or (istep < nblist_warmup):
            ### full nblist update
            dyn_state["nblist_countdown"] = nblist_stride - 1
            conformation = model.preprocessing.process(
                preproc_state,
                {
                    **conformation,
                    "coordinates": update_conformation(system["coordinates"]),
                },
            )
            preproc_state, state_up, conformation, overflow = (
                model.preprocessing.check_reallocate(preproc_state, conformation)
            )
            if nblist_verbose and overflow:
                print("step", istep, ", nblist overflow => reallocating nblist")
                print("size updates:", state_up)

            if print_timings:
                conformation["coordinates"].block_until_ready()
                timings["Preprocessing"] += time.time() - tstep0
                tstep0 = time.time()

        else:
            ### skin update
            if dyn_state["print_skin_activation"]:
                if nblist_verbose:
                    print(
                        "step",
                        istep,
                        ", end of nblist warmup phase => activating skin updates",
                    )
                dyn_state["print_skin_activation"] = False

            dyn_state["nblist_countdown"] = nblist_countdown - 1
            conformation = model.preprocessing.update_skin(
                {
                    **conformation,
                    "coordinates": update_conformation(system["coordinates"]),
                }
            )

            if print_timings:
                conformation["coordinates"].block_until_ready()
                timings["Skin update"] += time.time() - tstep0
                tstep0 = time.time()

        ## compute forces
        system = update_forces(system, conformation)
        if print_timings:
            system["coordinates"].block_until_ready()
            timings["Forces"] += time.time() - tstep0
            tstep0 = time.time()

        ## finish step
        system = stepB(system)

        ## end of step update (mostly for adQTB)
        if do_thermostat_post:
            system["thermostat"], dyn_state["thermostat_post_state"] = thermostat_post(
                system["thermostat"], dyn_state["thermostat_post_state"]
            )

        if print_timings:
            system["coordinates"].block_until_ready()
            timings["Integrator"] += time.time() - tstep0
            tstep0 = time.time()

            # store timings
            dyn_state["timings"] = timings

        return dyn_state, system, conformation, preproc_state

    return step, update_conformation, dyn_state, thermostat_state, vel
