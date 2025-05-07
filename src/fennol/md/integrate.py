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
from .barostats import get_barostat
from .colvars import setup_colvars
from .spectra import initialize_ir_spectrum

from copy import deepcopy
from .initial import initialize_system


def initialize_dynamics(
    simulation_parameters, system_data, conformation, model, fprec, rng_key
):
    step, update_conformation, dyn_state, thermo_state, vel = initialize_integrator(
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
    return step, update_conformation, dyn_state, {**system, **thermo_state}


def initialize_integrator(simulation_parameters, system_data, model, fprec, rng_key):
    dt = simulation_parameters.get("dt") * au.FS
    dt2 = 0.5 * dt
    nbeads = system_data.get("nbeads", None)

    mass = system_data["mass"]
    totmass_amu = system_data["totmass_amu"]
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
    thermostat_rng, rng_key = jax.random.split(rng_key)
    thermostat, thermostat_post, thermostat_state, vel, dyn_state["thermostat_name"] = (
        get_thermostat(simulation_parameters, dt, system_data, fprec, thermostat_rng)
    )

    do_thermostat_post = thermostat_post is not None
    if do_thermostat_post:
        thermostat_post, post_state = thermostat_post
        dyn_state["thermostat_post_state"] = post_state

    pbc_data = system_data.get("pbc", None)
    if pbc_data is not None:
        thermo_update, variable_cell, barostat_state = get_barostat(
            thermostat, simulation_parameters, dt, system_data, fprec, rng_key
        )
        estimate_pressure = variable_cell or pbc_data["estimate_pressure"]
        thermo_state = {"thermostat": thermostat_state, "barostat": barostat_state}

    else:
        estimate_pressure = False
        variable_cell = False

        def thermo_update(x, v, system):
            v, thermostat_state = thermostat(v, system["thermostat"])
            return x, v, {**system, "thermostat": thermostat_state}

        thermo_state = {"thermostat": thermostat_state}

    print("# Estimate pressure: ", estimate_pressure)

    dyn_state["estimate_pressure"] = estimate_pressure
    dyn_state["variable_cell"] = variable_cell

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

    ### ENSEMBLE
    ensemble_key = simulation_parameters.get("etot_ensemble_key", None)

    ### colvars
    colvars_definitions = simulation_parameters.get("colvars", None)
    use_colvars = colvars_definitions is not None
    if use_colvars:
        colvars_calculators = setup_colvars(colvars_definitions)

    ### ir spectrum
    do_ir_spectrum = simulation_parameters.get("ir_spectrum", False)
    assert isinstance(do_ir_spectrum, bool), "ir_spectrum must be a boolean"
    if do_ir_spectrum:
        is_qtb=dyn_state["thermostat_name"].endswith("QTB")
        model_ir, ir_state, save_dipole, ir_post = initialize_ir_spectrum(simulation_parameters,system_data,fprec,dt,is_qtb)
        dyn_state["ir_spectrum"] = ir_state

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
        def update_conformation(conformation, system):
            eigx = system["coordinates"]
            """update bead coordinates from ring polymer normal modes"""
            x = jnp.einsum("in,n...->i...", eigmat, eigx).reshape(nbeads * nat, 3) * (
                nbeads**0.5
            )
            conformation = {**conformation, "coordinates": x}
            if variable_cell:
                conformation["cells"] = system["cell"][None, :, :].repeat(nbeads, axis=0)
                conformation["reciprocal_cells"] = jnp.linalg.inv(system["cell"])[
                    None, :, :
                ].repeat(nbeads, axis=0)
            return conformation

        @jax.jit
        def coords_to_eig(x):
            """update normal modes from bead coordinates"""
            return jnp.einsum("in,i...->n...", eigmat, x.reshape(nbeads, nat, 3)) * (
                1.0 / nbeads**0.5
            )

        def halfstep_free_polymer(eigx0, eigv0):
            """update coordinates and velocities of a free ring polymer for a half time step"""
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
            eigx, eigv, system = thermo_update(eigx, eigv, system)
            eigx, eigv = halfstep_free_polymer(eigx, eigv)

            return {
                **system,
                "coordinates": eigx,
                "vel": eigv,
            }

        @jax.jit
        def update_forces(system, conformation):
            if estimate_pressure:
                epot, f, vir_t, out = model._energy_and_forces_and_virial(
                    model.variables, conformation
                )
                epot = epot / model_energy_unit
                f = f / model_energy_unit
                vir_t = vir_t / model_energy_unit

                new_sys = {
                    **system,
                    "forces": coords_to_eig(f),
                    "epot": jnp.mean(epot),
                    "virial": jnp.mean(vir_t, axis=0),
                }
            else:
                epot, f, out = model._energy_and_forces(model.variables, conformation)
                epot = epot / model_energy_unit
                f = f / model_energy_unit
                new_sys = {**system, "forces": coords_to_eig(f), "epot": jnp.mean(epot)}
            if ensemble_key is not None:
                kT = system_data["kT"]
                dE = (
                    jnp.mean(out[ensemble_key], axis=0) / model_energy_unit
                    - new_sys["epot"]
                )
                new_sys["ensemble_weights"] = -dE / kT
            
            if "total_dipole" in out:
                new_sys["total_dipole"] = jnp.mean(out["total_dipole"], axis=0)

            if use_colvars:
                coords = system["coordinates"][0]
                colvars = {}
                for colvar_name, colvar_calc in colvars_calculators.items():
                    colvars[colvar_name] = colvar_calc(coords)
                new_sys["colvars"] = colvars

            return new_sys,out

        @jax.jit
        def stepB(system):
            eigv = system["vel"] + dt2m * system["forces"]

            ek_c = 0.5 * jnp.sum(
                mass[:, None, None] * eigv[0, :, :, None] * eigv[0, :, None, :], axis=0
            )
            ek = ek_c - 0.5 * jnp.sum(
                system["coordinates"][1:, :, :, None]
                * system["forces"][1:, :, None, :],
                axis=(0, 1),
            )
            system = {
                **system,
                "vel": eigv,
                "ek_tensor": ek,
                "ek_c": jnp.trace(ek_c),
                "ek": jnp.trace(ek),
            }

            if estimate_pressure:
                vir = system["virial"]
                volume = jnp.abs(jnp.linalg.det(system["cell"]))
                Pres = (2 * ek - vir) / volume
                system["pressure_tensor"] = Pres
                system["pressure"] = jnp.trace(Pres) * (1.0 / 3.0)
                if variable_cell:
                    density = totmass_amu / volume
                    system["density"] = density
                    system["volume"] = volume

            return system

    else:
        nreplicas = system_data.get("nreplicas", 1)
        ### CLASSICAL MD INTEGRATOR
        @jax.jit
        def update_conformation(conformation, system):
            conformation = {**conformation, "coordinates": system["coordinates"]}
            if variable_cell:
                conformation["cells"] = system["cell"][None, :, :].repeat(nreplicas, axis=0)
                conformation["reciprocal_cells"] = jnp.linalg.inv(system["cell"])[
                    None, :, :
                ].repeat(nreplicas, axis=0)
            return conformation

        @jax.jit
        def stepA(system):
            v = system["vel"]
            f = system["forces"]
            x = system["coordinates"]

            v = v + f * dt2m
            x = x + dt2 * v
            x, v, system = thermo_update(x, v, system)
            x = x + dt2 * v

            return {**system, "coordinates": x, "vel": v}

        @jax.jit
        def update_forces(system, conformation):
            if estimate_pressure:
                epot, f, vir_t, out = model._energy_and_forces_and_virial(
                    model.variables, conformation
                )
                epot = epot / model_energy_unit
                f = f / model_energy_unit
                vir_t = vir_t / model_energy_unit
                new_sys = {
                    **system,
                    "forces": f,
                    "epot": jnp.mean(epot),
                    "virial": jnp.mean(vir_t, axis=0),
                }
            else:
                epot, f, out = model._energy_and_forces(model.variables, conformation)
                epot = epot / model_energy_unit
                f = f / model_energy_unit
                new_sys = {**system, "forces": f, "epot": jnp.mean(epot)}

            if ensemble_key is not None:
                kT = system_data["kT"]
                dE = jnp.mean(out[ensemble_key],axis=0) / model_energy_unit - new_sys["epot"]
                new_sys["ensemble_weights"] = -dE / kT
            
            if "total_dipole" in out:
                new_sys["total_dipole"] = out["total_dipole"][0]

            if use_colvars:
                coords = system["coordinates"][0]
                colvars = {}
                for colvar_name, colvar_calc in colvars_calculators.items():
                    colvars[colvar_name] = colvar_calc(coords)
                new_sys["colvars"] = colvars
                
            return new_sys,out

        @jax.jit
        def stepB(system):
            v = system["vel"]
            f = system["forces"]
            state_th = system["thermostat"]

            v = v + f * dt2m
            # ek = 0.5 * jnp.sum(mass[:, None] * v**2) / state_th.get("corr_kin", 1.0)
            ek_tensor = (
                (0.5 / nreplicas / state_th.get("corr_kin", 1.0) )
                * jnp.sum(mass[:, None, None] * v[:, :, None] * v[:, None, :], axis=0)
            )
            system = {
                **system,
                "vel": v,
                "ek": jnp.trace(ek_tensor),
                "ek_tensor": ek_tensor,
            }

            if estimate_pressure:
                vir = system["virial"]
                volume = jnp.abs(jnp.linalg.det(system["cell"]))
                Pres = (2 * ek_tensor - vir) / volume
                system["pressure_tensor"] = Pres
                system["pressure"] = jnp.trace(Pres) * (1.0 / 3.0)
                if variable_cell:
                    density = totmass_amu / volume
                    system["density"] = density
                    system["volume"] = volume

            return system
        
    if do_ir_spectrum:
        # @jax.jit
        # def update_dipole(ir_state,system,conformation):
        #     def mumodel(coords):
        #         out = model_ir._apply(model_ir.variables,{**conformation,"coordinates":coords})
        #         if nbeads is None:
        #             return out["total_dipole"][0]
        #         return out["total_dipole"].sum(axis=0)
        #     dmudqmodel = jax.jacobian(mumodel)

        #     dmudq = dmudqmodel(conformation["coordinates"])
        #     # print(dmudq.shape)
        #     if nbeads is None:
        #         vel = system["vel"].reshape(-1,1,nat,3)[0]
        #         mudot = (vel*dmudq).sum(axis=(1,2))
        #     else:
        #         dmudq = dmudq.reshape(3,nbeads,nat,3)#.mean(axis=1)
        #         vel = (jnp.einsum("in,n...->i...", eigmat, system["vel"]) *  nbeads**0.5
        #         )
        #         # vel = system["vel"][0].reshape(1,nat,3)
        #         mudot = (vel[None,...]*dmudq).sum(axis=(1,2,3))/nbeads


        #     ir_state = save_dipole(mudot,ir_state)
        #     return ir_state
        @jax.jit
        def update_conformation_ir(conformation, system):
            conformation = {**conformation, "coordinates": system["coordinates"].reshape(-1,nat,3)[0],"natoms":jnp.asarray([nat]),"batch_index":jnp.asarray([0]*nat),"species":jnp.asarray(system_data["species"].reshape(-1,nat)[0])}
            if variable_cell:
                conformation["cells"] = system["cell"][None, :, :]
                conformation["reciprocal_cells"] = jnp.linalg.inv(system["cell"])[
                    None, :, :
                ]
            return conformation

        @jax.jit
        def update_dipole(ir_state,system,conformation):
            if model_ir is not None:
                out = model_ir._apply(model_ir.variables,conformation)
                q = out.get("charges",jnp.zeros(nat)).reshape((-1,nat))
                dip = out.get("dipoles",jnp.zeros((nat,3))).reshape((-1,nat,3))
            else:
                q = system.get("charges",jnp.zeros(nat)).reshape((-1,nat))
                dip = system.get("dipoles",jnp.zeros((nat,3))).reshape((-1,nat,3))
            if nbeads is not None:
                q = jnp.mean(q,axis=0)
                dip = jnp.mean(dip,axis=0)
                vel = system["vel"][0]
                pos = system["coordinates"][0]
            else:
                q = q[0]
                dip = dip[0]
                vel = system["vel"].reshape(-1,nat,3)[0]
                pos = system["coordinates"].reshape(-1,nat,3)[0]
            
            if pbc_data is not None:
                cell_reciprocal = (conformation["cells"][0],conformation["reciprocal_cells"][0])
            else:
                cell_reciprocal = None
            
            ir_state = save_dipole(q,vel,pos,dip.sum(axis=0),cell_reciprocal,ir_state)
            return ir_state
            

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

        if do_ir_spectrum and "conformation_ir" not in dyn_state:
            if model_ir is not None:
                # initialize ir conformation
                dyn_state["preproc_state_ir"], dyn_state["conformation_ir"] = model_ir.preprocessing(model_ir.preproc_state, update_conformation_ir(conformation, system))
            else:
                dyn_state["conformation_ir"] = None

        
        ### update conformation and graphs
        nblist_countdown = dyn_state["nblist_countdown"]
        if nblist_countdown <= 0 or force_preprocess or (istep < nblist_warmup):
            ### full nblist update
            dyn_state["nblist_countdown"] = nblist_stride - 1
            conformation = model.preprocessing.process(
                preproc_state, update_conformation(conformation, system)
            )
            preproc_state, state_up, conformation, overflow = (
                model.preprocessing.check_reallocate(preproc_state, conformation)
            )
            if nblist_verbose and overflow:
                print("step", istep, ", nblist overflow => reallocating nblist")
                print("size updates:", state_up)

            if do_ir_spectrum and model_ir is not None:
                conformation_ir = model_ir.preprocessing.process(
                    dyn_state["preproc_state_ir"], update_conformation_ir(dyn_state["conformation_ir"], system)
                )
                dyn_state["preproc_state_ir"], _, dyn_state["conformation_ir"], overflow = (
                    model_ir.preprocessing.check_reallocate(dyn_state["preproc_state_ir"], conformation_ir)
                )
            

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
                update_conformation(conformation, system)
            )
            if do_ir_spectrum and model_ir is not None:
                dyn_state["conformation_ir"] = model_ir.preprocessing.update_skin(
                    update_conformation_ir(dyn_state["conformation_ir"], system)
                )

            if print_timings:
                conformation["coordinates"].block_until_ready()
                timings["Skin update"] += time.time() - tstep0
                tstep0 = time.time()

        ## compute forces
        system,out = update_forces(system, conformation)
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
        
        if do_ir_spectrum:
            ir_state = update_dipole(dyn_state["ir_spectrum"],system,dyn_state["conformation_ir"])
            dyn_state["ir_spectrum"] = ir_post(ir_state)

        if print_timings:
            system["coordinates"].block_until_ready()
            timings["Integrator"] += time.time() - tstep0
            tstep0 = time.time()

            # store timings
            dyn_state["timings"] = timings

        return dyn_state, system, conformation, preproc_state, out

    return step, update_conformation, dyn_state, thermo_state, vel
