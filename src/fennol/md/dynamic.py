import sys, os, io
import argparse
import time
from pathlib import Path
import math

import numpy as np
from typing import Optional, Callable
from functools import partial
import jax
import jax.numpy as jnp

from flax.core import freeze, unfreeze


from ..models import FENNIX
from ..utils.io import write_arc_frame, last_xyz_frame
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .thermostats import get_thermostat

from copy import deepcopy


def minmaxone(x, name=""):
    print(name, x.min(), x.max(), (x**2).mean() ** 0.5)


@jax.jit
def wrapbox(x, cell, reciprocal_cell):
    q = jnp.einsum("ij,sj->si", reciprocal_cell, x)
    q = q - jnp.floor(q)
    return jnp.einsum("ij,sj->si", cell, q)


def main():
    # os.environ["OMP_NUM_THREADS"] = "1"
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )
    ### Read the parameter file
    parser = argparse.ArgumentParser(prog="TinkerIO")
    parser.add_argument("param_file", type=Path, help="Parameter file")
    args = parser.parse_args()
    simulation_parameters = parse_input(args.param_file)

    ### Set the device
    device: str = simulation_parameters.get("device", "cpu")
    if device == "cpu":
        device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda"):
        num = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = num
        device = "gpu"

    _device = jax.devices(device)[0]
    jax.config.update("jax_default_device", _device)

    ### Set the precision
    enable_x64 = simulation_parameters.get("enable_x64", False)
    jax.config.update("jax_enable_x64", enable_x64)
    fprec = "float64" if enable_x64 else "float32"

    matmul_precision = simulation_parameters.get("matmul_prec", "high").lower()
    assert matmul_precision in [
        "default",
        "high",
        "highest",
    ], "matmul_prec must be one of 'default','high','highest'"
    jax.config.update("jax_default_matmul_precision", matmul_precision)

    # with jax.default_device(_device):
    dynamic(simulation_parameters, device, fprec)


def dynamic(simulation_parameters, device, fprec):
    ### Initialize the model
    # assert 'model_file' in simulation_parameters, "model_file not specified in parameter file"
    model_file = simulation_parameters.get("model_file")
    model_file = Path(str(model_file).strip())
    if not model_file.exists():
        raise FileNotFoundError(f"model file {model_file} not found")
    else:
        graph_config = simulation_parameters.get("graph_config", {})
        model = FENNIX.load(model_file, graph_config=graph_config)  # \
        print(f"model_file: {model_file}")

    if "energy_terms" in simulation_parameters:
        model.set_energy_terms(simulation_parameters["energy_terms"])
        print(model.energy_terms)

    ### Get the coordinates and species from the xyz file
    system_name = str(simulation_parameters.get("system", "system")).strip()
    indexed = simulation_parameters.get("xyz_input/indexed", True)
    box_info = simulation_parameters.get("xyz_input/box_info", False)
    crystal_input = simulation_parameters.get("xyz_input/crystal", False)
    xyzfile = Path(simulation_parameters.get("xyz_input/file", system_name + ".xyz"))
    if not xyzfile.exists():
        raise FileNotFoundError(f"xyz file {xyzfile} not found")
    system_name = str(simulation_parameters.get("system", xyzfile.stem)).strip()
    symbols, coordinates, _ = last_xyz_frame(
        xyzfile, indexed=indexed, box_info=box_info
    )
    coordinates = coordinates.astype(fprec)
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)
    nat = species.shape[0]
    mass_amu = np.array(ATOMIC_MASSES, dtype=fprec)[species]
    deuterate = simulation_parameters.get("deuterate", False)
    if deuterate:
        print("Replacing all hydrogens with deuteriums")
        mass_amu[species == 1] *= 2.0
    mass = mass_amu * (au.MPROT * (au.FS / au.BOHR) ** 2)
    totmass_amu = mass_amu.sum()

    ### Set boundary conditions
    # nblist_skin=simulation_parameters.get('nblist_skin',0.0)
    # nblist_refresh=simulation_parameters.get('nblist_refresh',1)
    # assert nblist_refresh > 0, "nblist_refresh must be > 0"
    # assert nblist_skin >= 0, "nblist_skin must be >= 0"
    # if nblist_refresh > 1: nblist_builder=NblistBuilder(model.nblist_builder.cutoff+nblist_skin).to(prec)

    cell = simulation_parameters.get("cell", None)
    if cell is not None:
        cell = np.array(cell, dtype=fprec).reshape(3, 3).T
        reciprocal_cell = np.linalg.inv(cell)
        volume = np.linalg.det(cell)
        print("cell matrix:")
        print(cell)
        dens = totmass_amu / 6.02214129e-1 / volume
        print("density: ", dens.item(), " g/cm^3")
        pscale = au.KBAR / (3.0 * volume / au.BOHR**3)

    if crystal_input:
        assert cell is not None, "cell must be specified for crystal units"
        coordinates = coordinates @ cell  # .double()
        with open("initial.arc", "w") as finit:
            write_arc_frame(finit, symbols, coordinates)

    estimate_pressure = simulation_parameters.get("estimate_pressure", False)
    if estimate_pressure:
        if cell is None:
            raise ValueError(
                "estimate_pressure requires cell to be specified in the input file"
            )
        # shift = jnp.einsum("ij,sj->si",cell,np.random.randint(-10,10,size=coordinates.shape))
        # print(shift)
        # coordinates = coordinates + shift

    ### Set simulation parameters
    dt = simulation_parameters.get("dt") * au.FS  # /au.FS
    dt2 = 0.5 * dt
    dt2m = jnp.asarray(dt2 / mass[:, None], dtype=fprec)

    nsteps = int(simulation_parameters.get("nsteps"))
    gamma = simulation_parameters.get("gamma", 1.0 / au.THZ) / au.FS
    kT = simulation_parameters.get("temperature", 300.0) / au.KELVIN
    start_time = 0.0
    start_step = 0

    ### Set I/O parameters
    Tdump = simulation_parameters.get("tdump", 1.0 / au.PS) * au.FS
    ndump = int(Tdump / dt)
    do_wrap_box = simulation_parameters.get("wrap_box", True) and cell is not None
    traj_file = Path(simulation_parameters.get("traj_file", system_name + ".arc"))

    random_seed = simulation_parameters.get(
        "random_seed", np.random.randint(0, 2**32 - 1)
    )
    print(f"random_seed: {random_seed}")
    rng_key = jax.random.PRNGKey(random_seed)

    state = {}
    ## window averaging
    window_size = simulation_parameters.get("tau_avg", -1.) * au.FS
    do_window_avg = window_size > 0
    
    if do_window_avg > 0:
        win_b1 = np.exp(-dt / window_size) if do_window_avg else 0.0
        state["window_avg"] = {}
        state["window_avg"]["n"] = 0
        state["window_avg"]["ek"] = 0.0
        state["window_avg"]["pressure"] = 0.0
        state["window_avg"]["Pkin"] = 0.0
        state["window_avg"]["Pvir"] = 0.0
        state["window_avg"]["epot"] = 0.0
        state["window_avg"]["ek_m"] = 0.0
        state["window_avg"]["pressure_m"] = 0.0
        state["window_avg"]["Pkin_m"] = 0.0
        state["window_avg"]["Pvir_m"] = 0.0
        state["window_avg"]["epot_m"] = 0.0

    ### Set the thermostat
    rng_key, t_key = jax.random.split(rng_key)
    thermostat_name = str(simulation_parameters.get("thermostat", "NONE")).upper()
    thermostat, thermostat_post, state["thermostat"], vel = get_thermostat(
        thermostat_name,
        rng_key=t_key,
        dt=dt,
        mass=mass,
        gamma=gamma,
        kT=kT,
        simulation_parameters=simulation_parameters,
        species=species,
    )
    do_thermostat_post = thermostat_post is not None
    if do_thermostat_post:
        thermostat_post, post_state = thermostat_post

    ### CONFIGURE PREPROCESSING
    minimum_image = simulation_parameters.get("minimum_image", True)
    nblist_verbose = simulation_parameters.get("nblist_verbose", True)
    nblist_stride = int(simulation_parameters.get("nblist_stride", -1))
    nblist_warmup_time = simulation_parameters.get("nblist_warmup_time", 0.)*au.FS
    nblist_warmup = int(nblist_warmup_time/dt)
    nblist_skin = simulation_parameters.get("nblist_skin", -1.)
    if nblist_skin > 0:
        if nblist_stride <= 0:
            ## reference skin parameters at 300K (from Tinker-HP) 
            ##   => skin of 2 A gives you 40 fs without complete rebuild
            t_ref = 40. # FS
            nblist_skin_ref = 2. # A
            nblist_stride = int(math.floor(nblist_skin/nblist_skin_ref*t_ref/dt))
        print(f"nblist_skin: {nblist_skin:.2f} A, nblist_stride: {nblist_stride} steps, nblist_warmup: {nblist_warmup} steps")

    if nblist_skin <= 0:
        nblist_stride = 1

    def configure_nblist(preproc_state, nblist_skin, nblist_stride=0):
        preproc_state = unfreeze(preproc_state)
        layer_state = []
        for st in preproc_state["layers_state"]:
            stnew = unfreeze(st)
            #     st["nblist_skin"] = nblist_skin
            #     if nblist_stride > 1:
            #         st["skin_stride"] = nblist_stride
            #         st["skin_count"] = nblist_stride
            if cell is not None:
                stnew["minimum_image"] = minimum_image
            if nblist_skin > 0:
                stnew["nblist_skin"] = nblist_skin
            if "nblist_mult_size" in simulation_parameters:
                stnew["nblist_mult_size"] = simulation_parameters["nblist_mult_size"]
            if "nblist_add_neigh" in simulation_parameters:
                stnew["add_neigh"] = simulation_parameters["nblist_add_neigh"]
            layer_state.append(freeze(stnew))
        preproc_state["layers_state"] = layer_state
        return freeze(preproc_state)

    preproc_state = configure_nblist(model.preproc_state, nblist_skin, nblist_stride)

    # nblist_updater = jax.jit(model.preprocessing.get_updaters())
    # if nblist_skin is not None:
    #     nblist_skin_updater = jax.jit(model.preprocessing.get_skin_updaters())
    graphs_keys = list(model._graphs_properties.keys())

    print("graphs_keys: ", graphs_keys)

    # @jax.jit
    # def nblist_overflow(system):
    #     """ check if any of the graphs has overflowed"""
    #     overflow = False
    #     for k in graphs_keys:
    #         graph = system[k]
    #         # if graph["overflow"]:
    #         #     return True
    #         overflow = jnp.logical_or(overflow,graph["overflow"])
    #     return overflow

    # def nblist_overflow(system):
    #     """ check if any of the graphs has overflowed"""
    #     for k in graphs_keys:
    #         graph = system[k]
    #         if graph["overflow"]:
    #             return k
    #     return None

    ## initial preprocessing
    preproc_state = preproc_state.copy({"check_input": True})
    system = dict(species=species, coordinates=coordinates)
    if cell is not None:
        system["cells"] = cell[None, :, :]
    preproc_state, system = model.preprocessing(preproc_state, system)

    preproc_state = preproc_state.copy({"check_input": False})
    # preproc_state["check_input"] = False
    if nblist_verbose:
        print("nblist state:",preproc_state)

    ### print model
    if simulation_parameters.get("print_model", False):
        print(model.summarize(example_data=system))
    ## initial energy and forces
    print("Computing initial energy and forces")
    e, f, output = model._energy_and_forces(model.variables, system)
    print(f"Initial energy: {e[0]} Ha")
    minmaxone(jnp.abs(f), "forces min/max/rms:")
    eat = np.array(output["atomic_energies"][:, None])
    f = np.array(f)
    # np.savetxt("initial_energy_forces.txt",np.column_stack((eat,f)),fmt="%.5f")

    # rng_key, v_key = jax.random.split(rng_key)
    # vel = (
    #     jax.random.normal(v_key, coordinates.shape) * (kT / mass[:, None]) ** 0.5
    # ).astype(fprec)
    ek = 0.5 * jnp.sum(mass[:, None] * vel**2)
    state["nwin_avg"] = 0
    state["vel"] = vel.astype(fprec)
    state["ek"] = ek
    state["forces"] = f
    state["epot"] = e

    ### Print initial pressure
    if estimate_pressure and fprec == "float64":
        Pkin = (2 * au.KBAR) * ek / ((3.0 / au.BOHR**3) * volume)
        e, f, vir_t, _ = model._energy_and_forces_and_virial(model.variables, system)
        Pvir = -(np.trace(vir_t[0]) * au.KBAR) / ((3.0 / au.BOHR**3) * volume)
        vstep = volume * 0.000001
        scalep = ((volume + vstep) / volume) ** (1.0 / 3.0)
        sysp = model.preprocess(
            **{
                **system,
                "coordinates": coordinates * scalep,
                "cells": cell[None, :, :] * scalep,
            }
        )
        ep, _ = model._total_energy(model.variables, sysp)
        scalem = ((volume - vstep) / volume) ** (1.0 / 3.0)
        sysm = model.preprocess(
            **{
                **system,
                "coordinates": coordinates * scalem,
                "cells": cell[None, :, :] * scalem,
            }
        )
        em, _ = model._total_energy(model.variables, sysm)
        Pvir_fd = -(ep[0] * au.KBAR - em[0] * au.KBAR) / (2.0 * vstep / au.BOHR**3)
        print(
            f"Initial pressure: {Pkin+Pvir:.3f} (virial); {Pkin+Pvir_fd:.3f} (finite difference) ; Pkin: {Pkin:.3f} ; Pvir: {Pvir:.3f} ; Pvir_fd: {Pvir_fd:.3f}"
        )

    ### DEFINE STEP FUNCTION ###
    @jax.jit
    def stepA(system, state):
        v = state["vel"]
        f = state["forces"]
        x = system["coordinates"]

        v = v + f * dt2m
        x = x + dt2 * v
        v, state_th = thermostat(v, state["thermostat"])
        x = x + dt2 * v
        # system = nblist_updater({**system, "coordinates": x})

        return {**system, "coordinates": x}, {**state, "vel": v, "thermostat": state_th}

    @jax.jit
    def stepB(system, state):
        v = state["vel"]
        state_th = state["thermostat"]

        if estimate_pressure:
            epot, f, vir_t, _ = model._energy_and_forces_and_virial(
                model.variables, system
            )
        else:
            epot, f, _ = model._energy_and_forces(model.variables, system)
        v = v + f * dt2m

        ek = 0.5 * jnp.sum(mass[:, None] * v**2) / state_th.get("corr_kin", 1.0)
        state = {
            **state,
            "vel": v,
            "forces": f,
            "ek": ek,
            "epot": epot[0],
            # "thermostat": state_th,
        }

        if do_window_avg:
            n = state["window_avg"]["n"] + 1
            state_avg = {**state["window_avg"], "n": n}
            state_avg["ek_m"] = state_avg["ek_m"] * win_b1 + ek * (1 - win_b1)
            state_avg["ek"] = state_avg["ek_m"] / (1 - win_b1**n)
            state_avg["epot_m"] = state_avg["epot_m"] * win_b1 + epot[0] * (1 - win_b1)
            state_avg["epot"] = state_avg["epot_m"] / (1 - win_b1**n)

        if estimate_pressure:
            vir = jnp.trace(vir_t[0])
            Pkin = (2 * pscale) * ek
            Pvir = (-pscale) * vir
            state["virial"] = vir
            state["pressure"] = Pkin + Pvir
            if do_window_avg:
                state_avg["Pkin_m"] = state_avg["Pkin_m"] * win_b1 + Pkin * (1 - win_b1)
                state_avg["Pkin"] = state_avg["Pkin_m"] / (1 - win_b1**n)
                state_avg["Pvir_m"] = state_avg["Pvir_m"] * win_b1 + Pvir * (1 - win_b1)
                state_avg["Pvir"] = state_avg["Pvir_m"] / (1 - win_b1**n)
                state_avg["pressure_m"] = state_avg["pressure_m"] * win_b1 + state[
                    "pressure"
                ] * (1 - win_b1)
                state_avg["pressure"] = state_avg["pressure_m"] / (1 - win_b1**n)

        if do_window_avg:
            state["window_avg"] = state_avg

        return system, state

    @jax.jit
    def check_nan(system, state):
        return jnp.any(jnp.isnan(state["vel"])) | jnp.any(jnp.isnan(system["coordinates"]))

    ### Print header
    nprint = int(simulation_parameters.get("nprint", 10))
    print(f"Running {nsteps} steps of MD simulation on {device}")
    header = "#     step   time(ps)   etot(Ha)   epot(Ha)     ek(Ha)    temp(K)     ns/day     step/s"
    if estimate_pressure:
        header += "        press"
    print(header)
    fout = open(traj_file, "a+")
    t0 = time.time()
    t0dump = t0
    tstart_dyn = t0
    istep = 0
    tpre = 0
    tup = 0
    tstep = 0
    t0full = time.time()
    print_timings = simulation_parameters.get("print_timings", False)
    force_preprocess = False
    nb_warmup_start = 0
    nblist_countdown = 0
    print_skin_activation = True
    for istep in range(1, nsteps + 1):
        ### BAOAB evolution
        # if istep % nblist_stride == 0 or force_preprocess:
        #     force_preprocess = False

        tstep0 = time.time()
        ## take a half step (update positions, nblist and half velocities)
        system, state = stepA(system, state)

        if print_timings:
            system["coordinates"].block_until_ready()
            tstep += time.time() - tstep0
            tstep0 = time.time()

        if nblist_countdown <= 0 or force_preprocess or (istep<nblist_warmup):
            nblist_countdown = nblist_stride - 1
            system = model.preprocessing.process(preproc_state, system)
            preproc_state, state_up, system, overflow = (
                model.preprocessing.check_reallocate(preproc_state, system)
            )
            if nblist_verbose and overflow:
                print("step", istep, ", nblist overflow => reallocating nblist")
                print("size updates:", state_up)

            if print_timings:
                system["coordinates"].block_until_ready()
                tpre += time.time() - tstep0
                tstep0 = time.time()

        else:
            if print_skin_activation:
                if nblist_verbose:
                    print("step", istep, ", end of nblist warmup phase => activating skin updates")
                print_skin_activation = False
            nblist_countdown -= 1
            # system = nblist_skin_updater(system)
            system = model.preprocessing.update_skin(system)

            if print_timings:
                system["coordinates"].block_until_ready()
                tup += time.time() - tstep0
                tstep0 = time.time()

        ## finish step
        system, state = stepB(system, state)

        ## end of state update (mostly for adQTB)
        if do_thermostat_post:
            state["thermostat"],post_state = thermostat_post(state["thermostat"],post_state)

        if print_timings:
            system["coordinates"].block_until_ready()
            tstep += time.time() - tstep0

        if istep % ndump == 0:
            if check_nan(system, state):
                raise ValueError(f"dynamics crashed at step {istep}.")
            
            tperstep = (time.time() - t0dump) / ndump
            nsperday = (24 * 60 * 60 / tperstep) * dt / 1e6
            if do_wrap_box:
                system["coordinates"] = wrapbox(
                    system["coordinates"], cell, reciprocal_cell
                )
                print("Wrap atoms into box")
                force_preprocess = True
            print("Write XYZ frame")
            write_arc_frame(fout, symbols, np.asarray(system["coordinates"]))
            print("ns/day: ", nsperday)
            if  nblist_verbose:
                print()
                print("nblist state:",preproc_state)
                print()
            print(header)
            t0dump = time.time()

            # model.reinitialize_preprocessing()

        if istep % nprint == 0:
            t1 = time.time()
            tperstep = (t1 - t0) / nprint
            t0 = t1
            nsperday = (24 * 60 * 60 / tperstep) * dt / 1e6
            if do_window_avg:
                ek = state["window_avg"]["ek"]
                e = state["window_avg"]["epot"]
            else:
                ek = state["ek"]
                e = state["epot"]

            line = f"{istep:10} {(start_time+istep*dt)/1000:10.3f} {(ek+e):10.5f} {e:10.3f} {ek:10.3f} {2*ek/(3.*nat)*au.KELVIN:10.3f} {nsperday:10.3f} {1./tperstep:10.3f}"
            if estimate_pressure:
                if do_window_avg:
                    pres = state["window_avg"]["pressure"]
                else:
                    pres = state["pressure"]
                line += f' {pres:10.3f}'

            print(line)

            if print_timings:
                tfull = time.time() - t0full
                tother = tfull - tpre - tstep - tup
                print(
                    f"tpre: {tpre:.5f} ({tpre/tfull*100:.2f} %); tup: {tup:.5f} ({tup/tfull*100:.2f} %); tstep: {tstep:.5f} ({tstep/tfull*100:.2f} %); tother: {tother:.5f} ({tother/tfull*100:.2f} %)"
                )
                tpre = 0
                tstep = 0
                tup = 0
                t0full = time.time()

    print(f"Run done in {(time.time()-tstart_dyn)/60.0} minutes")
    ### close trajectory file
    fout.close()


if __name__ == "__main__":
    main()
