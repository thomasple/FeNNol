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


from ..models import FENNIX
from ..utils.io import (
    write_arc_frame,
    last_xyz_frame,
    write_xyz_frame,
    write_extxyz_frame,
    human_time_duration,
)
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .thermostats import get_thermostat

from copy import deepcopy


def minmaxone(x, name=""):
    print(name, x.min(), x.max(), (x**2).mean() ** 0.5)


@jax.jit
def wrapbox(x, cell, reciprocal_cell):
    # q = jnp.einsum("ji,sj->si", reciprocal_cell, x)
    q = x @ reciprocal_cell
    q = q - jnp.floor(q)
    # return jnp.einsum("ji,sj->si", cell, q)
    return q @ cell


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
    elif device.startswith("cuda") or device.startswith("gpu"):
        if ":" in device:
            num = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = num
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = "gpu"

    _device = jax.devices(device)[0]
    jax.config.update("jax_default_device", _device)

    ### Set the precision
    enable_x64 = simulation_parameters.get("enable_x64", False)
    jax.config.update("jax_enable_x64", enable_x64)
    fprec = "float64" if enable_x64 else "float32"

    matmul_precision = simulation_parameters.get("matmul_prec", "highest").lower()
    assert matmul_precision in [
        "default",
        "high",
        "highest",
    ], "matmul_prec must be one of 'default','high','highest'"
    if matmul_precision != "highest":
        print(f"# Setting matmul precision to '{matmul_precision}'")
    if matmul_precision == "default" and fprec == "float32":
        print(
            "# Warning: default matmul precision involves float16 operations which may lead to large numerical errors on energy and pressure estimations ! It is recommended to set matmul_prec to 'high' or 'highest'."
        )
    jax.config.update("jax_default_matmul_precision", matmul_precision)

    # with jax.default_device(_device):
    dynamic(simulation_parameters, device, fprec)


def dynamic(simulation_parameters, device, fprec):
    tstart_dyn = time.time()
    ### Initialize the model
    # assert 'model_file' in simulation_parameters, "model_file not specified in parameter file"
    model_file = simulation_parameters.get("model_file")
    model_file = Path(str(model_file).strip())
    if not model_file.exists():
        raise FileNotFoundError(f"model file {model_file} not found")
    else:
        graph_config = simulation_parameters.get("graph_config", {})
        model = FENNIX.load(model_file, graph_config=graph_config)  # \
        print(f"# model_file: {model_file}")

    if "energy_terms" in simulation_parameters:
        model.set_energy_terms(simulation_parameters["energy_terms"])
        print("# energy terms:", model.energy_terms)

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
        print("# Replacing all hydrogens with deuteriums")
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
        cell = np.array(cell, dtype=fprec).reshape(3, 3)
        reciprocal_cell = np.linalg.inv(cell)
        volume = np.linalg.det(cell)
        print("# cell matrix:")
        for l in cell:
            print("# ", l)
        # print(cell)
        dens = totmass_amu / 6.02214129e-1 / volume
        print("# density: ", dens.item(), " g/cm^3")
        pscale = au.KBAR / (3.0 * volume / au.BOHR**3)

    if crystal_input:
        assert cell is not None, "cell must be specified for crystal units"
        coordinates = coordinates @ cell  
        with open("initial.xyz", "w") as finit:
            write_xyz_frame(finit, symbols, coordinates)

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
    temperature = np.clip(simulation_parameters.get("temperature",300.),1.e-6,None)
    kT = temperature / au.KELVIN
    start_time = 0.0
    start_step = 0

    ### Set I/O parameters
    Tdump = simulation_parameters.get("tdump", 1.0 / au.PS) * au.FS
    ndump = int(Tdump / dt)
    do_wrap_box = simulation_parameters.get("wrap_box", True) and cell is not None
    # traj_file = Path(simulation_parameters.get("traj_file", system_name + ".arc"))
    traj_format = simulation_parameters.get("traj_format", "extxyz").lower()
    if traj_format == "xyz":
        traj_file = system_name + ".traj.xyz"
        write_frame = write_xyz_frame
    elif traj_format == "extxyz":
        traj_file = system_name + ".traj.extxyz"
        write_frame = write_extxyz_frame
    elif traj_format == "arc":
        traj_file = system_name + ".arc"
        write_frame = write_arc_frame
    else:
        raise ValueError(
            f"Unknown trajectory format '{traj_format}'. Supported formats are 'arc' and 'xyz'"
        )

    random_seed = simulation_parameters.get(
        "random_seed", np.random.randint(0, 2**32 - 1)
    )
    print(f"# random_seed: {random_seed}")
    rng_key = jax.random.PRNGKey(random_seed)

    system = {}

    ### Set the thermostat
    rng_key, t_key = jax.random.split(rng_key)
    thermostat_name = str(simulation_parameters.get("thermostat", "NONE")).upper()
    thermostat, thermostat_post, system["thermostat"], vel = get_thermostat(
        thermostat_name,
        fprec=fprec,
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
    
    include_thermostat_energy = "thermostat_energy" in system["thermostat"]

    ### CONFIGURE PREPROCESSING
    minimum_image = simulation_parameters.get("minimum_image", True)
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
    graphs_keys = list(model._graphs_properties.keys())

    ## initial preprocessing
    preproc_state = preproc_state.copy({"check_input": True})
    conformation = dict(species=species, coordinates=coordinates)
    if cell is not None:
        conformation["cells"] = cell[None, :, :]
    preproc_state, conformation = model.preprocessing(preproc_state, conformation)

    preproc_state = preproc_state.copy({"check_input": False})
    # preproc_state["check_input"] = False
    if nblist_verbose:
        print("# graphs_keys: ", graphs_keys)
        print("# nblist state:", preproc_state)

    ### print model
    if simulation_parameters.get("print_model", False):
        print(model.summarize(example_data=conformation))
    ## initial energy and forces
    print("# Computing initial energy and forces")
    e, f, output = model._energy_and_forces(model.variables, conformation)
    eat = np.array(output["atomic_energies"][:, None])
    f = np.array(f)
    epot = e[0]
    # np.savetxt("initial_energy_forces.txt",np.column_stack((eat,f)),fmt="%.5f")

    # rng_key, v_key = jax.random.split(rng_key)
    # vel = (
    #     jax.random.normal(v_key, coordinates.shape) * (kT / mass[:, None]) ** 0.5
    # ).astype(fprec)
    ek = 0.5 * jnp.sum(mass[:, None] * vel**2)
    system["coordinates"] = conformation["coordinates"]
    system["vel"] = vel.astype(fprec)
    system["ek"] = ek
    system["forces"] = f
    system["epot"] = e

    ### Print initial pressure
    if estimate_pressure and fprec == "float64":
        Pkin = (2 * au.KBAR) * ek / ((3.0 / au.BOHR**3) * volume)
        e, f, vir_t, _ = model._energy_and_forces_and_virial(
            model.variables, conformation
        )
        Pvir = -(np.trace(vir_t[0]) * au.KBAR) / ((3.0 / au.BOHR**3) * volume)
        vstep = volume * 0.000001
        scalep = ((volume + vstep) / volume) ** (1.0 / 3.0)
        sysp = model.preprocess(
            **{
                **conformation,
                "coordinates": coordinates * scalep,
                "cells": cell[None, :, :] * scalep,
            }
        )
        ep, _ = model._total_energy(model.variables, sysp)
        scalem = ((volume - vstep) / volume) ** (1.0 / 3.0)
        sysm = model.preprocess(
            **{
                **conformation,
                "coordinates": coordinates * scalem,
                "cells": cell[None, :, :] * scalem,
            }
        )
        em, _ = model._total_energy(model.variables, sysm)
        Pvir_fd = -(ep[0] * au.KBAR - em[0] * au.KBAR) / (2.0 * vstep / au.BOHR**3)
        print(
            f"# Initial pressure: {Pkin+Pvir:.3f} (virial); {Pkin+Pvir_fd:.3f} (finite difference) ; Pkin: {Pkin:.3f} ; Pvir: {Pvir:.3f} ; Pvir_fd: {Pvir_fd:.3f}"
        )

    ### DEFINE STEP FUNCTION ###
    @jax.jit
    def stepA(system):
        v = system["vel"]
        f = system["forces"]
        x = system["coordinates"]

        v = v + f * dt2m
        x = x + dt2 * v
        v, state_th = thermostat(v, system["thermostat"])
        x = x + dt2 * v
        # system = nblist_updater({**system, "coordinates": x})

        return {**system, "coordinates": x, "vel": v, "thermostat": state_th}

    @jax.jit
    def update_forces(system, conformation):
        if estimate_pressure:
            epot, f, vir_t, _ = model._energy_and_forces_and_virial(
                model.variables, conformation
            )
            return {
                **system,
                "forces": f,
                "epot": epot[0],
                "virial": jnp.trace(vir_t[0]),
            }
        else:
            epot, f, _ = model._energy_and_forces(model.variables, conformation)
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

    @jax.jit
    def check_nan(system):
        return jnp.any(jnp.isnan(system["vel"])) | jnp.any(
            jnp.isnan(system["coordinates"])
        )

    ### Energy units and print initial energy
    per_atom_energy = simulation_parameters.get("per_atom_energy", True)
    energy_unit_str = simulation_parameters.get("energy_unit", "kcal/mol")
    print("# Energy unit: ", energy_unit_str)
    energy_unit = au.get_multiplier(energy_unit_str)
    atom_energy_unit = energy_unit
    atom_energy_unit_str = energy_unit_str
    if per_atom_energy:
        atom_energy_unit /= nat
        atom_energy_unit_str = f"{energy_unit_str}/atom"
        print("# Printing Energy per atom")
    print(
        f"# Initial potential energy: {epot*atom_energy_unit}; kinetic energy: {ek*atom_energy_unit}"
    )
    minmaxone(jnp.abs(f * energy_unit), "# forces min/max/rms:")

    ## printing options
    print_timings = simulation_parameters.get("print_timings", False)
    nprint = int(simulation_parameters.get("nprint", 10))
    assert nprint > 0, "nprint must be > 0"
    nsummary = simulation_parameters.get("nsummary", 100*nprint)
    assert nsummary > 0, "nsummary must be > 0"

    ### Print header
    print("#" * 84)
    print(
        f"# Running {nsteps:_} steps of {thermostat_name} MD simulation on {device}"
    )
    header = (
        "#     Step   Time[ps]        Etot        Epot        Ekin    Temp[K]"
    )
    if include_thermostat_energy:
        header += "      Etherm"
    if estimate_pressure:
        header += "  Press[kbar]"
    print(header)

    ### Open trajectory file
    fout = open(traj_file, "a+")

    ### initialize proprerty trajectories
    properties_traj = defaultdict(list)
    if print_timings:
        timings = defaultdict(lambda: 0.0)

    ### initialize counters and timers
    t0 = time.time()
    t0dump = t0
    istep = 0
    t0full = time.time()
    force_preprocess = False
    nb_warmup_start = 0
    nblist_countdown = 0
    print_skin_activation = nblist_warmup > 0

    for istep in range(1, nsteps + 1):
        ### BAOAB evolution
        # if istep % nblist_stride == 0 or force_preprocess:
        #     force_preprocess = False

        tstep0 = time.time()
        ## take a half step (update positions, nblist and half velocities)
        system = stepA(system)

        if print_timings:
            system["coordinates"].block_until_ready()
            timings["Integrator"] += time.time() - tstep0
            tstep0 = time.time()

        ### update conformation and graphs
        if nblist_countdown <= 0 or force_preprocess or (istep < nblist_warmup):
            ### full nblist update
            nblist_countdown = nblist_stride - 1
            conformation = model.preprocessing.process(
                preproc_state, {**conformation, "coordinates": system["coordinates"]}
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
            if print_skin_activation:
                if nblist_verbose:
                    print(
                        "step",
                        istep,
                        ", end of nblist warmup phase => activating skin updates",
                    )
                print_skin_activation = False
            nblist_countdown -= 1
            conformation = model.preprocessing.update_skin(
                {**conformation, "coordinates": system["coordinates"]}
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
            system["thermostat"], post_state = thermostat_post(
                system["thermostat"], post_state
            )

        if print_timings:
            system["coordinates"].block_until_ready()
            timings["Integrator"] += time.time() - tstep0
            tstep0 = time.time()

        ### print properties
        if istep % nprint == 0:
            t1 = time.time()
            tperstep = (t1 - t0) / nprint
            t0 = t1
            nsperday = (24 * 60 * 60 / tperstep) * dt / 1e6

            ek = system["ek"]
            epot = system["epot"]
            etot = ek + epot
            temper = 2 * ek / (3.0 * nat) * au.KELVIN

            th_state = system["thermostat"]
            if include_thermostat_energy:
                etherm = th_state["thermostat_energy"]
                etot = etot + etherm

            properties_traj[f"Etot[{atom_energy_unit_str}]"].append(etot * atom_energy_unit)
            properties_traj[f"Epot[{atom_energy_unit_str}]"].append(epot * atom_energy_unit)
            properties_traj[f"Ekin[{atom_energy_unit_str}]"].append(ek * atom_energy_unit)
            properties_traj["Temper[Kelvin]"].append(temper)

            ### construct line of properties
            line = f"{istep:10.6g} {(start_time+istep*dt)/1000: 10.3f}  {etot*atom_energy_unit: #10.4f}  {epot*atom_energy_unit: #10.4f}  {ek*atom_energy_unit: #10.4f} {temper: 10.2f}"
            if include_thermostat_energy:
                line += f"  {etherm*atom_energy_unit: #10.4f}"
                properties_traj[f"Etherm[{atom_energy_unit_str}]"].append(etherm * atom_energy_unit)
            if estimate_pressure:
                pres = system["pressure"]
                properties_traj["Pressure[kbar]"].append(pres * au.KBAR * au.BOHR**3)
                line += f" {pres*au.KBAR*au.BOHR**3:10.3f}"

            print(line)

        ### save frame
        if istep % ndump == 0:
            line = "# Write XYZ frame"
            if do_wrap_box:
                system["coordinates"] = wrapbox(
                    system["coordinates"], cell, reciprocal_cell
                )
                line += " (atoms have been wrapped into the box)"
                force_preprocess = True
            print(line)
            properties = {
                "energy": float(system["epot"]) * energy_unit,
                "Time": start_time + istep * dt,
                "energy_unit": energy_unit_str,
            }
            write_frame(
                fout,
                symbols,
                np.asarray(system["coordinates"]),
                cell=cell,
                properties=properties,
                forces=np.asarray(system["forces"]) * energy_unit,
            )

        ### summary over last nsummary steps
        if istep % (nsummary) == 0:
            if check_nan(system):
                raise ValueError(f"dynamics crashed at step {istep}.")
            tfull = time.time() - t0full
            t0full = time.time()
            tperstep = tfull / (nsummary)
            nsperday = (24 * 60 * 60 / tperstep) * dt / 1e6
            elapsed_time = time.time() - tstart_dyn
            estimated_remaining_time = tperstep*(nsteps-istep)
            estimated_total_time = elapsed_time + estimated_remaining_time

            print("#" * 50)
            print(f"# Step {istep:_} of {nsteps:_}  ({istep/nsteps*100:.5g} %)")
            print(f"# Tot. elapsed time   : {human_time_duration(elapsed_time)}")
            print(
                  f"# Est. total time     : {human_time_duration(estimated_total_time)}"
            )
            print(
                  f"# Est. remaining time : {human_time_duration(estimated_remaining_time)}"
            )
            print(f"# Time for {nsummary:_} steps : {human_time_duration(tfull)}")
            
            if print_timings:
                print(f"# Detailed per-step timings :")
                dsteps = nsummary
                tother = tfull - sum([t for t in timings.values()])
                timings["Other"] = tother
                # sort timings
                timings = {k: v for k, v in sorted(timings.items(), key=lambda item: item[1], reverse=True)}
                for k, v in timings.items():
                    print(
                        f"#   {k:15} : {human_time_duration(v/dsteps):>12} ({v/tfull*100:5.3g} %)"
                    )
                print(f"#   {'Total':15} : {human_time_duration(tfull/dsteps):>12}")
                ## reset timings
                timings = defaultdict(lambda: 0.0)

            
            print(f"# Averages over last {nsummary:_} steps :")
            for k, v in properties_traj.items():
                if len(properties_traj[k]) == 0:
                    continue
                mu = np.mean(properties_traj[k])
                sig = np.std(properties_traj[k])
                ksplit = k.split("[")
                name = ksplit[0].strip()
                unit = ksplit[1].replace("]", "").strip() if len(ksplit) > 1 else ""
                print(f"#   {name:10} : {mu: #10.5g}   +/- {sig: #9.3g}  {unit}")

            if nblist_verbose:
                print("# nblist state :", preproc_state)
            print(
                f"# Perf.: {nsperday:.2f} ns/day  ( {1.0 / tperstep:.2f} step/s )"
            )
            print("#" * 50)
            if istep < nsteps:
                print(header)
            ## reset property trajectories
            properties_traj = defaultdict(list)

    print(f"# Run done in {human_time_duration(time.time()-tstart_dyn)}")
    ### close trajectory file
    fout.close()


if __name__ == "__main__":
    main()
