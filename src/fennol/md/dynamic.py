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
from .initial import load_model, load_system_data, initialize_preprocessing
from .integrate import initialize_dynamics

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
    model = load_model(simulation_parameters)

    ### Get the coordinates and species from the xyz file
    system_data, conformation = load_system_data(simulation_parameters, fprec)
    nat = system_data["nat"]

    preproc_state, conformation = initialize_preprocessing(
        simulation_parameters, model, conformation, system_data
    )

    random_seed = simulation_parameters.get(
        "random_seed", np.random.randint(0, 2**32 - 1)
    )
    print(f"# random_seed: {random_seed}")
    rng_key = jax.random.PRNGKey(random_seed)
    rng_key, subkey = jax.random.split(rng_key)
    ## INITIALIZE INTEGRATOR AND SYSTEM
    step, update_conformation, dyn_state, system = initialize_dynamics(
        simulation_parameters, system_data, conformation, model, fprec, subkey
    )

    dt = dyn_state["dt"]
    ## get number of steps
    nsteps = int(simulation_parameters.get("nsteps"))
    start_time = 0.0
    start_step = 0


    ### Set I/O parameters
    Tdump = simulation_parameters.get("tdump", 1.0 / au.PS) * au.FS
    ndump = int(Tdump / dt)
    system_name = system_data["name"]
    
    # traj_file = Path(simulation_parameters.get("traj_file", system_name + ".arc"))
    traj_format = simulation_parameters.get("traj_format", "arc").lower()
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


    ### Print initial pressure
    estimate_pressure = dyn_state["estimate_pressure"]
    if estimate_pressure and fprec == "float64":
        volume = system_data["pbc"]["volume"]
        coordinates = conformation["coordinates"]
        cell = conformation["cells"][0]
        Pkin = (2 * au.KBAR) * ek / ((3.0 / au.BOHR**3) * volume)
        e, f, vir_t, _ = model._energy_and_forces_and_virial(
            model.variables, conformation
        )
        Pvir = -(np.trace(vir_t[0]) * au.KBAR) / ((3.0 / au.BOHR**3) * volume)
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
        Pvir_fd = -(ep[0] * au.KBAR - em[0] * au.KBAR) / (2.0 * vstep / au.BOHR**3)
        print(
            f"# Initial pressure: {Pkin+Pvir:.3f} (virial); {Pkin+Pvir_fd:.3f} (finite difference) ; Pkin: {Pkin:.3f} ; Pvir: {Pvir:.3f} ; Pvir_fd: {Pvir_fd:.3f}"
        )

    @jax.jit
    def check_nan(system):
        return jnp.any(jnp.isnan(system["vel"])) | jnp.any(
            jnp.isnan(system["coordinates"])
        )

    if system_data["pbc"] is not None:
        cell = system_data["pbc"]["cell"]
        reciprocal_cell = system_data["pbc"]["reciprocal_cell"]
        do_wrap_box = simulation_parameters.get("wrap_box", False)
    else:
        cell = None
        reciprocal_cell = None

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
        f"# Initial potential energy: {system['epot']*atom_energy_unit}; kinetic energy: {system['ek']*atom_energy_unit}"
    )
    f = system["forces"]
    minmaxone(jnp.abs(f * energy_unit), "# forces min/max/rms:")

    ## printing options
    print_timings = simulation_parameters.get("print_timings", False)
    nprint = int(simulation_parameters.get("nprint", 10))
    assert nprint > 0, "nprint must be > 0"
    nsummary = simulation_parameters.get("nsummary", 100 * nprint)
    assert nsummary > 0, "nsummary must be > 0"

    ### Print header
    include_thermostat_energy = "thermostat_energy" in system["thermostat"]
    thermostat_name = dyn_state["thermostat_name"]
    pimd = dyn_state["pimd"]
    nbeads = system_data.get("nbeads", 1)
    dyn_name = "PIMD" if pimd else "MD"
    print("#" * 84)
    print(
        f"# Running {nsteps:_} steps of {thermostat_name} {dyn_name} simulation on {device}"
    )
    header = "#     Step   Time[ps]        Etot        Epot        Ekin    Temp[K]"
    if pimd:
        header += "  Temp_c[K]"
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

    for istep in range(1, nsteps + 1):

        ### update the system
        dyn_state, system, conformation, preproc_state = step(
            istep, dyn_state, system, conformation, preproc_state, force_preprocess
        )

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

            properties_traj[f"Etot[{atom_energy_unit_str}]"].append(
                etot * atom_energy_unit
            )
            properties_traj[f"Epot[{atom_energy_unit_str}]"].append(
                epot * atom_energy_unit
            )
            properties_traj[f"Ekin[{atom_energy_unit_str}]"].append(
                ek * atom_energy_unit
            )
            properties_traj["Temper[Kelvin]"].append(temper)
            if pimd:
                ek_c = system["ek_c"]
                temper_c = 2 * ek_c / (3.0 * nat) * au.KELVIN
                properties_traj["Temper_c[Kelvin]"].append(temper_c)

            ### construct line of properties
            line = f"{istep:10.6g} {(start_time+istep*dt)/1000: 10.3f}  {etot*atom_energy_unit: #10.4f}  {epot*atom_energy_unit: #10.4f}  {ek*atom_energy_unit: #10.4f} {temper: 10.2f}"
            if pimd:
                line += f" {temper_c: 10.2f}"
            if include_thermostat_energy:
                line += f"  {etherm*atom_energy_unit: #10.4f}"
                properties_traj[f"Etherm[{atom_energy_unit_str}]"].append(
                    etherm * atom_energy_unit
                )
            if estimate_pressure:
                pres = system["pressure"]
                properties_traj["Pressure[kbar]"].append(pres * au.KBAR * au.BOHR**3)
                line += f" {pres*au.KBAR*au.BOHR**3:10.3f}"

            print(line)

        ### save frame
        if istep % ndump == 0:
            line = "# Write XYZ frame"
            if do_wrap_box:
                if pimd:
                    centroid = wrapbox(system["coordinates"][0], cell, reciprocal_cell)
                    system["coordinates"] = system["coordinates"].at[0].set(centroid)
                else:
                    system["coordinates"] = wrapbox(
                        system["coordinates"], cell, reciprocal_cell
                    )
                conformation["coordinates"] = update_conformation(system["coordinates"])
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
                system_data["symbols"],
                np.asarray(conformation["coordinates"].reshape(nbeads, nat, 3)[0]),
                cell=cell,
                properties=properties,
                forces=None,  # np.asarray(system["forces"].reshape(nbeads, nat, 3)[0]) * energy_unit,
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
            estimated_remaining_time = tperstep * (nsteps - istep)
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
                timings = {
                    k: v
                    for k, v in sorted(
                        timings.items(), key=lambda item: item[1], reverse=True
                    )
                }
                for k, v in timings.items():
                    print(
                        f"#   {k:15} : {human_time_duration(v/dsteps):>12} ({v/tfull*100:5.3g} %)"
                    )
                print(f"#   {'Total':15} : {human_time_duration(tfull/dsteps):>12}")
                ## reset timings
                timings = defaultdict(lambda: 0.0)
            
            corr_kin = system["thermostat"].get("corr_kin",None)
            if corr_kin is not None:
                print(f"# QTB kin. correction : {100*(corr_kin-1.):.2f} %")
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

            print(f"# Perf.: {nsperday:.2f} ns/day  ( {1.0 / tperstep:.2f} step/s )")
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
