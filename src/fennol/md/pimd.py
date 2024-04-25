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
    dt2m = jnp.asarray(dt2 / mass[None, :, None], dtype=fprec)

    nsteps = int(simulation_parameters.get("nsteps"))
    gamma = simulation_parameters.get("gamma", 1.0 / au.THZ) / au.FS
    temperature = np.clip(simulation_parameters.get("temperature", 300.0), 1.0e-6, None)
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

    ### number of beads
    nbeads = simulation_parameters.get("nbeads")
    eigmat = np.zeros((nbeads, nbeads))
    for i in range(nbeads - 1):
        eigmat[i, i] = 2.0
        eigmat[i, i + 1] = -1.0
        eigmat[i + 1, i] = -1.0
    eigmat[-1, -1] = 2.0
    eigmat[0, -1] = -1.0
    eigmat[-1, 0] = -1.0
    omk, eigmat = np.linalg.eigh(eigmat)
    omk[0] = 0.0
    omk = nbeads * kT * omk**0.5 / au.FS
    for i in range(nbeads):
        if eigmat[i, 0] < 0:
            eigmat[i] *= -1.0
    eigmat = jnp.asarray(eigmat, dtype=fprec)

    coordinates = np.repeat(coordinates[None, :, :], nbeads, axis=0)
    eigx = np.zeros_like(coordinates)
    eigx[0] = coordinates[0]
    coordinates = coordinates.reshape(nbeads * nat, 3)
    bead_index = np.arange(nbeads, dtype=np.int32).repeat(nat)
    natoms = np.array([nat] * nbeads, dtype=np.int32)
    species_ = np.tile(species, nbeads)

    ### Set the thermostat
    rng_key, t_key = jax.random.split(rng_key)
    thermostat_name = str(simulation_parameters.get("thermostat", "NONE")).upper()
    trpmd_lambda = simulation_parameters.get("trpmd_lambda", 1.0)
    gammak = np.maximum(trpmd_lambda * omk, gamma)

    thermostat, thermostat_post, system["thermostat"], vel = get_thermostat(
        thermostat_name,
        fprec=fprec,
        rng_key=t_key,
        dt=dt,
        mass=mass,
        gamma=gammak,
        kT=kT,
        simulation_parameters=simulation_parameters,
        species=species,
        nbeads=nbeads,
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
    conformation = dict(
        species=species_, coordinates=coordinates, natoms=natoms, batch_index=bead_index
    )
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
    epot = np.mean(e)

    ek = 0.5 * jnp.sum(mass[:, None] * vel**2)
    system["eigx"] = jnp.asarray(eigx, dtype=fprec)
    system["eigv"] = vel.astype(fprec)
    system["ek"] = ek
    system["eigf"] = jnp.einsum("in,i...->n...", eigmat, f.reshape(nbeads, nat, 3)) * (
        1.0 / nbeads**0.5
    )
    system["epot"] = epot

    ### DEFINE STEP FUNCTION ###
    cay_correction = simulation_parameters.get("cay_correction", True)

    def apply_A(system):
        if cay_correction:
            cayfact = 1.0 / (4.0 + (dt * omk[1:, None, None]) ** 2) ** 0.5
            axx = jnp.asarray(2 * cayfact)
            axv = jnp.asarray(dt * cayfact)
            avx = jnp.asarray(-dt * cayfact * omk[1:, None, None] ** 2)
        else:
            axx = jnp.asarray(np.cos(omk[1:, None, None] * dt2))
            axv = jnp.asarray(np.sin(omk[1:, None, None] * dt2) / omk[1:, None, None])
            avx = jnp.asarray(-omk[1:, None, None] * np.sin(omk[1:, None, None] * dt2))
        eigx0 = system["eigx"]
        eigv0 = system["eigv"]
        eigx_c = eigx0[0] + dt2 * eigv0[0]
        eigv_c = eigv0[0]
        eigx = eigx0[1:] * axx + eigv0[1:] * axv
        eigv = eigx0[1:] * avx + eigv0[1:] * axx

        return {
            **system,
            "eigx": jnp.concatenate((eigx_c[None], eigx), axis=0),
            "eigv": jnp.concatenate((eigv_c[None], eigv), axis=0),
        }

    def apply_B(system):
        return {**system, "eigv": system["eigv"] + dt2m * system["eigf"]}

    @jax.jit
    def stepA(system):
        system = apply_B(system)
        system = apply_A(system)
        system["eigv"], system["thermostat"] = thermostat(
            system["eigv"], system["thermostat"]
        )
        system = apply_A(system)

        return system

    @jax.jit
    def eig_to_coords(eigx):
        return jnp.einsum("in,n...->i...", eigmat, eigx).reshape(nbeads * nat, 3) * (
            nbeads**0.5
        )

    @jax.jit
    def coords_to_eig(x):
        return jnp.einsum("in,i...->n...", eigmat, x.reshape(nbeads, nat, 3)) * (
            1.0 / nbeads**0.5
        )

    @jax.jit
    def update_forces(system, conformation):
        if estimate_pressure:
            epot, f, vir_t, _ = model._energy_and_forces_and_virial(
                model.variables, conformation
            )
            return {
                **system,
                "eigf": coords_to_eig(f),
                "epot": jnp.mean(epot),
                "virial": jnp.trace(jnp.mean(vir_t, axis=0)),
            }
        else:
            epot, f, _ = model._energy_and_forces(model.variables, conformation)
            return {**system, "eigf": coords_to_eig(f), "epot": jnp.mean(epot)}

    @jax.jit
    def stepB(system):
        system = apply_B(system)

        ek_c = 0.5 * jnp.sum(mass[:, None] * system["eigv"][0] ** 2)
        ek = ek_c - 0.5 * jnp.sum(system["eigx"][1:] * system["eigf"][1:])
        system["ek"] = ek
        system["ek_c"] = ek_c

        if estimate_pressure:
            vir = system["virial"]
            Pkin = (2 * pscale) * ek
            Pvir = (-pscale) * vir
            system["pressure"] = Pkin + Pvir

        return system

    @jax.jit
    def check_nan(system):
        return jnp.any(jnp.isnan(system["eigv"])) | jnp.any(
            jnp.isnan(system["eigx"])
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
    nsummary = simulation_parameters.get("nsummary", 100 * nprint)
    assert nsummary > 0, "nsummary must be > 0"

    ### Print header
    print("#" * 84)
    print(f"# Running {nsteps:_} steps of {thermostat_name} PIMD simulation on {device}")
    header = "#     Step   Time[ps]        Etot        Epot        Ekin    Temp[K]  Temp_c[K]"
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
            system["eigv"].block_until_ready()
            timings["Integrator"] += time.time() - tstep0
            tstep0 = time.time()

        ### update conformation and graphs
        if nblist_countdown <= 0 or force_preprocess or (istep < nblist_warmup):
            ### full nblist update
            nblist_countdown = nblist_stride - 1
            conformation = model.preprocessing.process(
                preproc_state,
                {**conformation, "coordinates": eig_to_coords(system["eigx"])},
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
                {**conformation, "coordinates": eig_to_coords(system["eigx"])}
            )

            if print_timings:
                conformation["coordinates"].block_until_ready()
                timings["Skin update"] += time.time() - tstep0
                tstep0 = time.time()

        ## compute forces
        system = update_forces(system, conformation)
        if print_timings:
            system["eigf"].block_until_ready()
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
            system["eigx"].block_until_ready()
            timings["Integrator"] += time.time() - tstep0
            tstep0 = time.time()

        ### print properties
        if istep % nprint == 0:
            t1 = time.time()
            tperstep = (t1 - t0) / nprint
            t0 = t1
            nsperday = (24 * 60 * 60 / tperstep) * dt / 1e6

            ek = system["ek"]
            ek_c = system["ek_c"]
            epot = system["epot"]
            etot = ek + epot
            temper = 2 * ek / (3.0 * nat) * au.KELVIN
            temper_c = 2 * ek_c / (3.0 * nat) * au.KELVIN

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
            properties_traj["Temper_c[Kelvin]"].append(temper_c)

            ### construct line of properties
            line = f"{istep:10.6g} {(start_time+istep*dt)/1000: 10.3f}  {etot*atom_energy_unit: #10.4f}  {epot*atom_energy_unit: #10.4f}  {ek*atom_energy_unit: #10.4f} {temper: 10.2f} {temper_c: 10.2f}"
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
                np.asarray(conformation["coordinates"].reshape(nbeads, nat, 3)[0]),
                cell=cell,
                properties=properties,
                forces=None,
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
