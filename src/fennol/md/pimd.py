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


from ..models import FENNIX
from ..utils.io import write_arc_frame, last_xyz_frame
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .thermostats import get_thermostat


def minmaxone(x, name=""):
    print(name, x.min(), x.max(), (x**2).mean() ** 0.5)


def main():
    os.environ["OMP_NUM_THREADS"] = "1"
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
    elif device.startswith("cuda"):
        num = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = num
        device = "gpu"

    _device = jax.devices(device)[0]

    ### Set the precision
    jax.config.update("jax_enable_x64", simulation_parameters.get("enable_x64", False))
    fprec = "float32"

    with jax.default_device(_device):
        pimd(simulation_parameters, device, fprec)


def pimd(simulation_parameters, device, fprec):
    ### Initialize the model
    # assert 'model_file' in simulation_parameters, "model_file not specified in parameter file"
    model_file = simulation_parameters.get("model_file")
    model_file = Path(str(model_file).strip())
    if not model_file.exists():
        raise FileNotFoundError(f"model file {model_file} not found")
    else:
        model = FENNIX.load(model_file, fixed_preprocessing=True)  # \
        print(f"model_file: {model_file}")
        # .train(False).requires_grad_(False).to(prec)

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
    mass = mass_amu * (au.MPROT / au.BOHR**2)
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
        volume = np.linalg.det(cell)
        print("cell matrix:")
        print(cell)
        dens = totmass_amu / 6.02214129e-1 / volume
        print("density: ", dens.item(), " g/cm^3")

    if crystal_input:
        assert cell is not None, "cell must be specified for crystal units"
        coordinates = coordinates @ cell  # .double()
        with open("initial.arc", "w") as finit:
            write_arc_frame(finit, symbols, coordinates)

    ### Set simulation parameters
    dt = simulation_parameters.get("dt")  # /au.FS
    dt2 = 0.5 * dt
    nsteps = int(simulation_parameters.get("nsteps"))
    gamma = simulation_parameters.get("gamma", 1.0 / au.THZ)
    kT = simulation_parameters.get("temperature", 300.0) / au.KELVIN
    start_time = 0.0
    start_step = 0

    ### Set I/O parameters
    Tdump = simulation_parameters.get("tdump", 1.0 / au.PS)
    ndump = int(Tdump / dt)
    wrap_box = simulation_parameters.get("wrap_box", True) and cell is not None
    traj_file = Path(simulation_parameters.get("traj_file", system_name + ".arc"))

    nblist_stride = int(simulation_parameters.get("nblist_stride", 1))
    nblist_skin = simulation_parameters.get("nblist_skin", 0.0)
    nprint = int(simulation_parameters.get("nprint", 10))
    print("nprint: ", nprint, "; nblist_stride: ", nblist_stride)

    random_seed = simulation_parameters.get(
        "random_seed", np.random.randint(0, 2**32 - 1)
    )
    print(f"random_seed: {random_seed}")
    rng_key = jax.random.PRNGKey(random_seed)

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
    omk = nbeads * kT * omk**0.5
    for i in range(nbeads):
        if eigmat[i, 0] < 0:
            eigmat[i] *= -1.0
    eigmat = jnp.asarray(eigmat, dtype=fprec)

    coordinates = np.repeat(coordinates[None, :, :], nbeads, axis=0)
    eigx = np.zeros_like(coordinates)
    eigx[0] = coordinates[0]
    coordinates = coordinates.reshape(nbeads * nat, 3)
    isys = np.arange(nbeads, dtype=np.int32).repeat(nat)
    natoms = np.array([nat] * nbeads, dtype=np.int32)
    species_ = np.tile(species, nbeads)

    ### Set the thermostat
    rng_key, t_key = jax.random.split(rng_key)
    thermostat_name = str(simulation_parameters.get("thermostat", "NONE")).upper()
    qtb_parameters = simulation_parameters.get("qtb", {})
    assert isinstance(qtb_parameters, dict), "qtb must be a dictionary"
    trpmd_lambda = simulation_parameters.get("trpmd_lambda", 1.0)
    gammak = np.maximum(trpmd_lambda * omk, gamma)

    thermostat, thermostat_post, state = get_thermostat(
        thermostat_name,
        rng_key=t_key,
        dt=dt,
        mass=mass,
        gamma=gammak,
        kT=kT,
        qtb_parameters=None,
        species=species,
        nbeads=nbeads,
    )

    dt2m = jnp.asarray(dt2 / mass[None, :, None], dtype=fprec)

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
    def step(system, state):
        system = apply_B(system)
        system = apply_A(system)
        system["eigv"], state = thermostat(system["eigv"], state)
        system = apply_A(system)

        system["coordinates"] = jnp.einsum(
            "in,n...->i...", eigmat, system["eigx"]
        ).reshape(nbeads * nat, 3) * (nbeads**0.5)

        _, f, system = model._energy_and_forces(model.variables, system)

        system["eigf"] = jnp.einsum(
            "in,i...->n...", eigmat, f.reshape(nbeads, nat, 3)
        ) * (1.0 / nbeads**0.5)

        system = apply_B(system)

        ek_c = 0.5 * jnp.sum(mass[:, None] * system["eigv"][0] ** 2)
        system["ek_c"] = ek_c
        ek = ek_c - 0.5 * jnp.sum(system["eigx"][1:] * system["eigf"][1:])
        system["ek"] = ek

        return system, state

    # @partial(jax.jit, static_argnums=1)
    # def integrate(system,nsteps):
    #     return jax.lax.fori_loop(0,nsteps,step,system)
    if cell is None:
        system = model.preprocess(
            species=species_, coordinates=coordinates, isys=isys, natoms=natoms
        )
    else:
        system = model.preprocess(
            species=species_,
            coordinates=coordinates,
            cells=cell[None, :, :],
            isys=isys,
            natoms=natoms,
        )

    if simulation_parameters.get("print_model_summary", False):
        print(model.summarize(example_data=system))
    # initial energy and forces
    print("Computing initial energy and forces")
    e, f, system = model._energy_and_forces(model.variables, system)
    system["eigf"] = jnp.einsum("in,i...->n...", eigmat, f.reshape(nbeads, nat, 3)) * (
        1.0 / nbeads**0.5
    )
    print(f"Initial energy: {np.mean(np.array(e))}")

    rng_key, v_key = jax.random.split(rng_key)
    eigv = (
        jax.random.normal(v_key, eigx.shape) * (kT / mass[None, :, None]) ** 0.5
    ).astype(fprec)
    ek = 0.5 * jnp.sum(mass[:, None] * eigv[0] ** 2)
    system["eigv"] = jnp.asarray(eigv, dtype=fprec)
    system["eigx"] = eigx
    system["ek"] = ek
    system["nblist_skin"] = nblist_skin
    if "nblist_mult_size" in simulation_parameters:
        system["nblist_mult_size"] = simulation_parameters["nblist_mult_size"]

    ### Print header
    print(f"Running {nsteps} steps of MD simulation on {device}")
    header = "#     step       time       etot       epot         ek       temp     temp_c     ns/day   step/s"
    print(header)
    fout = open(traj_file, "a+")
    t0 = time.time()
    t0dump = t0
    tstart_dyn = t0
    istep = 0
    tpre = 0
    tstep = 0
    print_timings = simulation_parameters.get("print_timings", False)
    for istep in range(1, nsteps + 1):
        ### BAOAB evolution
        if istep % nblist_stride == 0:
            tpre0 = time.time()
            system = model.preprocess(**system)
            if print_timings:
                system["coordinates"].block_until_ready()
                tpre += time.time() - tpre0

        tstep0 = time.time()
        system, state = step(system, state)
        state = thermostat_post(state)
        if print_timings:
            system["coordinates"].block_until_ready()
            tstep += time.time() - tstep0

        if istep % ndump == 0:
            print("write XYZ frame")
            write_arc_frame(
                fout,
                symbols,
                np.asarray(system["coordinates"].reshape(nbeads, nat, 3)[0]),
            )
            print("ns/day: ", nsperday)
            print(header)
            tperstep = (time.time() - t0dump) / ndump
            t0dump = time.time()
            nsperday = (24 * 60 * 60 / tperstep) * dt * au.PS / 1000
            # model.reinitialize_preprocessing()

        if istep % nprint == 0:
            if jnp.any(jnp.isnan(system["eigv"])) or jnp.any(
                jnp.isnan(system["coordinates"])
            ):
                raise ValueError("dynamics crashed.")
            t1 = time.time()
            tperstep = (t1 - t0) / nprint
            t0 = t1
            nsperday = (24 * 60 * 60 / tperstep) * dt * au.PS / 1000
            ek = system["ek"]
            ek_c = system["ek_c"]
            e = np.mean(np.array(system["total_energy"]))
            line = f"{istep:10} {(start_time+istep*dt)*au.PS:10.3f} {ek+e:10.3f} {e:10.3f} {ek:10.3f} {2*ek/(3.*nat)*au.KELVIN:10.3f} {2*ek_c/(3.*nat)*au.KELVIN:10.3f} {nsperday:10.3f} {1./tperstep:10.3f}"

            print(line)
            if print_timings:
                print(f"tpre: {tpre/nprint:.5f}; tstep: {tstep/nprint:.5f}")
                tpre = 0
                tstep = 0

    print(f"Run done in {(time.time()-tstart_dyn)/60.0} minutes")
    ### close trajectory file
    fout.close()


if __name__ == "__main__":
    main()
