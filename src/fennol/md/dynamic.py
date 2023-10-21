import sys, os, io
import argparse
import time
from pathlib import Path
import math

import numpy as np
from typing import Optional,Callable
from functools import partial
import jax
import jax.numpy as jnp


from ..models import FENNIX
from ..utils.io import write_arc_frame, last_xyz_frame
from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX,ATOMIC_MASSES
from ..utils.atomic_units import AtomicUnits as au
from ..utils.input_parser import parse_input
from .thermostats import get_thermostat

def minmaxone(x,name=""):
    print(name,x.min(),x.max(),(x**2).mean())


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
    device:str = simulation_parameters.get("device", "cpu")
    if device == "cpu":
        device="cpu"
    elif device.startswith("cuda"):
        num=device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = num
        device = "gpu"
    
    _device = jax.devices(device)[0]

    ### Set the precision
    jax.config.update("jax_enable_x64", True)
    fprec='float32'

    with jax.default_device(_device):
        dynamic(simulation_parameters,device,fprec)


def dynamic(simulation_parameters,device,fprec):
    ### Initialize the model
    # assert 'model_file' in simulation_parameters, "model_file not specified in parameter file"
    model_file = simulation_parameters.get("model_file")
    model_file = Path(str(model_file).strip())
    if not model_file.exists():
        raise FileNotFoundError(f"model file {model_file} not found")
    else:
        model = FENNIX.load(model_file)  # \
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
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols],dtype=np.int32)
    nat = species.shape[0]
    mass_amu = np.array(ATOMIC_MASSES,dtype=fprec)[species]
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
        cell = np.array(cell,dtype=fprec).reshape(3, 3).T
        volume = np.linalg.det(cell)
        print("cell matrix:")
        print(cell)
        dens = totmass_amu / 6.02214129e-1 / volume
        print("density: ", dens.item(), " g/cm^3")

    if crystal_input:
        assert cell is not None, "cell must be specified for crystal units"
        coordinates = coordinates @ cell #.double()
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
    nblist_skin = simulation_parameters.get("nblist_skin", 0.)
    nprint = int(simulation_parameters.get("nprint", 10))
    print("nprint: ", nprint, "; nblist_stride: ", nblist_stride)

    random_seed = simulation_parameters.get("random_seed",np.random.randint(0,2**32-1))
    print(f"random_seed: {random_seed}")
    rng_key = jax.random.PRNGKey(random_seed)

    ### Set the thermostat
    thermostat_name = str(simulation_parameters.get("thermostat", "NONE")).upper()
    thermostat  = get_thermostat(thermostat_name,dt=dt,mass=mass,gamma=gamma,kT=kT)

    dt2m = jnp.asarray(dt2 / mass[:, None],dtype=fprec)

    @jax.jit
    def step(system):
        v = system["vel"]
        f = system["forces"]
        x = system["coordinates"]

        v = v + f * dt2m
        x = x + dt2 * v
        v,rng_key = thermostat(v,system["rng_key"])
        x = x + dt2 * v
        system = {**system, "coordinates": x , "rng_key": rng_key}
        _,f, system = model._energy_and_forces(model.variables,system)
        v = v + f * dt2m

        ek=0.5*jnp.sum(mass[:,None]*v**2)
        system["vel"]=v
        system["ek"]=ek

        return system
    
    # @partial(jax.jit, static_argnums=1)
    # def integrate(system,nsteps):
    #     return jax.lax.fori_loop(0,nsteps,step,system)
    
    system = model.preprocess(species=species, coordinates=coordinates)
    if simulation_parameters.get("print_model_summary",False):
        print(model.summarize(example_data=system))
    # initial energy and forces
    print("Computing initial energy and forces")
    e, f, system = model._energy_and_forces(model.variables,system)
    print(f"Initial energy: {e}")

    rng_key,v_key = jax.random.split(rng_key)
    v=jnp.zeros_like(coordinates) #jax.random.normal(v_key,coordinates.shape)*(kT/mass[:,None])**0.5
    system["rng_key"]=rng_key
    ek=0.5*jnp.sum(mass[:,None]*v**2)
    system["vel"]=v
    system["ek"]=ek
    system["nblist_skin"]=nblist_skin
    if "nblist_mult_size" in simulation_parameters:
        system["nblist_mult_size"]=simulation_parameters["nblist_mult_size"]

    ### Print header
    print(f"Running {nsteps} steps of MD simulation on {device}")
    header = "#     step       time       etot       epot         ek       temp     ns/day   step/s"
    print(header)
    fout = open(traj_file, "a+")
    t0 = time.time()
    tstart_dyn=t0
    istep=0
    for istep in range(1,nsteps+1):
        ### BAOAB evolution
        if istep % nblist_stride == 0:
            system = model.preprocess(**system)

        system = step(system)

        if istep % ndump == 0:
            print("write XYZ frame")
            write_arc_frame(fout, symbols, coordinates)
            print(header)
            # model.reinitialize_preprocessing()

        if istep % nprint == 0:
            if jnp.any(jnp.isnan(system["vel"])) or jnp.any(jnp.isnan(system["coordinates"])):
                raise ValueError("dynamics crashed.")
            t1 = time.time()
            tperstep = (t1 - t0) / nprint
            t0 = t1
            nsperday = (24 * 60 * 60 / tperstep) * dt * au.PS / 1000
            ek=system["ek"]
            e=system["total_energy"][0]
            line = f"{istep:10} {(start_time+istep*dt)*au.PS:10.3f} {ek+e:10.3f} {e:10.3f} {ek:10.3f} {2*ek/(3.*nat)*au.KELVIN:10.3f} {nsperday:10.3f} {1./tperstep:10.3f}"
            
            print(line)

    print(f"Run done in {(time.time()-tstart_dyn)/60.0} minutes")
    ### close trajectory file
    fout.close()


if __name__ == "__main__":
    main()
