import sys, os, io
import argparse
import time
from pathlib import Path

import numpy as np
from collections import defaultdict
import jax
import jax.numpy as jnp
import pickle
import yaml

from ..utils.io import (
    write_arc_frame,
    write_xyz_frame,
    write_extxyz_frame,
    human_time_duration,
)
from .utils import wrapbox, save_dynamics_restart
from ..utils import minmaxone, AtomicUnits as au,read_tinker_interval
from ..utils.input_parser import parse_input,convert_dict_units, InputFile
from .integrate import initialize_dynamics


def main():
    """
    Main entry point for the fennol_md command-line interface.
    
    Parses command-line arguments and runs a molecular dynamics simulation
    based on the provided parameter file.
    
    Command-line Usage:
        fennol_md input.fnl
        fennol_md config.yaml
    
    Returns:
        int: Exit code (0 for success)
    """
    # os.environ["OMP_NUM_THREADS"] = "1"
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )
    
    ### Read the parameter file
    parser = argparse.ArgumentParser(prog="fennol_md")
    parser.add_argument("param_file", type=Path, help="Parameter file")
    args = parser.parse_args()
    param_file = args.param_file

    return config_and_run_dynamic(param_file)

def config_and_run_dynamic(param_file: Path):
    """
    Configure and run a molecular dynamics simulation.
    
    This function loads simulation parameters from a configuration file,
    sets up the computation device and precision, and runs the MD simulation.
    
    Parameters:
        param_file (Path): Path to the parameter file (.fnl, .yaml, or .yml)
    
    Returns:
        int: Exit code (0 for success)
        
    Raises:
        FileNotFoundError: If the parameter file doesn't exist
        ValueError: If the parameter file format is unsupported
        
    Supported file formats:
        - .fnl: FeNNol native format
        - .yaml/.yml: YAML format
    
    Unit conversion in the parameter file:
        - Units specified in brackets: dt[fs] = 0.5
        - All units converted to atomic units internally
    """

    if not param_file.exists() and not param_file.is_file():
        raise FileNotFoundError(f"Parameter file {param_file} not found")
    
    if param_file.suffix in [".yaml", ".yml"]:
        with open(param_file, "r") as f:
            simulation_parameters = convert_dict_units(yaml.safe_load(f))
            simulation_parameters = InputFile(**simulation_parameters)
    elif param_file.suffix == ".fnl":
        simulation_parameters = parse_input(param_file)
    else:
        raise ValueError(
            f"Unknown parameter file format '{param_file.suffix}'. Supported formats are '.yaml', '.yml' and '.fnl'"
        )

    ### Set the device
    if "FENNOL_DEVICE" in os.environ:
        device = os.environ["FENNOL_DEVICE"].lower()
        print(f"# Setting device from env FENNOL_DEVICE={device}")
    else:
        device = simulation_parameters.get("device", "cpu").lower()
        """@keyword[fennol_md] device
        Computation device. Options: 'cpu', 'cuda:N', 'gpu:N' where N is device number.
        Default: 'cpu'
        """
    if device == "cpu":
        jax.config.update('jax_platforms', 'cpu')
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
    enable_x64 = simulation_parameters.get("double_precision", False)
    """@keyword[fennol_md] double_precision
    Enable double precision (64-bit) calculations. Default is single precision (32-bit).
    Default: False
    """
    jax.config.update("jax_enable_x64", enable_x64)
    fprec = "float64" if enable_x64 else "float32"

    matmul_precision = simulation_parameters.get("matmul_prec", "highest").lower()
    """@keyword[fennol_md] matmul_prec
    Matrix multiplication precision. Options: 'default', 'high', 'highest'.
    Default: "highest"
    """
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
    return dynamic(simulation_parameters, device, fprec)


def dynamic(simulation_parameters, device, fprec):
    """
    Execute the molecular dynamics simulation loop.
    
    This function performs the main MD simulation using the initialized
    system, integrator, and thermostats/barostats. It handles trajectory
    output, property monitoring, and restart file generation.
    
    Parameters:
        simulation_parameters: Parsed simulation parameters
        device (str): Computation device ('cpu' or 'gpu')
        fprec (str): Floating point precision ('float32' or 'float64')
        
    Returns:
        int: Exit code (0 for success)
    """
    tstart_dyn = time.time()

    random_seed = simulation_parameters.get(
        "random_seed", np.random.randint(0, 2**32 - 1)
    )
    """@keyword[fennol_md] random_seed
    Random seed for reproducible simulations. If not specified, a random seed is generated.
    Default: Random integer between 0 and 2^32-1
    """
    print(f"# random_seed: {random_seed}")
    rng_key = jax.random.PRNGKey(random_seed)

    ## INITIALIZE INTEGRATOR AND SYSTEM
    rng_key, subkey = jax.random.split(rng_key)
    step, update_conformation, system_data, dyn_state, conformation, system = (
        initialize_dynamics(simulation_parameters, fprec, subkey)
    )

    nat = system_data["nat"]
    dt = dyn_state["dt"]
    ## get number of steps
    nsteps = int(simulation_parameters.get("nsteps"))
    """@keyword[fennol_md] nsteps
    Total number of MD steps to perform. Required parameter.
    Type: int, Required
    """
    start_time_ps = dyn_state.get("start_time_ps", 0.0)

    ### Set I/O parameters
    Tdump = simulation_parameters.get("tdump", 1.0 / au.PS) * au.FS
    """@keyword[fennol_md] tdump
    Time interval between trajectory frames.
    Default: 1.0 ps
    """
    ndump = int(Tdump / dt)
    system_name = system_data["name"]
    estimate_pressure = dyn_state["estimate_pressure"]

    @jax.jit
    def check_nan(system):
        return jnp.any(jnp.isnan(system["vel"])) | jnp.any(
            jnp.isnan(system["coordinates"])
        )

    if system_data["pbc"] is not None:
        cell = system["cell"]
        reciprocal_cell = np.linalg.inv(cell)
        do_wrap_box = simulation_parameters.get("wrap_box", False)
        """@keyword[fennol_md] wrap_box
        Wrap coordinates into primary unit cell.
        Default: False
        """
        if do_wrap_box:
            wrap_groups_def = simulation_parameters.get("wrap_groups",None)
            """@keyword[fennol_md] wrap_groups
            Specific atom groups to wrap independently. Dictionary mapping group names to atom indices.
            Default: None
            """
            if wrap_groups_def is None:
                wrap_groups = None
            else:
                wrap_groups = {}
                assert isinstance(wrap_groups_def, dict), "wrap_groups must be a dictionary"
                for k, v in wrap_groups_def.items():
                    wrap_groups[k]=read_tinker_interval(v)
                # check that pairwise intersection of wrap groups is empty
                wrap_groups_keys = list(wrap_groups.keys())
                for i in range(len(wrap_groups_keys)):
                    i_key = wrap_groups_keys[i]
                    w1 = set(wrap_groups[i_key])
                    for j in range(i + 1, len(wrap_groups_keys)):
                        j_key = wrap_groups_keys[j]
                        w2 = set(wrap_groups[j_key])
                        if  w1.intersection(w2):
                            raise ValueError(
                                f"Wrap groups {i_key} and {j_key} have common atoms: {w1.intersection(w2)}"
                            )
                group_all = np.concatenate(list(wrap_groups.values()))
                # get all atoms that are not in any wrap group
                group_none = np.setdiff1d(np.arange(nat), group_all)
                print(f"# Wrap groups: {wrap_groups}")
                wrap_groups["__other"] = group_none
                wrap_groups = ((k, v) for k, v in wrap_groups.items())

    else:
        cell = None
        reciprocal_cell = None
        do_wrap_box = False
        wrap_groups = None

    ### Energy units and print initial energy
    model_energy_unit = system_data["model_energy_unit"]
    model_energy_unit_str = system_data["model_energy_unit_str"]
    per_atom_energy = simulation_parameters.get("per_atom_energy", True)
    """@keyword[fennol_md] per_atom_energy
    Print energies per atom instead of total energies.
    Default: True
    """
    energy_unit_str = system_data["energy_unit_str"]
    energy_unit = system_data["energy_unit"]
    print("# Energy unit: ", energy_unit_str)
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
    nprint = int(simulation_parameters.get("nprint", 10))
    """@keyword[fennol_md] nprint
    Number of steps between energy/property printing.
    Default: 10
    """
    assert nprint > 0, "nprint must be > 0"
    nsummary = simulation_parameters.get("nsummary", 100 * nprint)
    """@keyword[fennol_md] nsummary
    Number of steps between summary statistics.
    Default: 100 * nprint
    """
    assert nsummary > nprint, "nsummary must be > nprint"

    save_keys = simulation_parameters.get("save_keys", [])
    """@keyword[fennol_md] save_keys
    Additional model output keys to save to trajectory.
    Default: []
    """
    if save_keys:
        print(f"# Saving keys: {save_keys}")
        fkeys = open(f"{system_name}.traj.pkl", "wb+")
    else:
        fkeys = None
    
    ### initialize colvars
    use_colvars = "colvars" in dyn_state
    if use_colvars:
        print(f"# Colvars: {dyn_state['colvars']}")
        colvars_names = dyn_state["colvars"]
        # open colvars file and print header
        fcolvars = open(f"{system_name}.colvars.traj", "a")
        fcolvars.write("#time[ps] ")
        fcolvars.write(" ".join(colvars_names))
        fcolvars.write("\n")
        fcolvars.flush()

    ### Print header
    include_thermostat_energy = "thermostat_energy" in system["thermostat"]
    thermostat_name = dyn_state["thermostat_name"]
    pimd = dyn_state["pimd"]
    variable_cell = dyn_state["variable_cell"]
    nbeads = system_data.get("nbeads", 1)
    dyn_name = "PIMD" if pimd else "MD"
    print("#" * 84)
    print(
        f"# Running {nsteps:_} steps of {thermostat_name} {dyn_name} simulation on {device}"
    )
    header = f"#{'Step':>9} {'Time[ps]':>10} {'Etot':>10} {'Epot':>10} {'Ekin':>10} {'Temp[K]':>10}"
    if pimd:
        header += f" {'Temp_c[K]':>10}"
    if include_thermostat_energy:
        header += f" {'Etherm':>10}"
    if estimate_pressure:
        print_aniso_pressure = simulation_parameters.get("print_aniso_pressure", False)
        """@keyword[fennol_md] print_aniso_pressure
        Print anisotropic pressure tensor components.
        Default: False
        """
        pressure_unit_str = simulation_parameters.get("pressure_unit", "atm")
        """@keyword[fennol_md] pressure_unit
        Pressure unit for output. Options: 'atm', 'bar', 'Pa', 'GPa'.
        Default: "atm"
        """
        pressure_unit = au.get_multiplier(pressure_unit_str) * au.BOHR**3
        p_str = f"  Press[{pressure_unit_str}]"
        header += f" {p_str:>10}"
    if variable_cell:
        header += f" {'Density':>10}"
    print(header)

    ### Open trajectory file
    traj_format = simulation_parameters.get("traj_format", "arc").lower()
    """@keyword[fennol_md] traj_format
    Trajectory file format. Options: 'arc' (Tinker), 'xyz' (standard), 'extxyz' (extended).
    Default: "arc"
    """
    if traj_format == "xyz":
        traj_ext = ".traj.xyz"
        write_frame = write_xyz_frame
    elif traj_format == "extxyz":
        traj_ext = ".traj.extxyz"
        write_frame = write_extxyz_frame
    elif traj_format == "arc":
        traj_ext = ".arc"
        write_frame = write_arc_frame
    else:
        raise ValueError(
            f"Unknown trajectory format '{traj_format}'. Supported formats are 'arc' and 'xyz'"
        )

    write_all_beads = simulation_parameters.get("write_all_beads", False) and pimd
    """@keyword[fennol_md] write_all_beads
    Write all PIMD beads to separate trajectory files.
    Default: False
    """

    if write_all_beads:
        fout = [
            open(f"{system_name}_bead{i+1:03d}" + traj_ext, "a") for i in range(nbeads)
        ]
    else:
        fout = open(system_name + traj_ext, "a")

    ensemble_key = simulation_parameters.get("etot_ensemble_key", None)
    """@keyword[fennol_md] etot_ensemble_key
    Key for ensemble weighting in enhanced sampling.
    Default: None
    """
    if ensemble_key is not None:
        fens = open(f"{system_name}.ensemble_weights.traj", "a")

    write_centroid = simulation_parameters.get("write_centroid", False) and pimd
    """@keyword[fennol_md] write_centroid
    Write PIMD centroid coordinates to separate file.
    Default: False
    """
    if write_centroid:
        fcentroid = open(f"{system_name}_centroid" + traj_ext, "a")

    ### initialize proprerty trajectories
    properties_traj = defaultdict(list)

    ### initialize counters and timers
    t0 = time.time()
    t0dump = t0
    istep = 0
    t0full = time.time()
    force_preprocess = False

    for istep in range(1, nsteps + 1):

        ### update the system
        dyn_state, system, conformation, model_output = step(
            istep, dyn_state, system, conformation, force_preprocess
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
            simulated_time = start_time_ps + istep * dt / 1000
            line = f"{istep:10.6g} {simulated_time:10.3f} {etot*atom_energy_unit:#10.4f}  {epot*atom_energy_unit:#10.4f} {ek*atom_energy_unit:#10.4f} {temper:10.2f}"
            if pimd:
                line += f" {temper_c:10.2f}"
            if include_thermostat_energy:
                line += f" {etherm*atom_energy_unit:#10.4f}"
                properties_traj[f"Etherm[{atom_energy_unit_str}]"].append(
                    etherm * atom_energy_unit
                )
            if estimate_pressure:
                pres = system["pressure"] * pressure_unit
                properties_traj[f"Pressure[{pressure_unit_str}]"].append(pres)
                if print_aniso_pressure:
                    pres_tensor = system["pressure_tensor"] * pressure_unit
                    pres_tensor = 0.5 * (pres_tensor + pres_tensor.T)
                    properties_traj[f"Pressure_xx[{pressure_unit_str}]"].append(
                        pres_tensor[0, 0]
                    )
                    properties_traj[f"Pressure_yy[{pressure_unit_str}]"].append(
                        pres_tensor[1, 1]
                    )
                    properties_traj[f"Pressure_zz[{pressure_unit_str}]"].append(
                        pres_tensor[2, 2]
                    )
                    properties_traj[f"Pressure_xy[{pressure_unit_str}]"].append(
                        pres_tensor[0, 1]
                    )
                    properties_traj[f"Pressure_xz[{pressure_unit_str}]"].append(
                        pres_tensor[0, 2]
                    )
                    properties_traj[f"Pressure_yz[{pressure_unit_str}]"].append(
                        pres_tensor[1, 2]
                    )
                line += f" {pres:10.3f}"
            if variable_cell:
                density = system["density"]
                properties_traj["Density[g/cm^3]"].append(density)
                if print_aniso_pressure:
                    cell = system["cell"]
                    properties_traj[f"Cell_Ax[Angstrom]"].append(cell[0, 0])
                    properties_traj[f"Cell_Ay[Angstrom]"].append(cell[0, 1])
                    properties_traj[f"Cell_Az[Angstrom]"].append(cell[0, 2])
                    properties_traj[f"Cell_Bx[Angstrom]"].append(cell[1, 0])
                    properties_traj[f"Cell_By[Angstrom]"].append(cell[1, 1])
                    properties_traj[f"Cell_Bz[Angstrom]"].append(cell[1, 2])
                    properties_traj[f"Cell_Cx[Angstrom]"].append(cell[2, 0])
                    properties_traj[f"Cell_Cy[Angstrom]"].append(cell[2, 1])
                    properties_traj[f"Cell_Cz[Angstrom]"].append(cell[2, 2])
                line += f" {density:10.4f}"
                if "piston_temperature" in system["barostat"]:
                    piston_temperature = system["barostat"]["piston_temperature"]
                    properties_traj["T_piston[Kelvin]"].append(piston_temperature)

            print(line)

            ### save colvars
            if use_colvars:
                colvars = system["colvars"]
                fcolvars.write(f"{simulated_time:.3f} ")
                fcolvars.write(" ".join([f"{colvars[k]:.6f}" for k in colvars_names]))
                fcolvars.write("\n")
                fcolvars.flush()

            if save_keys:
                data = {
                    k: (
                        np.asarray(model_output[k])
                        if isinstance(model_output[k], jnp.ndarray)
                        else model_output[k]
                    )
                    for k in save_keys
                }
                data["properties"] = {
                    k: float(v) for k, v in zip(header.split()[1:], line.split())
                }
                data["properties"]["properties_energy_unit"] = (atom_energy_unit,atom_energy_unit_str)
                data["properties"]["model_energy_unit"] = (model_energy_unit,model_energy_unit_str)

                pickle.dump(data, fkeys)

        ### save frame
        if istep % ndump == 0:
            line = "# Write XYZ frame"
            if variable_cell:
                cell = np.array(system["cell"])
                reciprocal_cell = np.linalg.inv(cell)
            if do_wrap_box:
                if pimd:
                    centroid = wrapbox(system["coordinates"][0], cell, reciprocal_cell,wrap_groups=wrap_groups)
                    system["coordinates"] = system["coordinates"].at[0].set(centroid)
                else:
                    system["coordinates"] = wrapbox(
                        system["coordinates"], cell, reciprocal_cell,wrap_groups=wrap_groups
                    )
                conformation = update_conformation(conformation, system)
                line += " (atoms have been wrapped into the box)"
                force_preprocess = True
            print(line)

            save_dynamics_restart(system_data, conformation, dyn_state, system)

            properties = {
                "energy": float(system["epot"]) * energy_unit,
                "Time_ps": start_time_ps + istep * dt / 1000,
                "energy_unit": energy_unit_str,
            }

            if write_all_beads:
                coords = np.asarray(conformation["coordinates"].reshape(-1, nat, 3))
                for i, fb in enumerate(fout):
                    write_frame(
                        fb,
                        system_data["symbols"],
                        coords[i],
                        cell=cell,
                        properties=properties,
                        forces=None,  # np.asarray(system["forces"].reshape(nbeads, nat, 3)[0]) * energy_unit,
                    )
            else:
                write_frame(
                    fout,
                    system_data["symbols"],
                    np.asarray(conformation["coordinates"].reshape(-1, nat, 3)[0]),
                    cell=cell,
                    properties=properties,
                    forces=None,  # np.asarray(system["forces"].reshape(nbeads, nat, 3)[0]) * energy_unit,
                )
            if write_centroid:
                centroid = np.asarray(system["coordinates"][0])
                write_frame(
                    fcentroid,
                    system_data["symbols"],
                    centroid,
                    cell=cell,
                    properties=properties,
                    forces=np.asarray(system["forces"].reshape(nbeads, nat, 3)[0])
                    * energy_unit,
                )
            if ensemble_key is not None:
                weights = " ".join(
                    [f"{w:.6f}" for w in system["ensemble_weights"].tolist()]
                )
                fens.write(f"{weights}\n")
                fens.flush()

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
            print(f"# Simulated time      : {istep * dt*1.e-3:.3f} ps")
            print(f"# Tot. Simu. time     : {start_time_ps + istep * dt*1.e-3:.3f} ps")
            print(f"# Tot. elapsed time   : {human_time_duration(elapsed_time)}")
            print(
                f"# Est. total duration   : {human_time_duration(estimated_total_time)}"
            )
            print(
                f"# Est. remaining time : {human_time_duration(estimated_remaining_time)}"
            )
            print(f"# Time for {nsummary:_} steps : {human_time_duration(tfull)}")

            corr_kin = system["thermostat"].get("corr_kin", None)
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
    if ensemble_key is not None:
        fens.close()
    if use_colvars:
        fcolvars.close()
    if fkeys is not None:
        fkeys.close()
    if write_centroid:
        fcentroid.close()

    return 0


if __name__ == "__main__":
    main()
