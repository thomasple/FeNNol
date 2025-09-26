from pathlib import Path

import numpy as np
import jax.numpy as jnp

from flax.core import freeze, unfreeze

from ..utils.io import last_xyz_frame


from ..models import FENNIX

from ..utils.periodic_table import PERIODIC_TABLE_REV_IDX, ATOMIC_MASSES
from .utils import us
from ..utils import detect_topology,parse_cell,cell_is_triangular,tril_cell


def load_model(simulation_parameters):
    model_file = simulation_parameters.get("model_file")
    """@keyword[fennol_md] model_file
    Path to the machine learning model file (.fnx format). Required parameter.
    Type: str, Required
    """
    model_file = Path(str(model_file).strip())
    if not model_file.exists():
        raise FileNotFoundError(f"model file {model_file} not found")
    else:
        graph_config = simulation_parameters.get("graph_config", {})
        """@keyword[fennol_md] graph_config
        Advanced graph configuration for model initialization.
        Default: {}
        """
        model = FENNIX.load(model_file, graph_config=graph_config)  # \
        print(f"# model_file: {model_file}")

    if "energy_terms" in simulation_parameters:
        energy_terms = simulation_parameters["energy_terms"]
        if isinstance(energy_terms, str):
            energy_terms = energy_terms.split()
        model.set_energy_terms(energy_terms)
        print("# energy terms:", model.energy_terms)

    return model


def load_system_data(simulation_parameters, fprec):
    ## LOAD SYSTEM CONFORMATION FROM FILES
    system_name = str(simulation_parameters.get("system_name", "system")).strip()
    """@keyword[fennol_md] system_name
    Name prefix for output files. If not specified, uses the xyz filename stem.
    Default: "system"
    """
    indexed = simulation_parameters.get("xyz_input/indexed", False)
    """@keyword[fennol_md] xyz_input/indexed
    Whether first column contains atom indices (Tinker format).
    Default: False
    """
    has_comment_line = simulation_parameters.get("xyz_input/has_comment_line", True)
    """@keyword[fennol_md] xyz_input/has_comment_line
    Whether file contains comment lines.
    Default: True
    """
    xyzfile = Path(simulation_parameters.get("xyz_input/file", system_name + ".xyz"))
    """@keyword[fennol_md] xyz_input/file
    Path to xyz/arc coordinate file. Required parameter.
    Type: str, Required
    """
    if not xyzfile.exists():
        raise FileNotFoundError(f"xyz file {xyzfile} not found")
    system_name = str(simulation_parameters.get("system_name", xyzfile.stem)).strip()
    symbols, coordinates, _ = last_xyz_frame(
        xyzfile, indexed=indexed, has_comment_line=has_comment_line
    )
    coordinates = coordinates.astype(fprec)
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)
    nat = species.shape[0]

    ## GET MASS
    mass_Da = np.array(ATOMIC_MASSES, dtype=fprec)[species]
    deuterate = simulation_parameters.get("deuterate", False)
    """@keyword[fennol_md] deuterate
    Replace hydrogen masses with deuterium masses.
    Default: False
    """
    if deuterate:
        print("# Replacing all hydrogens with deuteriums")
        mass_Da[species == 1] *= 2.0

    totmass_Da = mass_Da.sum()

    mass = mass_Da.copy()
    hmr = simulation_parameters.get("hmr", 0)
    """@keyword[fennol_md] hmr
    Hydrogen mass repartitioning factor. 0 = no repartitioning.
    Default: 0
    """
    if hmr > 0:
        print(f"# Adding {hmr} Da to H masses and repartitioning on others for total mass conservation.")
        Hmask = species == 1
        added_mass = hmr * Hmask.sum()
        mass[Hmask] += hmr
        wmass = mass[~Hmask]
        mass[~Hmask] -= added_mass * wmass/ wmass.sum()

        assert np.isclose(mass.sum(), totmass_Da), "Mass conservation failed"

    # convert to internal units
    mass = mass / us.DA

    ### GET TEMPERATURE
    temperature = np.clip(simulation_parameters.get("temperature", 300.0), 1.0e-6, None)
    """@keyword[fennol_md] temperature
    Target temperature in Kelvin.
    Default: 300.0
    """
    kT = us.K_B * temperature 

    ### GET TOTAL CHARGE
    total_charge = simulation_parameters.get("total_charge", None)
    """@keyword[fennol_md] total_charge
    Total system charge for charged systems.
    Default: None (interpreted as 0)
    """
    if total_charge is None:
        total_charge = 0
    else:
        total_charge = int(total_charge)
        print("# total charge: ", total_charge,"e")

    ### ENERGY UNIT
    energy_unit_str = simulation_parameters.get("energy_unit", "kcal/mol")
    """@keyword[fennol_md] energy_unit
    Energy unit for output. Common options: 'kcal/mol', 'eV', 'Ha', 'kJ/mol'.
    Default: "kcal/mol"
    """
    energy_unit = us.get_multiplier(energy_unit_str)

    ## SYSTEM DATA
    system_data = {
        "name": system_name,
        "nat": nat,
        "symbols": symbols,
        "species": species,
        "mass": mass,
        "mass_Da": mass_Da,
        "totmass_Da": totmass_Da,
        "temperature": temperature,
        "kT": kT,
        "total_charge": total_charge,
        "energy_unit": energy_unit,
        "energy_unit_str": energy_unit_str,
    }
    input_flags = simulation_parameters.get("model_flags", [])
    """@keyword[fennol_md] model_flags
    Additional flags to pass to the model.
    Default: []
    """
    flags = {f:None for f in input_flags}

    ### Set boundary conditions
    cell = simulation_parameters.get("cell", None)
    """@keyword[fennol_md] cell
    Unit cell vectors. Required for PBC. It is a sequence of floats:
    - 9 floats: components of cell vectors [ax, ay, az, bx, by, bz, cx, cy, cz]
    - 6 floats: lengths and angles [a, b, c, alpha, beta, gamma]
    - 3 floats: lengths of cell vectors [a, b, c] (orthorhombic)
    - 1 float: length of cell vectors (cubic cell)
    Lengths are in Angstroms, angles in degrees.
    Default: None
    """
    if cell is not None:
        cell = parse_cell(cell).astype(fprec)
        rotate_cell = not cell_is_triangular(cell)
        if rotate_cell:
            print("# Warning: provided cell is not lower triangular. Rotating to canonical cell orientation.")
            cell, cell_rotation = tril_cell(cell)
        # cell = np.array(cell, dtype=fprec).reshape(3, 3)
        reciprocal_cell = np.linalg.inv(cell)
        volume = np.abs(np.linalg.det(cell))
        print("# cell matrix:")
        for l in cell:
            print("# ", l)
        # print(cell)
        dens = (totmass_Da/volume) * (us.MOL/us.CM**3)
        print("# density: ", dens.item(), " g/cm^3")
        minimum_image = simulation_parameters.get("minimum_image", True)
        """@keyword[fennol_md] minimum_image
        Use minimum image convention for neighbor lists in periodic systems.
        Default: True
        """
        estimate_pressure = simulation_parameters.get("estimate_pressure", False)
        """@keyword[fennol_md] estimate_pressure
        Calculate and print pressure during simulation.
        Default: False
        """
        print("# minimum_image: ", minimum_image)

        crystal_input = simulation_parameters.get("xyz_input/crystal", False)
        """@keyword[fennol_md] xyz_input/crystal
        Use crystal coordinates.
        Default: False
        """
        if crystal_input:
            coordinates = coordinates @ cell
        
        if rotate_cell:
            coordinates = coordinates @ cell_rotation

        pbc_data = {
            "cell": cell,
            "reciprocal_cell": reciprocal_cell,
            "volume": volume,
            "minimum_image": minimum_image,
            "estimate_pressure": estimate_pressure,
        }
        if minimum_image:
            flags["minimum_image"] = None
    else:
        pbc_data = None
    system_data["pbc"] = pbc_data
    system_data["initial_coordinates"] = coordinates.copy()

    ### TOPOLOGY
    topology_key = simulation_parameters.get("topology", None)
    """@keyword[fennol_md] topology
    Topology specification for molecular systems. Use "detect" for automatic detection.
    Default: None
    """
    if topology_key is not None:
        topology_key = str(topology_key).strip()
        if topology_key.lower() == "detect":
            topology = detect_topology(species,coordinates,cell=cell)
            np.savetxt(system_name +".topo", topology+1, fmt="%d")
            print("# Detected topology saved to", system_name + ".topo")
        else:
            assert Path(topology_key).exists(), f"Topology file {topology_key} not found"
            topology = np.loadtxt(topology_key, dtype=np.int32)-1
            assert topology.shape[1] == 2, "Topology file must have two columns (source, target)"
            print("# Topology loaded from", topology_key)
    else:
        topology = None
    
    system_data["topology"] = topology

    ### PIMD
    nbeads = simulation_parameters.get("nbeads", None)
    """@keyword[fennol_md] nbeads
    Number of beads for Path Integral MD.
    Default: None
    """
    nreplicas = simulation_parameters.get("nreplicas", None)
    """@keyword[fennol_md] nreplicas
    Number of replicas for independent replica simulations.
    Default: None
    """
    if nbeads is not None:
        nbeads = int(nbeads)
        print("# nbeads: ", nbeads)
        system_data["nbeads"] = nbeads
        coordinates = np.repeat(coordinates[None, :, :], nbeads, axis=0).reshape(-1, 3)
        species = np.repeat(species[None, :], nbeads, axis=0).reshape(-1)
        bead_index = np.arange(nbeads, dtype=np.int32).repeat(nat)
        natoms = np.array([nat] * nbeads, dtype=np.int32)

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
        omk = (nbeads * kT / us.HBAR) * omk**0.5
        for i in range(nbeads):
            if eigmat[i, 0] < 0:
                eigmat[i] *= -1.0
        eigmat = jnp.asarray(eigmat, dtype=fprec)
        system_data["omk"] = omk
        system_data["eigmat"] = eigmat
        nreplicas = None
    elif nreplicas is not None:
        nreplicas = int(nreplicas)
        print("# nreplicas: ", nreplicas)
        system_data["nreplicas"] = nreplicas
        system_data["mass"] = np.repeat(mass[None, :], nreplicas, axis=0).reshape(-1)
        system_data["species"] = np.repeat(species[None, :], nreplicas, axis=0).reshape(
            -1
        )
        coordinates = np.repeat(coordinates[None, :, :], nreplicas, axis=0).reshape(
            -1, 3
        )
        species = np.repeat(species[None, :], nreplicas, axis=0).reshape(-1)
        bead_index = np.arange(nreplicas, dtype=np.int32).repeat(nat)
        natoms = np.array([nat] * nreplicas, dtype=np.int32)
    else:
        system_data["nreplicas"] = 1
        bead_index = np.array([0] * nat, dtype=np.int32)
        natoms = np.array([nat], dtype=np.int32)

    conformation = {
        "species": species,
        "coordinates": coordinates,
        "batch_index": bead_index,
        "natoms": natoms,
        "total_charge": total_charge,
    }
    if cell is not None:
        cell = cell[None, :, :]
        reciprocal_cell = reciprocal_cell[None, :, :]
        if nbeads is not None:
            cell = np.repeat(cell, nbeads, axis=0)
            reciprocal_cell = np.repeat(reciprocal_cell, nbeads, axis=0)
        elif nreplicas is not None:
            cell = np.repeat(cell, nreplicas, axis=0)
            reciprocal_cell = np.repeat(reciprocal_cell, nreplicas, axis=0)
        conformation["cells"] = cell
        conformation["reciprocal_cells"] = reciprocal_cell

    additional_keys = simulation_parameters.get("additional_keys", {})
    """@keyword[fennol_md] additional_keys
    Additional custom keys for model input.
    Default: {}
    """
    for key, value in additional_keys.items():
        conformation[key] = value
    
    conformation["flags"] = flags

    return system_data, conformation


def initialize_preprocessing(simulation_parameters, model, conformation, system_data):
    nblist_verbose = simulation_parameters.get("nblist_verbose", False)
    """@keyword[fennol_md] nblist_verbose
    Print detailed neighbor list information.
    Default: False
    """
    nblist_skin = simulation_parameters.get("nblist_skin", -1.0)
    """@keyword[fennol_md] nblist_skin
    Neighbor list skin distance in Angstroms.
    Default: -1.0 (automatic)
    """

    ### CONFIGURE PREPROCESSING
    preproc_state = unfreeze(model.preproc_state)
    layer_state = []
    for st in preproc_state["layers_state"]:
        stnew = unfreeze(st)
        if nblist_skin > 0:
            stnew["nblist_skin"] = nblist_skin
        if "nblist_mult_size" in simulation_parameters:
            stnew["nblist_mult_size"] = simulation_parameters["nblist_mult_size"]
            """@keyword[fennol_md] nblist_mult_size
            Multiplier for neighbor list size.
            Default: None
            """
        if "nblist_add_neigh" in simulation_parameters:
            stnew["add_neigh"] = simulation_parameters["nblist_add_neigh"]
            """@keyword[fennol_md] nblist_add_neigh
            Additional neighbors to include in lists.
            Default: None
            """
        layer_state.append(freeze(stnew))
    preproc_state["layers_state"] = layer_state
    preproc_state = freeze(preproc_state)

    ## initial preprocessing
    preproc_state = preproc_state.copy({"check_input": True})
    preproc_state, conformation = model.preprocessing(preproc_state, conformation)

    preproc_state = preproc_state.copy({"check_input": False})

    if nblist_verbose:
        graphs_keys = list(model._graphs_properties.keys())
        print("# graphs_keys: ", graphs_keys)
        print("# nblist state:", preproc_state)

    ### print model
    if simulation_parameters.get("print_model", False):
        """@keyword[fennol_md] print_model
        Print detailed model information at startup.
        Default: False
        """
        print(model.summarize(example_data=conformation))

    return preproc_state, conformation



