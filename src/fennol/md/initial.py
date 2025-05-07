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


def load_model(simulation_parameters):
    model_file = simulation_parameters.get("model_file")
    model_file = Path(str(model_file).strip())
    if not model_file.exists():
        raise FileNotFoundError(f"model file {model_file} not found")
    else:
        graph_config = simulation_parameters.get("graph_config", {})
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
    system_name = str(simulation_parameters.get("system", "system")).strip()
    indexed = simulation_parameters.get("xyz_input/indexed", True)
    has_comment_line = simulation_parameters.get("xyz_input/has_comment_line", False)
    xyzfile = Path(simulation_parameters.get("xyz_input/file", system_name + ".xyz"))
    if not xyzfile.exists():
        raise FileNotFoundError(f"xyz file {xyzfile} not found")
    system_name = str(simulation_parameters.get("system", xyzfile.stem)).strip()
    symbols, coordinates, _ = last_xyz_frame(
        xyzfile, indexed=indexed, has_comment_line=has_comment_line
    )
    coordinates = coordinates.astype(fprec)
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)
    nat = species.shape[0]

    ## GET MASS
    mass_amu = np.array(ATOMIC_MASSES, dtype=fprec)[species]
    deuterate = simulation_parameters.get("deuterate", False)
    if deuterate:
        print("# Replacing all hydrogens with deuteriums")
        mass_amu[species == 1] *= 2.0
    mass = mass_amu * (au.MPROT * (au.FS / au.BOHR) ** 2)

    ### GET TEMPERATURE
    temperature = np.clip(simulation_parameters.get("temperature", 300.0), 1.0e-6, None)
    kT = temperature / au.KELVIN
    totmass_amu = mass_amu.sum()/6.02214129e-1

    ## SYSTEM DATA
    system_data = {
        "name": system_name,
        "nat": nat,
        "symbols": symbols,
        "species": species,
        "mass": mass,
        "temperature": temperature,
        "kT": kT,
        "totmass_amu": totmass_amu,
    }

    ### Set boundary conditions
    cell = simulation_parameters.get("cell", None)
    if cell is not None:
        cell = np.array(cell, dtype=fprec).reshape(3, 3)
        reciprocal_cell = np.linalg.inv(cell)
        volume = np.abs(np.linalg.det(cell))
        print("# cell matrix:")
        for l in cell:
            print("# ", l)
        # print(cell)
        dens = totmass_amu  / volume
        print("# density: ", dens.item(), " g/cm^3")
        minimum_image = simulation_parameters.get("minimum_image", True)
        estimate_pressure = simulation_parameters.get("estimate_pressure", False)
        print("# minimum_image: ", minimum_image)

        crystal_input = simulation_parameters.get("xyz_input/crystal", False)
        if crystal_input:
            coordinates = coordinates @ cell

        pbc_data = {
            "cell": cell,
            "reciprocal_cell": reciprocal_cell,
            "volume": volume,
            "minimum_image": minimum_image,
            "estimate_pressure": estimate_pressure,
        }
    else:
        pbc_data = None
    system_data["pbc"] = pbc_data

    ### PIMD
    nbeads = simulation_parameters.get("nbeads", None)
    nreplicas = simulation_parameters.get("nreplicas", None)
    if nbeads is not None:
        nbeads = int(nbeads)
        print("# nbeads: ", nbeads)
        system_data["nbeads"] = nbeads
        coordinates = np.repeat(coordinates[None, :, :], nbeads, axis=0).reshape(-1,3)
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
        omk = nbeads * kT * omk**0.5 / au.FS
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
        system_data["species"] = np.repeat(species[None, :], nreplicas, axis=0).reshape(-1)
        coordinates = np.repeat(coordinates[None, :, :], nreplicas, axis=0).reshape(-1,3)
        species = np.repeat(species[None, :], nreplicas, axis=0).reshape(-1)
        bead_index = np.arange(nreplicas, dtype=np.int32).repeat(nat)
        natoms = np.array([nat] * nreplicas, dtype=np.int32)
    else:
        system_data["nreplicas"]=1
        bead_index = np.array([0] * nat, dtype=np.int32)
        natoms = np.array([nat], dtype=np.int32)

    conformation = {
        "species": species,
        "coordinates": coordinates,
        "batch_index": bead_index,
        "natoms": natoms,
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
    for key, value in additional_keys.items():
        conformation[key] = value

    return system_data, conformation


def initialize_preprocessing(simulation_parameters, model, conformation, system_data):
    nblist_verbose = simulation_parameters.get("nblist_verbose", False)
    nblist_skin = simulation_parameters.get("nblist_skin", -1.0)
    pbc_data = system_data.get("pbc", None)

    ### CONFIGURE PREPROCESSING
    preproc_state = unfreeze(model.preproc_state)
    layer_state = []
    for st in preproc_state["layers_state"]:
        stnew = unfreeze(st)
        if pbc_data is not None:
            stnew["minimum_image"] = pbc_data["minimum_image"]
        if nblist_skin > 0:
            stnew["nblist_skin"] = nblist_skin
        if "nblist_mult_size" in simulation_parameters:
            stnew["nblist_mult_size"] = simulation_parameters["nblist_mult_size"]
        if "nblist_add_neigh" in simulation_parameters:
            stnew["add_neigh"] = simulation_parameters["nblist_add_neigh"]
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
        print(model.summarize(example_data=conformation))

    return preproc_state, conformation


def initialize_system(conformation, vel, model, system_data, fprec):
    ## initial energy and forces
    print("# Computing initial energy and forces")
    e, f, vir,_ = model._energy_and_forces_and_virial(model.variables, conformation)
    model_energy_unit = model.Ha_to_model_energy
    f = np.array(f) / model_energy_unit
    epot = np.mean(e) / model_energy_unit
    vir = np.mean(vir, axis=0) / model_energy_unit
    
    if "nbeads" in system_data:
        ek = 0.5 * jnp.sum(system_data["mass"][:, None,None] * vel[0,:,:,None]*vel[0,:,None,:],axis=0)
    else:
        ek = 0.5 * jnp.sum(system_data["mass"][:, None,None] * vel[:,:,None]*vel[:,None,:],axis=0)

    ## build system
    system = {}
    system["ek_tensor"] = ek
    system["ek"] = jnp.trace(ek)
    system["epot"] = epot
    system["vel"] = vel.astype(fprec)
    if "cells" in conformation:
        system["virial"] = vir
        system["cell"] = conformation["cells"][0]
    if "nbeads" in system_data:
        nbeads = system_data["nbeads"]
        coordinates = conformation["coordinates"].reshape(nbeads, -1, 3)
        eigx = jnp.zeros_like(coordinates).at[0].set(coordinates[0])
        system["coordinates"] = eigx
        system["forces"] = jnp.einsum(
            "in,i...->n...", system_data["eigmat"], f.reshape(nbeads, -1, 3)
        ) * (1.0 / nbeads**0.5)
    else:
        system["coordinates"] = conformation["coordinates"]
        system["forces"] = f

    return system
