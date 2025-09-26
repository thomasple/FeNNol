from .spherical_harmonics import CG_SO3, generate_spherical_harmonics
from .atomic_units import AtomicUnits, UnitSystem
from typing import Dict, Any,Sequence, Union
import jax
import jax.numpy as jnp
import numpy as np
from ase.geometry.cell import cellpar_to_cell
import numba

def minmaxone(x, name=""):
    print(name, x.min(), x.max(), (x**2).mean() ** 0.5)

def minmaxone_jax(x, name=""):
    jax.debug.print(
        "{name}  {min}  {max}  {mean}",
        name=name,
        min=x.min(),
        max=x.max(),
        mean=(x**2).mean(),
    )

def cell_vectors_to_lengths_angles(cell):
    cell = cell.reshape(3, 3)
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])
    degree = 180.0 / np.pi
    alpha = np.arccos(np.dot(cell[1], cell[2]) / (b * c))
    beta = np.arccos(np.dot(cell[0], cell[2]) / (a * c))
    gamma = np.arccos(np.dot(cell[0], cell[1]) / (a * b))
    return np.array([a, b, c, alpha*degree, beta*degree, gamma*degree], dtype=cell.dtype)

def cell_lengths_angles_to_vectors(lengths_angles, ab_normal=(0, 0, 1), a_direction=None):
    return cellpar_to_cell(lengths_angles, ab_normal=ab_normal, a_direction=a_direction)

def parse_cell(cell):
    if cell is None:
        return None
    cell = np.asarray(cell, dtype=float).flatten()
    assert cell.size in [1, 3, 6, 9], "Cell must be of size 1, 3, 6 or 9"
    if cell.size == 9:
        return cell.reshape(3, 3)
    
    return cell_lengths_angles_to_vectors(cell)

def cell_is_triangular(cell, tol=1e-5):
    if cell is None:
        return False
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    return np.all(np.abs(cell - np.tril(cell)) < tol)

def tril_cell(cell,reciprocal_cell=None):
    if cell is None:
        return None
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    if reciprocal_cell is None:
        reciprocal_cell = np.linalg.inv(cell)
    length_angles = cell_vectors_to_lengths_angles(cell)
    cell_tril = cell_lengths_angles_to_vectors(length_angles)
    rotation = reciprocal_cell @ cell_tril
    return cell_tril, rotation


def mask_filter_1d(mask, max_size, *values_fill):
    cumsum = jnp.cumsum(mask,dtype=jnp.int32)
    scatter_idx = jnp.where(mask, cumsum - 1, max_size)
    outputs = []
    for value, fill in values_fill:
        shape = list(value.shape)
        shape[0] = max_size
        output = (
            jnp.full(shape, fill, dtype=value.dtype)
            .at[scatter_idx]
            .set(value, mode="drop")
        )
        outputs.append(output)
    if cumsum.size == 0:
        return outputs, scatter_idx, 0
    return outputs, scatter_idx, cumsum[-1]


def deep_update(
    mapping: Dict[Any, Any], *updating_mappings: Dict[Any, Any]
) -> Dict[Any, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


class Counter:
    def __init__(self, nseg, startsave=1):
        self.i = 0
        self.i_avg = 0
        self.nseg = nseg
        self.startsave = startsave

    @property
    def count(self):
        return self.i

    @property
    def count_avg(self):
        return self.i_avg

    @property
    def nsample(self):
        return max(self.count_avg - self.startsave + 1, 1)

    @property
    def is_reset_step(self):
        return self.count == 0

    def reset_avg(self):
        self.i_avg = 0

    def reset_all(self):
        self.i = 0
        self.i_avg = 0

    def increment(self):
        self.i = self.i + 1
        if self.i >= self.nseg:
            self.i = 0
            self.i_avg = self.i_avg + 1

### TOPLOGY DETECTION
@numba.njit
def _detect_bonds_pbc(radii,coordinates,cell):
    reciprocal_cell = np.linalg.inv(cell).T
    cell = cell.T
    nat = len(radii)
    bond1 = []
    bond2 = []
    distances = []
    for i in range(nat):
        for j in range(i + 1, nat):
            vec = coordinates[i] - coordinates[j]
            vecpbc = reciprocal_cell @ vec
            vecpbc -= np.round(vecpbc)
            vec = cell @ vecpbc
            dist = np.linalg.norm(vec)
            if dist < radii[i] + radii[j] + 0.4 and dist > 0.4:
                bond1.append(i)
                bond2.append(j)
                distances.append(dist)
    return bond1,bond2, distances

@numba.njit
def _detect_bonds(radii,coordinates):
    nat = len(radii)
    bond1 = []
    bond2 = []
    distances = []
    for i in range(nat):
        for j in range(i + 1, nat):
            vec = coordinates[i] - coordinates[j]
            dist = np.linalg.norm(vec)
            if dist < radii[i] + radii[j] + 0.4 and dist > 0.4:
                bond1.append(i)
                bond2.append(j)
                distances.append(dist)
    return bond1,bond2, distances

def detect_topology(species,coordinates, cell=None):
    """
    Detects the topology of a system based on species and coordinates.
    Returns a np.ndarray of shape [nbonds,2] containing the two indices for each bond.
    Inspired by OpenBabel's ConnectTheDots in mol.cpp
    """
    from .periodic_table import COV_RADII, UFF_MAX_COORDINATION
    radii = (COV_RADII* AtomicUnits.ANG)[species]
    max_coord = UFF_MAX_COORDINATION[species]

    if cell is not None:
        bond1,bond2,distances = _detect_bonds_pbc(radii, coordinates, cell)
    else:
        bond1,bond2,distances = _detect_bonds(radii, coordinates)

    bond1 = np.array(bond1, dtype=np.int32)
    bond2 = np.array(bond2, dtype=np.int32)
    bonds = np.stack((bond1, bond2), axis=1)
    
    coord = np.zeros(len(species), dtype=np.int32)
    np.add.at(coord, bonds[:, 0], 1)
    np.add.at(coord, bonds[:, 1], 1)

    if np.all(coord <= max_coord):
        return bonds

    distances = np.array(distances, dtype=np.float32)
    radiibonds = radii[bonds]
    req = radiibonds.sum(axis=1)
    rminbonds = radiibonds.min(axis=1)
    sorted_indices = np.lexsort((-distances/req, rminbonds))

    bonds = bonds[sorted_indices,:]
    distances = distances[sorted_indices]

    true_bonds = []
    for ibond in range(bonds.shape[0]):
        i,j = bonds[ibond]
        ci, cj = coord[i], coord[j]
        mci, mcj = max_coord[i], max_coord[j]
        if ci <= mci and cj <= mcj:
            true_bonds.append((i, j))
        else:
            coord[i] -= 1
            coord[j] -= 1

    true_bonds = np.array(true_bonds, dtype=np.int32)
    sorted_indices = np.lexsort((true_bonds[:, 1], true_bonds[:, 0]))
    true_bonds = true_bonds[sorted_indices, :]

    return true_bonds

def get_energy_gradient_function(
        energy_function,
        gradient_keys: Sequence[str],
        jit: bool = True,
    ):
        """Return a function that computes the energy and the gradient of the energy with respect to the keys in gradient_keys"""

        def energy_gradient(data):
            def _etot(inputs):
                if "strain" in inputs:
                    scaling = inputs["strain"]
                    batch_index = data["batch_index"]
                    coordinates = inputs["coordinates"] if "coordinates" in inputs else data["coordinates"]
                    coordinates = jax.vmap(jnp.matmul)(
                        coordinates, scaling[batch_index]
                    )
                    inputs = {**inputs, "coordinates": coordinates}
                    if "cells" in inputs or "cells" in data:
                        cells = inputs["cells"] if "cells" in inputs else data["cells"]
                        cells = jax.vmap(jnp.matmul)(cells, scaling)
                        inputs["cells"] = cells
                if "cells" in inputs:
                    reciprocal_cells = jnp.linalg.inv(inputs["cells"])
                    inputs = {**inputs, "reciprocal_cells": reciprocal_cells}
                energy, out = energy_function(inputs)
                return energy.sum(), out

            if "strain" in gradient_keys and "strain" not in data:
                data = {**data, "strain": jnp.array(np.eye(3)[None, :, :].repeat(data["natoms"].shape[0], axis=0))}
            inputs = {k: data[k] for k in gradient_keys}
            de, out = jax.grad(_etot, argnums=1, has_aux=True)(inputs)

            return (
                de,
                {**out, **{"dEd_" + k: de[k] for k in gradient_keys}},
            )

        if jit:
            return jax.jit(energy_gradient)
        else:
            return energy_gradient


def read_tinker_interval(indices_interval: Sequence[Union[int,str]]) -> np.ndarray:
    interval = [int(i) for i in indices_interval]
    indices = []
    while len(interval) > 0:
        i = interval.pop(0)
        if i > 0:
            indices.append(i)
        elif i < 0:
            start = -i
            end = interval.pop(0)
            assert end > start, "Syntax error in ligand indices. End index must be greater than start index."
            indices.extend(range(start, end + 1))
        else:
            raise ValueError("Syntax error in ligand indices. Indicing should be 1-based.")
    indices = np.unique(np.array(indices, dtype=np.int32))
    return indices - 1  # Convert to zero-based indexing