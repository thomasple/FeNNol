import numba
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
import math

from ase.neighborlist import neighbor_list as ase_neighbor_list
from ase import Atoms


@numba.njit
def compute_nblist_flatbatch(
    coords,
    cutoff,
    batch_index,
    natoms,
    mult_size,
    prev_nblist_size=0,
    padding_value=None,
):
    src, dst = [], []
    d12s = []
    c2 = cutoff**2
    shifts = np.cumsum(natoms)

    for i in range(coords.shape[0]):
        for j in range(i + 1, shifts[batch_index[i]]):
            vec = coords[j] - coords[i]
            d12 = np.sum(vec**2)
            if d12 < c2:
                src.append(i)
                dst.append(j)
                d12s.append(d12)
    nattot = shifts[-1]
    iedge = np.arange(len(src))
    isym = np.concatenate((iedge + iedge.shape[0], iedge))
    src, dst = np.array(src + dst, dtype=np.int64), np.array(dst + src, dtype=np.int64)
    d12s = np.array(d12s + d12s, dtype=np.float32)

    nblist_size = src.shape[0]
    if nblist_size > prev_nblist_size:
        prev_nblist_size = int(mult_size * nblist_size)

    if padding_value is None:
        padding_value = nattot
    src = np.append(
        src, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    dst = np.append(
        dst, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    d12s = np.append(
        d12s, c2 * np.ones(prev_nblist_size - nblist_size, dtype=np.float32)
    )
    isym = np.append(isym, -np.ones(prev_nblist_size - nblist_size, dtype=np.int64))
    return src, dst, d12s, prev_nblist_size, isym


def compute_nblist_ase(
    coords,
    cutoff,
    batch_index,
    natoms,
    mult_size,
    prev_nblist_size=0,
    padding_value=None,
):
    shifts = np.cumsum(natoms)
    atoms = Atoms(positions=coords[: shifts[0]])
    src, dst, d = ase_neighbor_list("ijd", atoms, cutoff=cutoff)
    d12s = d**2
    if len(shifts) > 1:
        src = [src]
        dst = [dst]
        d12s = [d12s]
        for i in range(1, len(shifts)):
            atoms = Atoms(positions=coords[shifts[i - 1] : shifts[i]])
            edge_src, edge_dst, d = ase_neighbor_list("ijd", atoms, cutoff=cutoff)
            src.append(edge_src + shifts[i - 1])
            dst.append(edge_dst + shifts[i - 1])
            d12s.append(d**2)
        src = np.concatenate(src, dtype=np.int64)
        dst = np.concatenate(dst, dtype=np.int64)
        d12s = np.concatenate(d12s, dtype=np.float32)

    sel = src < dst
    src = src[sel]
    dst = dst[sel]
    d12s = d12s[sel]
    iedge = np.arange(len(src))
    isym = np.concatenate((iedge + iedge.shape[0], iedge))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    d12s = np.concatenate((d12s, d12s))

    nblist_size = src.shape[0]
    if nblist_size > prev_nblist_size:
        prev_nblist_size = int(mult_size * nblist_size)

    if padding_value is None:
        padding_value = shifts[-1]
    src = np.append(
        src, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    dst = np.append(
        dst, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    d12s = np.append(
        d12s, cutoff**2 * np.ones(prev_nblist_size - nblist_size, dtype=np.float32)
    )
    isym = np.append(isym, -np.ones(prev_nblist_size - nblist_size, dtype=np.int64))
    return src, dst, d12s, prev_nblist_size, isym


def compute_nblist_ase_pbc(
    coords,
    cutoff,
    batch_index,
    natoms,
    mult_size,
    cells,
    reciprocal_cells,
    prev_nblist_size=0,
    padding_value=None,
):
    shifts = np.cumsum(natoms)
    atoms = Atoms(positions=coords[: shifts[0]], cell=cells[0], pbc=True)
    src, dst, d, pbc_shifts = ase_neighbor_list("ijdS", atoms, cutoff=cutoff)
    d12s = d**2
    if len(shifts) > 1:
        src = [src]
        dst = [dst]
        d12s = [d12s]
        pbc_shifts = [pbc_shifts]
        for i in range(1, len(shifts)):
            atoms = Atoms(
                positions=coords[shifts[i - 1] : shifts[i]], cell=cells[i], pbc=True
            )
            edge_src, edge_dst, d, pbc_shift = ase_neighbor_list(
                "ijdS", atoms, cutoff=cutoff
            )
            src.append(edge_src + shifts[i - 1])
            dst.append(edge_dst + shifts[i - 1])
            d12s.append(d**2)
            pbc_shifts.append(pbc_shift)
        src = np.concatenate(src, dtype=np.int64)
        dst = np.concatenate(dst, dtype=np.int64)
        d12s = np.concatenate(d12s, dtype=np.float32)
        pbc_shifts = np.concatenate(pbc_shifts, axis=0, dtype=np.float32)

    sel = src < dst
    src = src[sel]
    dst = dst[sel]
    d12s = d12s[sel]
    pbc_shifts = pbc_shifts[sel]
    iedge = np.arange(len(src))
    isym = np.concatenate((iedge + iedge.shape[0], iedge))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    d12s = np.concatenate((d12s, d12s))
    pbc_shifts = np.concatenate((pbc_shifts, -pbc_shifts), axis=0)

    nblist_size = src.shape[0]
    if nblist_size > prev_nblist_size:
        prev_nblist_size = int(mult_size * nblist_size)

    if padding_value is None:
        padding_value = shifts[-1]
    src = np.append(
        src, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    dst = np.append(
        dst, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    d12s = np.append(
        d12s, cutoff**2 * np.ones(prev_nblist_size - nblist_size, dtype=np.float32)
    )
    pbc_shifts = np.append(
        pbc_shifts,
        np.zeros((prev_nblist_size - nblist_size, 3), dtype=np.float32),
        axis=0,
    )
    isym = np.append(isym, -np.ones(prev_nblist_size - nblist_size, dtype=np.int64))
    return src, dst, d12s, pbc_shifts, prev_nblist_size, isym


@numba.njit
def compute_nblist_flatbatch_minimage(
    coords,
    cutoff,
    batch_index,
    natoms,
    mult_size,
    cells,
    reciprocal_cells,
    prev_nblist_size=0,
    padding_value=None,
):
    src, dst = [], []
    pbc_shifts = []
    d12s = []
    c2 = cutoff**2
    shifts = np.cumsum(natoms)

    for i in range(coords.shape[0]):
        cell = cells[batch_index[i]]
        reciprocal_cell = reciprocal_cells[batch_index[i]]
        for j in range(i + 1, shifts[batch_index[i]]):
            vec = coords[j] - coords[i]
            vecpbc = np.dot(reciprocal_cell, vec)
            shift = -np.round(vecpbc)
            vecpbc = np.dot(cell, vecpbc + shift)
            d12 = np.sum(vecpbc**2)
            if d12 < c2:
                src.append(i)
                dst.append(j)
                d12s.append(d12)
                pbc_shifts.append(list(shift))
    nattot = shifts[-1]
    iedge = np.arange(len(src))
    isym = np.concatenate((iedge + iedge.shape[0], iedge))
    src, dst = np.array(src + dst, dtype=np.int64), np.array(dst + src, dtype=np.int64)
    d12s = np.array(d12s + d12s, dtype=np.float32)
    pbc_shifts = np.array(pbc_shifts, dtype=np.float32)
    pbc_shifts = np.concatenate((pbc_shifts, -pbc_shifts), axis=0)

    nblist_size = src.shape[0]
    if nblist_size > prev_nblist_size:
        prev_nblist_size = int(mult_size * nblist_size)

    if padding_value is None:
        padding_value = nattot
    src = np.append(
        src, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    dst = np.append(
        dst, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    d12s = np.append(
        d12s, c2 * np.ones(prev_nblist_size - nblist_size, dtype=np.float32)
    )
    pbc_shifts = np.append(
        pbc_shifts,
        np.zeros((prev_nblist_size - nblist_size, 3), dtype=np.float32),
        axis=0,
    )
    isym = np.append(isym, -np.ones(prev_nblist_size - nblist_size, dtype=np.int64))
    return src, dst, d12s, pbc_shifts, prev_nblist_size, isym


@numba.njit
def compute_nblist_flatbatch_fullpbc(
    coords,
    cutoff,
    batch_index,
    natoms,
    mult_size,
    cells,
    reciprocal_cells,
    prev_nblist_size=0,
    padding_value=None,
):
    src, dst = [], []
    pbc_shifts = []
    d12s = []
    c2 = cutoff**2
    shifts = np.cumsum(natoms)

    coordspbc = np.empty_like(coords)
    at_shifts = np.empty_like(coords)
    for i in range(coords.shape[0]):
        ii = batch_index[i]
        cell = cells[ii]
        reciprocal_cell = reciprocal_cells[ii]
        qi = np.dot(reciprocal_cell, coords[i])
        shifti = -np.floor(qi)
        coordspbc[i, :] = np.dot(cell, qi + shifti)
        at_shifts[i, :] = shifti
        # coordspbc.append(coordsi)
        # at_shifts.append(shifti)
    # coordspbc = np.array(coordspbc,dtype=np.float32)
    # at_shifts = np.array(at_shifts,dtype=np.float32)

    inv_distances = (np.sum(reciprocal_cells**2, axis=1)) ** 0.5
    num_repeats_all = np.ceil(cutoff * inv_distances).astype(np.int32)

    for i in range(coords.shape[0]):
        ii = batch_index[i]
        cell = cells[ii]
        num_repeats = num_repeats_all[ii]
        for ix in range(-num_repeats[0], num_repeats[0] + 1):
            for iy in range(-num_repeats[1], num_repeats[1] + 1):
                for iz in range(-num_repeats[2], num_repeats[2] + 1):
                    shift = ix * cell[:, 0] + iy * cell[:, 1] + iz * cell[:, 2]
                    for j in range(i + 1, shifts[ii]):
                        vec = coordspbc[j] - coordspbc[i] + shift
                        d12 = np.sum(vec**2)
                        if d12 < c2:
                            src.append(i)
                            dst.append(j)
                            d12s.append(d12)
                            pbc_shifts.append([ix, iy, iz])
    nattot = shifts[-1]
    pbc_shifts = (
        np.array(pbc_shifts, dtype=np.float32)
        + at_shifts[np.array(dst, dtype=np.int64)]
        - at_shifts[np.array(src, dtype=np.int64)]
    )
    iedge = np.arange(len(src))
    isym = np.concatenate((iedge + iedge.shape[0], iedge))
    src, dst = np.array(src + dst, dtype=np.int64), np.array(dst + src, dtype=np.int64)
    d12s = np.array(d12s + d12s, dtype=np.float32)
    pbc_shifts = np.concatenate((pbc_shifts, -pbc_shifts), axis=0)

    nblist_size = src.shape[0]
    if nblist_size > prev_nblist_size:
        prev_nblist_size = int(mult_size * nblist_size)

    if padding_value is None:
        padding_value = nattot
    src = np.append(
        src, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    dst = np.append(
        dst, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    d12s = np.append(
        d12s, c2 * np.ones(prev_nblist_size - nblist_size, dtype=np.float32)
    )
    pbc_shifts = np.append(
        pbc_shifts,
        np.zeros((prev_nblist_size - nblist_size, 3), dtype=np.float32),
        axis=0,
    )
    isym = np.append(isym, -np.ones(prev_nblist_size - nblist_size, dtype=np.int64))
    return src, dst, d12s, pbc_shifts, prev_nblist_size, isym


def hash_cell(cell_coords, batch_index, nel):
    return (
        15823 * cell_coords[..., 0]
        + 9737333 * cell_coords[..., 1]
        + 95483 * cell_coords[..., 2]
        + 79411 * batch_index
    ) % nel


def compute_cell_list(
    coords,
    cutoff,
    batch_index,
    natoms,
    mult_size,
    prev_nblist_size=0,
    padding_value=None,
):
    """NOT FUNCTIONAL YET !!!"""
    cell_coords = np.floor(coords / cutoff).astype(int)
    # hash cell ids
    nattot = coords.shape[0]
    nel = max(1000, nattot)
    spatial_lookup = hash_cell(cell_coords, batch_index, nel)
    grid_count = np.zeros(nel, dtype=int)
    np.add.at(grid_count, spatial_lookup, 1)
    max_count = np.max(grid_count)

    idx = np.argsort(spatial_lookup)
    spatial_lookup = spatial_lookup[idx]
    iat = np.arange(nattot)[idx]

    is_start = np.concatenate([[True], spatial_lookup[1:] != spatial_lookup[:-1]]) * (
        np.arange(nattot) + 1
    )
    istart = np.zeros(nel, dtype=int)
    np.add.at(istart, spatial_lookup, is_start)
    istart -= 1

    idx_start = istart[spatial_lookup]
    shift = np.arange(nattot) - idx_start
    indices = -np.ones((nel, max_count), dtype=int)
    indices[spatial_lookup, shift] = iat

    cell_offsets = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, -1],
            [-1, 1, 0],
            [-1, 1, 1],
            [0, -1, -1],
            [0, -1, 0],
            [0, -1, 1],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    batch_index_neigh = np.repeat(batch_index[:, None], cell_offsets.shape[0], axis=-1)
    neigh_cells = hash_cell(
        cell_coords[:, None, :] + cell_offsets[None, :, :], batch_index_neigh, nel
    )
    neighbors = indices[neigh_cells].reshape(nattot, -1)
    neighbors = np.unique(neighbors, axis=1)
    edge_src = np.repeat(np.arange(nattot), neighbors.shape[1])
    neighbors = neighbors.flatten()
    mask = np.logical_and(
        neighbors >= 0,
        edge_src < neighbors,
        batch_index[edge_src] == batch_index[neighbors],
    )
    edge_src, edge_dst = edge_src[mask], neighbors[mask]

    d12s = np.sum((coords[edge_dst] - coords[edge_src]) ** 2, axis=-1)
    mask = d12s < cutoff**2
    edge_src, edge_dst, d12s = edge_src[mask], edge_dst[mask], d12s[mask]

    src, dst = np.concatenate([edge_src, edge_dst]), np.concatenate(
        [edge_dst, edge_src]
    )
    d12s = np.concatenate([d12s, d12s])

    nblist_size = src.shape[0]
    if nblist_size > prev_nblist_size:
        prev_nblist_size = int(mult_size * nblist_size)

    if padding_value is None:
        padding_value = nattot
    src = np.append(
        src, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    dst = np.append(
        dst, padding_value * np.ones(prev_nblist_size - nblist_size, dtype=np.int64)
    )
    d12s = np.append(
        d12s, cutoff**2 * np.ones(prev_nblist_size - nblist_size, dtype=np.float32)
    )
    return src, dst, d12s, prev_nblist_size


@numba.njit
def angular_nblist(edge_src, natoms):
    idx = np.argsort(edge_src)

    counts = np.zeros(natoms, dtype=np.int64)
    for i in edge_src:
        counts[i] += 1

    pair_sizes = (counts * (counts - 1)) // 2
    nangles = np.sum(pair_sizes)
    max_neigh = np.max(counts)

    shift = 0
    p1s = np.zeros(nangles, dtype=np.int64)
    p2s = np.zeros(nangles, dtype=np.int64)
    central_atom_index = np.zeros(nangles, dtype=np.int64)
    iang = 0
    for i, c in enumerate(counts):
        if c >= 2:
            for j in range(c - 1):
                for k in range(j + 1, c):
                    # print(iang,i,j,k,shift)
                    p1s[iang] = k + shift
                    p2s[iang] = j + shift
                    central_atom_index[iang] = i
                    iang += 1
        shift += c
    # return central_atom_index, p1s, p2s
    angle_src = idx[p1s]
    angle_dst = idx[p2s]

    return central_atom_index, angle_src, angle_dst, max_neigh


@partial(jax.jit, static_argnums=(1, 4, 5))
def compute_cell_list_fixed(
    coords, cutoff, batch_index, natoms, max_occupancy, max_pairs, padding_value
):
    """NOT FUNCTIONAL YET !!!"""
    cell_coords = jnp.floor(coords / cutoff).astype(int)
    # hash cell ids
    nattot = coords.shape[0]
    nel = max(1000, nattot)
    spatial_lookup = hash_cell(cell_coords, batch_index, nel)
    grid_count = jnp.zeros(nel, dtype=int).at[spatial_lookup].add(1)
    max_count = jnp.max(grid_count)

    idx = jnp.argsort(spatial_lookup)
    spatial_lookup = spatial_lookup[idx]
    iat = jnp.arange(nattot)[idx]

    is_start = jnp.concatenate(
        (jnp.asarray([True]), spatial_lookup[1:] != spatial_lookup[:-1])
    ) * (jnp.arange(nattot) + 1)
    istart = jnp.zeros(nel, dtype=int).at[spatial_lookup].add(is_start) - 1

    idx_start = istart[spatial_lookup]
    shift = jnp.arange(nattot) - idx_start
    indices = (
        (-jnp.ones((nel, max_occupancy), dtype=int)).at[spatial_lookup, shift].set(iat)
    )
    # indices[spatial_lookup,shift] = iat

    cell_offsets = jnp.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, -1],
            [-1, 1, 0],
            [-1, 1, 1],
            [0, -1, -1],
            [0, -1, 0],
            [0, -1, 1],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    batch_index_neigh = jnp.repeat(batch_index[:, None], cell_offsets.shape[0], axis=-1)
    neigh_cells = hash_cell(
        cell_coords[:, None, :] + cell_offsets[None, :, :], batch_index_neigh, nel
    )
    neighbors = indices[neigh_cells].reshape(nattot, -1)
    src = jnp.asarray(np.repeat(np.arange(nattot), neighbors.shape[1]))
    neighbors = neighbors.flatten()

    d12 = jnp.sum((coords[neighbors] - coords[src]) ** 2, axis=-1)
    mask = (
        (d12 < cutoff**2)
        * (neighbors >= 0)
        * (batch_index[src] == batch_index[neighbors])
    )
    npairs = jnp.sum(mask)
    d12 = d12 * mask + (cutoff**2) * (1 - mask)
    src = src * mask + padding_value * (1 - mask)
    neighbors = neighbors * mask + padding_value * (1 - mask)
    idx = jnp.argsort(d12)[:max_pairs]

    return src[idx], neighbors[idx], d12[idx], max_count, npairs


@partial(jax.jit, static_argnums=(4, 5))
def compute_nblist_fixed(
    coords, cutoff, batch_index, natoms, max_nat, max_pairs, padding_value
):
    p1, p2 = jnp.triu_indices(max_nat, 1)
    if natoms.shape[0] == 1:
        shift = jnp.array([0])
        mask_p12 = None
    else:
        mask_p12 = (
            (p1[None, :] < natoms[:, None]) * (p2[None, :] < natoms[:, None])
        ).flatten()
        shift = jnp.concatenate((jnp.array([0]), jnp.cumsum(natoms[:-1])))
        p1 = jnp.where(mask_p12, (p1[None, :] + shift[:, None]).flatten(), -1)
        p2 = jnp.where(mask_p12, (p2[None, :] + shift[:, None]).flatten(), -1)

    d12 = jnp.sum((coords[p2] - coords[p1]) ** 2, axis=-1)

    if natoms.shape[0] == 1:
        mask = d12 < cutoff**2
    else:
        mask = mask_p12 * (d12 < cutoff**2)
    d12 = jnp.where(mask, d12, cutoff**2)
    edge_src = jnp.where(mask, p1, padding_value)
    edge_dst = jnp.where(mask, p2, padding_value)
    npairs = jnp.sum(mask)
    idx = jnp.argsort(d12)  # [:max_pairs]

    return edge_src[idx], edge_dst[idx], d12[idx], npairs, (p1, p2, mask_p12)


@partial(jax.jit, static_argnums=(4, 5))
def compute_nblist_fixed_minimage(
    coords,
    cutoff,
    batch_index,
    natoms,
    max_nat,
    max_pairs,
    padding_value,
    cells,
    reciprocal_cells,
):
    p1, p2 = jnp.triu_indices(max_nat, 1)

    if natoms.shape[0] == 1:
        shift = jnp.array([0])
        mask_p12 = None
    else:
        mask_p12 = (
            (p1[None, :] < natoms[:, None]) * (p2[None, :] < natoms[:, None])
        ).flatten()
        shift = jnp.concatenate((jnp.array([0]), jnp.cumsum(natoms[:-1])))

        p1 = jnp.where(mask_p12, (p1[None, :] + shift[:, None]).flatten(), -1)
        p2 = jnp.where(mask_p12, (p2[None, :] + shift[:, None]).flatten(), -1)

    batch_indexvec = batch_index[p1]
    vec = coords[p2] - coords[p1]
    vecpbc = jnp.einsum("sij,sj->si", reciprocal_cells[batch_indexvec], vec)
    pbc_shifts = -jnp.round(vecpbc)
    vecpbc = jnp.einsum(
        "sij,sj->si", cells[batch_indexvec], vecpbc + pbc_shifts
    )  # jnp.dot(cells[batch_indexvec],vecpbc.T).T
    d12 = jnp.sum(vecpbc**2, axis=-1)

    if natoms.shape[0] == 1:
        mask = d12 < cutoff**2
    else:
        mask = mask_p12 * (d12 < cutoff**2)
    d12 = jnp.where(mask, d12, cutoff**2)
    edge_src = jnp.where(mask, p1, padding_value)
    edge_dst = jnp.where(mask, p2, padding_value)
    pbc_shifts = jnp.where(mask[:, None], pbc_shifts, 0)
    npairs = jnp.sum(mask)
    idx = jnp.argsort(d12)  # [:max_pairs]

    return (
        edge_src[idx],
        edge_dst[idx],
        d12[idx],
        pbc_shifts[idx],
        npairs,
        (p1, p2, mask_p12),
    )


@partial(jax.jit, static_argnums=(2, 3))
def to_dense_nblist(edge_src, edge_dst, nat, max_neigh):
    Nedge = len(edge_src)

    counts = jnp.zeros(nat, dtype=jnp.int32).at[edge_src].add(1, mode="drop")
    max_count = jnp.max(counts)
    offset = jnp.tile(jnp.arange(max_neigh), nat)[:Nedge]
    offset = jnp.where(edge_src < nat, offset, 0)
    indices = edge_src * max_neigh + offset
    dense_idx = (
        jnp.full((nat) * max_neigh, nat, dtype=jnp.int32)
        .at[indices]
        .set(edge_dst)
        .reshape(nat, max_neigh)
    )[:-1]
    edge_idx = (
        jnp.full((nat) * max_neigh, nat, dtype=jnp.int32)
        .at[indices]
        .set(jnp.arange(Nedge))
        .reshape(nat, max_neigh)
    )

    return dense_idx, counts, max_count


def to_sparse_nblist(neigh_index, d12, nblist_size):
    edge_src = jnp.asarray(
        np.broadcast_to(
            np.arange(neigh_index.shape[0], dtype=np.int32)[:, None],
            neigh_index.shape,
        ).flatten()
    )
    edge_dst = neigh_index.flatten()
    idx = jnp.argsort(d12.flatten())[:nblist_size]
    # mask = mask.flatten()
    # cumsum = jnp.cumsum(mask)
    # id_edge = jnp.arange(edge_dst.shape[0])
    # idx = (
    #     jnp.full(nblist_size, -1, dtype=jnp.int32)
    #     .at[cumsum - 1]
    #     .add(jnp.where(mask, id_edge + 1, 0))
    # )
    edge_src = edge_src[idx]
    edge_dst = edge_dst[idx]

    return edge_src, edge_dst, idx


@partial(jax.jit, static_argnums=(4, 5))
def build_dense_nblist(
    coords, natoms, batch_index, cutoff, max_neigh, max_nat, mask_atoms=None
):
    if natoms.shape[0] == 1:
        p12 = jnp.broadcast_to(jnp.arange(max_nat)[None, :], (coords.shape[0], max_nat))
        mask_p12 = p12 != jnp.arange(coords.shape[0])[:, None]
    else:
        shift = jnp.concatenate((jnp.array([0]), jnp.cumsum(natoms[:-1])))
        p12 = jnp.arange(max_nat)[None, :] + shift[batch_index, None]
        mask_p12 = (jnp.arange(max_nat)[None, :] < natoms[batch_index, None]) & (
            p12 != jnp.arange(coords.shape[0])[:, None]
        )
    if mask_atoms is not None:
        mask_p12 = mask_p12 & mask_atoms[:, None] & mask_atoms[p12]
    d12 = ((coords[:, None, :] - coords[p12]) ** 2).sum(axis=-1)
    d12 = jnp.where(mask_p12, d12, cutoff**2)

    mask = d12 < cutoff**2
    count = mask.sum(axis=-1)
    max_count = jnp.max(count)

    idx = jnp.argsort(d12, axis=-1)[:, :max_neigh]
    d12 = jnp.take_along_axis(d12, idx, axis=-1)
    mask = jnp.take_along_axis(mask, idx, axis=-1)
    neigh_index = jnp.where(
        mask, jnp.take_along_axis(p12, idx, axis=-1), coords.shape[0]
    )

    return neigh_index, d12, count, max_count, (p12, mask_p12)


@partial(jax.jit, static_argnums=(6, 7))
def build_dense_nblist_minimage(
    coords,
    natoms,
    batch_index,
    cells,
    reciprocal_cells,
    cutoff,
    max_neigh,
    max_nat,
    mask_atoms=None,
):
    if natoms.shape[0] == 1:
        p12 = jnp.broadcast_to(jnp.arange(max_nat)[None, :], (coords.shape[0], max_nat))
        mask_p12 = p12 != jnp.arange(coords.shape[0])[:, None]

        vec = coords[p12] - coords[:, None, :]
        rcell = reciprocal_cells[0].T
        vecpbc = jnp.dot(vec, rcell)
        # vecpbc = jnp.einsum("ij,abj->abi",reciprocal_cells[0],vec)
        pbc_shifts = -jnp.round(vecpbc)
        cell = cells[0].T
        # vec = jnp.einsum("ij,abj->abi",cells[0],vecpbc+pbc_shifts)
        vec = jnp.dot(vecpbc + pbc_shifts, cell)
    else:
        shift = jnp.concatenate((jnp.array([0]), jnp.cumsum(natoms[:-1])))
        p12 = jnp.arange(max_nat)[None, :] + shift[batch_index, None]
        mask_p12 = (jnp.arange(max_nat)[None, :] < natoms[batch_index, None]) & (
            p12 != jnp.arange(coords.shape[0])[:, None]
        )

        vec = coords[p12] - coords[:, None, :]

        def compute_pbc(vec, reciprocal_cell, cell):
            vecpbc = jnp.dot(vec, reciprocal_cell.T)
            pbc_shifts = -jnp.round(vecpbc)
            return jnp.dot(vecpbc + pbc_shifts, cell.T), pbc_shifts

        vec, pbc_shifts = jax.vmap(compute_pbc)(
            vec, reciprocal_cells[batch_index], cells[batch_index]
        )
        # vecpbc = jnp.einsum("aij,abj->abi",reciprocal_cells[batch_index],vec)
        # pbc_shifts = -jnp.round(vecpbc)
        # vec = jnp.einsum("aij,abj->abi",cells[batch_index],vecpbc+pbc_shifts)

    if mask_atoms is not None:
        mask_p12 = mask_p12 & mask_atoms[:, None] & mask_atoms[p12]

    d12 = (vec**2).sum(axis=-1)
    d12 = jnp.where(mask_p12, d12, cutoff**2)

    mask = d12 < cutoff**2
    count = mask.sum(axis=-1)
    max_count = jnp.max(count)

    idx = jnp.argsort(d12, axis=-1)[:, :max_neigh]
    d12 = jnp.take_along_axis(d12, idx, axis=-1)
    mask = jnp.take_along_axis(mask, idx, axis=-1)
    neigh_index = jnp.where(
        mask, jnp.take_along_axis(p12, idx, axis=-1), coords.shape[0]
    )
    pbc_shifts = jnp.where(
        mask[:, :, None], jnp.take_along_axis(pbc_shifts, idx[:, :, None], axis=1), 0
    )

    return neigh_index, d12, pbc_shifts, count, max_count, (p12, mask_p12)


@numba.njit
def build_dense_nblist_np(
    coords, natoms, batch_index, cutoff, max_neigh, max_nat, mask_atoms
):

    c2 = cutoff**2
    shifts = np.cumsum(natoms)
    neigh_index = np.full((coords.shape[0], max_neigh), coords.shape[0], dtype=np.int32)
    d12 = np.full((coords.shape[0], max_neigh), c2, dtype=coords.dtype)
    count = np.zeros(coords.shape[0], dtype=np.int32)

    for i in range(coords.shape[0]):
        if not mask_atoms[i]:
            continue
        for j in range(i + 1, shifts[batch_index[i]]):
            if not mask_atoms[j]:
                continue
            vec = coords[j] - coords[i]
            dij = np.sum(vec**2)
            if dij < c2:
                if count[i] < max_neigh:
                    d12[i, count[i]] = dij
                    neigh_index[i, count[i]] = j
                if count[j] < max_neigh:
                    d12[j, count[j]] = dij
                    neigh_index[j, count[j]] = i
                count[i] += 1
                count[j] += 1

    max_count = np.max(count)

    return neigh_index, d12, count, max_count, None


def build_dense_nblist_minimage_np(
    coords,
    natoms,
    batch_index,
    cells,
    reciprocal_cells,
    cutoff,
    max_neigh,
    max_nat,
    mask_atoms=None,
):
    if natoms.shape[0] == 1:
        p12 = np.broadcast_to(np.arange(max_nat)[None, :], (coords.shape[0], max_nat))
        mask_p12 = p12 != np.arange(coords.shape[0])[:, None]

        vec = coords[p12] - coords[:, None, :]
        rcell = reciprocal_cells[0].T
        vecpbc = np.dot(vec, rcell)
        # vecpbc = np.einsum("ij,abj->abi",reciprocal_cells[0],vec)
        pbc_shifts = -np.round(vecpbc)
        cell = cells[0].T
        # vec = np.einsum("ij,abj->abi",cells[0],vecpbc+pbc_shifts)
        vec = np.dot(vecpbc + pbc_shifts, cell)
    else:
        shift = np.concatenate((np.array([0]), np.cumsum(natoms[:-1])))
        p12 = np.arange(max_nat)[None, :] + shift[batch_index, None]
        mask_p12 = (np.arange(max_nat)[None, :] < natoms[batch_index, None]) & (
            p12 != np.arange(coords.shape[0])[:, None]
        )

        vec = coords[p12] - coords[:, None, :]

        def compute_pbc(vec, reciprocal_cell, cell):
            vecpbc = np.dot(vec, reciprocal_cell.T)
            pbc_shifts = -np.round(vecpbc)
            return np.dot(vecpbc + pbc_shifts, cell.T), pbc_shifts

        vec, pbc_shifts = jax.vmap(compute_pbc)(
            vec, reciprocal_cells[batch_index], cells[batch_index]
        )
        # vecpbc = np.einsum("aij,abj->abi",reciprocal_cells[batch_index],vec)
        # pbc_shifts = -np.round(vecpbc)
        # vec = np.einsum("aij,abj->abi",cells[batch_index],vecpbc+pbc_shifts)

    if mask_atoms is not None:
        mask_p12 = mask_p12 & mask_atoms[:, None] & mask_atoms[p12]

    d12 = (vec**2).sum(axis=-1)
    d12 = np.where(mask_p12, d12, cutoff**2)

    mask = d12 < cutoff**2
    count = mask.sum(axis=-1)
    max_count = np.max(count)

    idx = np.argsort(d12, axis=-1)[:, :max_neigh]
    d12 = np.take_along_axis(d12, idx, axis=-1)
    mask = np.take_along_axis(mask, idx, axis=-1)
    neigh_index = np.where(mask, np.take_along_axis(p12, idx, axis=-1), coords.shape[0])
    pbc_shifts = np.where(
        mask[:, :, None], np.take_along_axis(pbc_shifts, idx[:, :, None], axis=1), 0
    )

    return neigh_index, d12, pbc_shifts, count, max_count, (p12, mask_p12)


# @partial(jax.jit, static_argnums=(1,4,5))
# def compute_nblist_fixed_fullpbc(coords,cutoff,batch_index,natoms,max_nat,max_pairs,padding_value,cells,reciprocal_cells):
#     p1,p2=jnp.triu_indices(max_nat, 1)
#     mask = ((p1[None,:]<natoms[:,None])*(p2[None,:]<natoms[:,None])).flatten()

#     if natoms.shape[0]==1:
#         shift=jnp.array([0])
#     else:
#         shift=jnp.concatenate((jnp.array([0]),jnp.cumsum(natoms[:-1])))

#     qi = jnp.einsum("sij,sj->si",reciprocal_cells[batch_index],coords)
#     at_shifts = -jnp.floor(qi)
#     coordspbc = jnp.einsum("sij,sj->si",cells[batch_index],qi+at_shifts)


#     p1 = (p1[None,:]+shift[:,None]).flatten()*mask - (1-mask)
#     p2 = (p2[None,:]+shift[:,None]).flatten()*mask - (1-mask)
#     batch_indexvec=batch_index[p1]
#     vec = coords[p2]-coords[p1]
#     vecpbc = jnp.einsum("sij,sj->si",reciprocal_cells[batch_indexvec],vec)
#     pbc_shifts = -jnp.round(vecpbc)
#     vecpbc = jnp.einsum("sij,sj->si",cells[batch_indexvec],vecpbc+pbc_shifts) #jnp.dot(cells[batch_indexvec],vecpbc.T).T
#     d12=jnp.sum(vecpbc**2,axis=-1)

#     mask = mask*(d12<cutoff**2)
#     d12 = d12*mask + (cutoff**2)*(1-mask)
#     p1 = p1*mask + padding_value*(1-mask)
#     p2 = p2*mask + padding_value*(1-mask)
#     npairs=jnp.sum(mask)
#     idx = jnp.argsort(d12)[:max_pairs]

#     edge_src,edge_dst = p1[idx],p2[idx]
#     return edge_src,edge_dst,d12[idx],pbc_shifts[idx],npairs


def get_reciprocal_space_parameters(reciprocal_cells, cutoff, kmax=30, kthr=1.0e-6):
    # find optimal ewald parameters (preprocessing)
    eps = 1.0e-8
    ratio = eps + 1
    x = 0.5
    i = 0
    # approximate value
    while ratio > eps:
        x *= 2
        ratio = math.erfc(x * cutoff) / cutoff
    # refine with binary search
    k = i + 60
    xlo = 0.0
    xhi = x
    for i in range(1, k + 1):
        x = (xlo + xhi) / 2.0
        ratio = math.erfc(x * cutoff) / cutoff
        if ratio > eps:
            xlo = x
        else:
            xhi = x
    bewald = x

    # set k points
    kxs = np.arange(kmax + 1)
    kxs = np.concatenate((kxs, -kxs[1:]))
    k = np.array(np.meshgrid(kxs, kxs, kxs)).reshape(3, -1).T[1:]
    # set exp factor
    m2s = []
    expfacs = []
    ks = []
    nks = []
    for i, A in enumerate(range(reciprocal_cells.shape[0])):
        A = reciprocal_cells[i]
        m2 = np.sum(
            (
                k[:, 0, None] * A[None, 0, :]
                + k[:, 1, None] * A[None, 1, :]
                + k[:, 2, None] * A[None, 2, :]
            )
            ** 2,
            axis=-1,
        )
        a2 = (np.pi / bewald) ** 2
        expfac = np.exp(-a2 * m2) / m2
        isort = np.argsort(expfac)[::-1]
        expfac = expfac[isort]
        m2 = m2[isort]
        ki = k[isort]
        sel = (expfac > kthr).nonzero()[0]
        nks.append(len(sel))
        m2s.append(m2)
        expfacs.append(expfac)
        ks.append(ki)

    ks = np.stack(ks, axis=0)
    m2s = np.stack(m2s, axis=0)
    expfacs = np.stack(expfacs, axis=0)
    nks = np.array(nks, dtype=np.int64)
    nk = np.max(nks)
    return ks[:, :nk, :], expfacs[:, :nk], m2s[:, :nk], bewald
