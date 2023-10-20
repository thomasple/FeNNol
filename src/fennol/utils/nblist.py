import numba
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp

@numba.njit
def compute_nblist_flatbatch(
    coords, cutoff, isys, natoms, mult_size, prev_nblist_size=0,padding_value=None
):
    src, dst = [], []
    d12s = []
    c2 = cutoff**2
    shifts = np.cumsum(natoms)

    for i in range(coords.shape[0]):
        for j in range(i + 1, shifts[isys[i]]):
            vec = coords[i] - coords[j]
            d12 = np.sum(vec**2)
            if d12 < c2:
                src.append(i)
                dst.append(j)
                d12s.append(d12)
    nattot = shifts[-1]
    src, dst = np.array(src + dst,dtype=np.int64), np.array(dst + src,dtype=np.int64)
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
    return src, dst, d12s, prev_nblist_size

def hash_cell(cell_coords,isys,nel):
    return (15823*cell_coords[...,0]+9737333*cell_coords[...,1]+95483*cell_coords[...,2] + 79411*isys)%nel

def compute_cell_list(coords,cutoff, isys, natoms, mult_size, prev_nblist_size=0,padding_value=None):
    """ NOT FUNCTIONAL YET !!! """
    cell_coords = np.floor(coords/cutoff).astype(int)
    # hash cell ids
    nattot = coords.shape[0]
    nel = max(1000,nattot)
    spatial_lookup = hash_cell(cell_coords,isys,nel)
    grid_count = np.zeros(nel,dtype=int)
    np.add.at(grid_count,spatial_lookup,1)
    max_count = np.max(grid_count)

    idx=np.argsort(spatial_lookup)
    spatial_lookup = spatial_lookup[idx]
    iat = np.arange(nattot)[idx]

    is_start = np.concatenate([[True],spatial_lookup[1:]!=spatial_lookup[:-1]])*(np.arange(nattot)+1)
    istart = np.zeros(nel,dtype=int)
    np.add.at(istart,spatial_lookup,is_start)
    istart -= 1

    idx_start = istart[spatial_lookup]
    shift = np.arange(nattot)-idx_start
    indices = -np.ones((nel,max_count),dtype=int)
    indices[spatial_lookup,shift] = iat

    cell_offsets = np.array([[-1,-1,-1],[-1,-1,0],[-1,-1,1],
                    [-1,0,-1],[-1,0,0],[-1,0,1],
                    [-1,1,-1],[-1,1,0],[-1,1,1],
                    [0,-1,-1],[0,-1,0],[0,-1,1],
                    [0,0,-1],[0,0,0],[0,0,1],
                    [0,1,-1],[0,1,0],[0,1,1],
                    [1,-1,-1],[1,-1,0],[1,-1,1],
                    [1,0,-1],[1,0,0],[1,0,1],
                    [1,1,-1],[1,1,0],[1,1,1]])
    isysneigh=np.repeat(isys[:,None],cell_offsets.shape[0],axis=-1)
    neigh_cells = hash_cell(cell_coords[:,None,:]+cell_offsets[None,:,:],isysneigh,nel)
    neighbors = indices[neigh_cells].reshape(nattot,-1)
    neighbors = np.unique(neighbors,axis=1)
    edge_src = np.repeat(np.arange(nattot),neighbors.shape[1])
    neighbors = neighbors.flatten()
    mask = np.logical_and(neighbors>=0,edge_src<neighbors,isys[edge_src]==isys[neighbors])
    edge_src,edge_dst = edge_src[mask],neighbors[mask]

    d12s = np.sum((coords[edge_src]-coords[edge_dst])**2,axis=-1)
    mask = d12s<cutoff**2
    edge_src,edge_dst,d12s = edge_src[mask],edge_dst[mask],d12s[mask]

    src, dst = np.concatenate([edge_src,edge_dst]),np.concatenate([edge_dst,edge_src])
    d12s = np.concatenate([d12s,d12s])
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

    return central_atom_index, angle_src, angle_dst

@partial(jax.jit, static_argnums=(1,4,5))
def compute_cell_list_fixed(coords,cutoff,isys,natoms,max_occupancy,max_pairs,padding_value):
    """ NOT FUNCTIONAL YET !!! """
    cell_coords = jnp.floor(coords/cutoff).astype(int)
    # hash cell ids
    nattot = coords.shape[0]
    nel = max(1000,nattot)
    spatial_lookup = hash_cell(cell_coords,isys,nel)
    grid_count = jnp.zeros(nel,dtype=int).at[spatial_lookup].add(1)
    max_count = jnp.max(grid_count)

    idx=jnp.argsort(spatial_lookup)
    spatial_lookup = spatial_lookup[idx]
    iat = jnp.arange(nattot)[idx]

    is_start = jnp.concatenate((jnp.asarray([True]),spatial_lookup[1:]!=spatial_lookup[:-1]))*(jnp.arange(nattot)+1)
    istart = jnp.zeros(nel,dtype=int).at[spatial_lookup].add(is_start) - 1

    idx_start = istart[spatial_lookup]
    shift = jnp.arange(nattot)-idx_start
    indices = (-jnp.ones((nel,max_occupancy),dtype=int)).at[spatial_lookup,shift].set(iat)
    # indices[spatial_lookup,shift] = iat

    cell_offsets = jnp.asarray([[-1,-1,-1],[-1,-1,0],[-1,-1,1],
                    [-1,0,-1],[-1,0,0],[-1,0,1],
                    [-1,1,-1],[-1,1,0],[-1,1,1],
                    [0,-1,-1],[0,-1,0],[0,-1,1],
                    [0,0,-1],[0,0,0],[0,0,1],
                    [0,1,-1],[0,1,0],[0,1,1],
                    [1,-1,-1],[1,-1,0],[1,-1,1],
                    [1,0,-1],[1,0,0],[1,0,1],
                    [1,1,-1],[1,1,0],[1,1,1]])
    isysneigh=jnp.repeat(isys[:,None],cell_offsets.shape[0],axis=-1)
    neigh_cells = hash_cell(cell_coords[:,None,:]+cell_offsets[None,:,:],isysneigh,nel)
    neighbors = indices[neigh_cells].reshape(nattot,-1)
    src = jnp.asarray(np.repeat(np.arange(nattot),neighbors.shape[1]))
    neighbors = neighbors.flatten()

    d12 = jnp.sum((coords[src]-coords[neighbors])**2,axis=-1)
    mask = (d12<cutoff**2)*(neighbors>=0)*(isys[src]==isys[neighbors])
    npairs=jnp.sum(mask)
    d12 = d12*mask + (cutoff**2)*(1-mask)
    src = src*mask + padding_value*(1-mask)
    neighbors = neighbors*mask + padding_value*(1-mask)
    idx = jnp.argsort(d12)[:max_pairs]

    return src[idx],neighbors[idx],d12[idx],max_count,npairs

@partial(jax.jit, static_argnums=(1,4,5))
def compute_nblist_fixed(coords,cutoff,isys,natoms,max_nat,max_pairs,padding_value):
    p1,p2=jnp.triu_indices(max_nat, 1)
    mask = ((p1[None,:]<natoms[:,None])*(p2[None,:]<natoms[:,None])).flatten()

    if natoms.shape[0]==1:
        shift=jnp.array([0])
    else:
        shift=jnp.concatenate((jnp.array([0]),jnp.cumsum(natoms[:-1])))

    p1 = (p1[None,:]+shift[:,None]).flatten()*mask - (1-mask)
    p2 = (p2[None,:]+shift[:,None]).flatten()*mask - (1-mask)
    d12=jnp.sum((coords[p1]-coords[p2])**2,axis=-1)

    mask = mask*(d12<cutoff**2)
    d12 = d12*mask + (cutoff**2)*(1-mask)
    p1 = p1*mask + padding_value*(1-mask)
    p2 = p2*mask + padding_value*(1-mask)
    npairs=jnp.sum(mask)
    idx = jnp.argsort(d12)[:max_pairs]

    edge_src,edge_dst = p1[idx],p2[idx]
    return edge_src,edge_dst,d12[idx],npairs