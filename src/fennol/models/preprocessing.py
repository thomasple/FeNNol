import flax.linen as nn
from typing import Sequence, Callable, Union, Dict, Any
import jax.numpy as jnp
import jax
import numpy as np
from typing import Optional, Tuple
import numba
import dataclasses
from functools import partial

from flax.core.frozen_dict import FrozenDict


from ..utils.activations import chain
from ..utils import deep_update
from .modules import FENNIXModules
from .misc.misc import SwitchFunction
from ..utils.nblist import to_sparse_nblist


_SPARSE_DEFAULT = False
_ANGLE_SPARSE_DEFAULT = True


def minmaxone(x, name=""):
    print(name, x.min(), x.max(), (x**2).mean())


def update_size(sizes, key, new_size):
    prev_size = sizes.get(key, (0, 1))[0]
    size = new_size[0]
    return {**sizes, key: (jnp.maximum(size, prev_size), new_size[1])}

@dataclasses.dataclass(frozen=True)
class GraphGenerator:
    cutoff: float
    add_neigh: int = 5
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict, hash=False)
    kmax: int = 30
    kthr: float = 1e-6
    k_space: bool = False
    sparse: bool = _SPARSE_DEFAULT
    mult_size: float = 1.05
    # covalent_cutoff: bool = False

    def init(self):
        return FrozenDict(
            {
                "max_nat": 1,
                "max_neigh": 1,
                "add_neigh": self.add_neigh,
                "nblist_mult_size": self.mult_size,
            }
        )

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "switch_params": self.switch_params,
            "sparse": self.sparse,
            "name": f"{self.graph_key}_Processor",
        }

    def get_graph_properties(self):
        return {
            self.graph_key: {
                "cutoff": self.cutoff,
                "directed": True,
                "has_sparse": self.sparse,
            }
        }

    def __call__(self, state, inputs, parent_overflow=False):
        output = self.process(state, inputs)
        (state, _), output, parent_overflow = self.check_reallocate(
            state, output, parent_overflow
        )
        return state, output, parent_overflow

    def check_reallocate(self, state, inputs, parent_overflow=False):
        overflow = parent_overflow or inputs[self.graph_key].get("overflow", False)
        if not overflow:
            return (state, {}), inputs, False

        up_sizes = {}
        while overflow:
            sizes = jax.tree_map(int, inputs[self.graph_key]["sizes"])
            state_up = {}
            max_nat, prev_max_nat = sizes["max_nat"]
            if max_nat > prev_max_nat:
                state_up["max_nat"] = max_nat
                up_sizes["max_nat"] = (max_nat, prev_max_nat)

            max_neigh, prev_max_neigh = sizes["max_neigh"]
            if max_neigh > prev_max_neigh:
                state_up["max_neigh"] = max_neigh + state.get(
                    "add_neigh", self.add_neigh
                )
                up_sizes["max_neigh"] = (state_up["max_neigh"], prev_max_neigh)

            if "max_neigh_skin" in sizes:
                max_neigh_skin, prev_max_neigh_skin = sizes["max_neigh_skin"]
                if max_neigh_skin > prev_max_neigh_skin:
                    state_up["max_neigh_skin"] = max_neigh_skin + state.get(
                        "add_neigh", self.add_neigh
                    )
                    up_sizes["max_neigh_skin"] = (state_up["max_neigh_skin"], prev_max_neigh_skin)

            if "npairs" in sizes:
                npairs, prev_npairs = sizes["npairs"]
                if npairs > prev_npairs:
                    mult_size = state.get("nblist_mult_size", self.mult_size)
                    state_up["npairs"] = int(mult_size * npairs) + 1
                    up_sizes["npairs"] = (state_up["npairs"], prev_npairs)

            state = state.copy(state_up)
            inputs[self.graph_key]["overflow"] = False
            inputs = self.process(state, inputs)
            overflow = inputs[self.graph_key].get("overflow", False)
        return (state, up_sizes), inputs, True

    def process(self, state, inputs):
        coords = inputs["coordinates"]
        natoms = inputs["natoms"]
        batch_index = inputs["batch_index"]
        # assert (
        #     inputs["natoms"].shape[0] == 1
        # ), "Only one system is supported for graph_generator"

        max_nat = state.get("max_nat", round(coords.shape[0] / natoms.shape[0]))
        mask_atoms = inputs.get("true_atoms", None)

        ## prepare PBC computations
        if "cells" in inputs:
            cells = inputs["cells"]
            reciprocal_cells = inputs["reciprocal_cells"]

            def compute_pbc(vec, reciprocal_cell, cell):
                vecpbc = jnp.dot(vec, reciprocal_cell.T)
                pbc_shifts = -jnp.round(vecpbc)
                return vec + jnp.dot(pbc_shifts, cell.T), pbc_shifts

        ### compute all vectors and apply PBC
        pbc_shifts = None
        if natoms.shape[0] == 1:
            p12 = jnp.broadcast_to(
                jnp.arange(coords.shape[0])[None, :], (coords.shape[0], coords.shape[0])
            )
            mask_p12 = ~jnp.eye(coords.shape[0], dtype=bool)
            # mask_p12 = p12 != jnp.arange(coords.shape[0])[:, None]

            vec = coords[p12] - coords[:, None, :]
            if "cells" in inputs:
                vec, pbc_shifts = compute_pbc(vec, reciprocal_cells[0], cells[0])
        else:
            shift = jnp.concatenate((jnp.array([0]), jnp.cumsum(natoms[:-1])))
            p12 = jnp.arange(max_nat)[None, :] + shift[batch_index, None]
            mask_p12 = (jnp.arange(max_nat)[None, :] < natoms[batch_index, None]) & (
                p12 != jnp.arange(coords.shape[0])[:, None]
            )

            vec = coords[p12] - coords[:, None, :]
            if "cells" in inputs:
                vec, pbc_shifts = jax.vmap(compute_pbc)(
                    vec, reciprocal_cells[batch_index], cells[batch_index]
                )

        ## mask padding atoms
        if "true_atoms" in inputs:
            mask_atoms = inputs["true_atoms"]
            mask_p12 = mask_p12 & mask_atoms[:, None] & mask_atoms[p12]

        ## compute distances
        cutoff_skin = self.cutoff + state.get("nblist_skin", 0.0)
        d12 = (vec**2).sum(axis=-1)
        d12 = jnp.where(mask_p12, d12, cutoff_skin**2)

        ## count neighbors
        mask = d12 < cutoff_skin**2
        count = jnp.count_nonzero(mask,axis=-1) #.sum(axis=-1)
        max_count = jnp.max(count)
        max_neigh = min(state.get("max_neigh", 1),max_nat)

        ## sort neighbors and filter
        idx = jnp.argsort(d12, axis=-1)[:, :max_neigh]
        d12 = jnp.take_along_axis(d12, idx, axis=-1)
        # d12, idx = jax.lax.top_k(-d12, max_neigh)
        # d12 = -d12
        edge_mask = jnp.take_along_axis(mask, idx, axis=-1)
        # neigh_index = jnp.take_along_axis(p12, idx, axis=-1)
        neigh_index = jnp.where(
            edge_mask, jnp.take_along_axis(p12, idx, axis=-1), coords.shape[0]
        )
        if "cells" in inputs:
            # pbc_shifts = jnp.take_along_axis(pbc_shifts, idx[:, :, None], axis=1)
            pbc_shifts = jnp.where(
                edge_mask[:, :, None],
                jnp.take_along_axis(pbc_shifts, idx[:, :, None], axis=1),
                0,
            )

        ## check for overflow
        if natoms.shape[0] == 1:
            true_max_nat = coords.shape[0]
        else:
            true_max_nat = jnp.max(natoms)
        overflow_count = max_count > max_neigh
        overflow_at = true_max_nat > max_nat
        overflow = overflow_count | overflow_at

        if "nblist_skin" in state:
            # edge_mask_skin = edge_mask
            neigh_index_skin = neigh_index
            if "cells" in inputs:
                pbc_shifts_skin = pbc_shifts
            mask = d12 < self.cutoff**2
            count = jnp.count_nonzero(mask,axis=-1) #.sum(axis=-1)
            max_count_skin = count.max()
            max_neigh_skin = state.get("max_neigh_skin", 1)
            max_neigh_skin = min(max_neigh_skin, max_neigh)
            overflow = overflow | (max_count_skin > max_neigh_skin)
            neigh_index = neigh_index[:, :max_neigh_skin]
            d12 = d12[:, :max_neigh_skin]
            # edge_mask = edge_mask[:, :max_neigh_skin]
            if "cells" in inputs:
                pbc_shifts = pbc_shifts[:, :max_neigh_skin, :]

        graph = inputs[self.graph_key] if self.graph_key in inputs else {}
        sizes = graph.get("sizes", {})
        sizes = update_size(sizes, "max_nat", (true_max_nat, max_nat))
        sizes = update_size(sizes, "max_neigh", (max_count, max_neigh))
        graph_out = {
            **graph,
            # "edge_mask": edge_mask,
            "neigh_index": neigh_index,
            "d12": d12,
            "overflow": overflow,
            "pbc_shifts": pbc_shifts,
        }
        if "nblist_skin" in state:
            # graph_out["edge_mask_skin"] = edge_mask_skin
            graph_out["neigh_index_skin"] = neigh_index_skin
            if "cells" in inputs:
                graph_out["pbc_shifts_skin"] = pbc_shifts_skin
            # graph_out["sizes"]["max_neigh_skin"] = (max_count_skin, max_neigh_skin)
            sizes = update_size(sizes, "max_neigh_skin", (max_count_skin, max_neigh_skin))

        if self.sparse:
            prev_nblist_size = state.get("npairs", 1)
            npairs = count.sum()
            edge_src, edge_dst, idx = to_sparse_nblist(
                neigh_index, d12, prev_nblist_size
            )
            graph_out["edge_src"] = edge_src
            graph_out["edge_dst"] = edge_dst
            graph_out["sparse_index"] = idx
            graph_out["overflow_sparse"] = npairs > prev_nblist_size
            graph_out["overflow"] = graph_out["overflow"] | graph_out["overflow_sparse"]
            sizes = update_size(sizes, "npairs", (npairs, prev_nblist_size))

        graph_out["sizes"] = sizes
        return {**inputs, self.graph_key: graph_out}

    def update_skin(self, inputs):
        graph = inputs[self.graph_key]
        max_neigh = graph["neigh_index"].shape[1]

        neigh_index_skin = graph["neigh_index_skin"]
        coords = inputs["coordinates"]
        vec = coords[neigh_index_skin] - coords[:, None, :]
        if "cells" in inputs:
            pbc_shifts_skin = graph["pbc_shifts_skin"]
            cells = inputs["cells"]
            if cells.shape[0] == 1:
                cells = cells[0].T
                vec = vec + jnp.dot(pbc_shifts_skin, cells)
                # vec = vec + jnp.einsum("aij,abj->abi", cells, pbc_shifts_skin)
            else:
                batch_index = inputs["batch_index"]
                cells = jnp.swapaxes(cells, -2, -1)[batch_index]
                # vec = vec + jnp.einsum("aij,abj->abi", cells, pbc_shifts_skin)
                vec = vec + jax.vmap(jnp.dot)(pbc_shifts_skin, cells)

        # vec = jnp.where(edge_mask[:, :, None], vec, cutoff)
        # edge_mask = graph["edge_mask_skin"]
        edge_mask = neigh_index_skin < coords.shape[0]
        d12 = jnp.where(edge_mask, jnp.sum(vec**2, axis=-1), self.cutoff**2)
        mask = d12 < self.cutoff**2
        counts = jnp.count_nonzero(mask,axis=-1)
        max_counts = counts.max()
        overflow = max_counts > max_neigh

        idx = jnp.argsort(d12, axis=-1)[:, :max_neigh]
        d12 = jnp.take_along_axis(d12, idx, axis=-1)
        # d12, idx = jax.lax.top_k(-d12, max_neigh)
        # d12 = -d12
        edge_mask = jnp.take_along_axis(mask, idx, axis=-1)
        # neigh_index = jnp.take_along_axis(neigh_index_skin, idx, axis=-1)
        neigh_index = jnp.where(
            edge_mask,
            jnp.take_along_axis(neigh_index_skin, idx, axis=-1),
            coords.shape[0],
        )
        if "cells" in inputs:
            # pbc_shifts = jnp.take_along_axis(pbc_shifts_skin, idx[:, :, None], axis=1)
            pbc_shifts = jnp.where(
                edge_mask[:, :, None],
                jnp.take_along_axis(pbc_shifts_skin, idx[:, :, None], axis=1),
                0,
            )

        sizes = graph["sizes"]
        sizes = update_size(sizes, "max_neigh", (max_counts, max_neigh))
        graph_out = {
            **graph,
            # "edge_mask": edge_mask,
            "neigh_index": neigh_index,
            "d12": d12,
            "overflow": jnp.logical_or(overflow, graph.get("overflow", False)),
        }
        if "cells" in inputs:
            graph_out["pbc_shifts"] = pbc_shifts

        if self.sparse:
            prev_nblist_size = graph["edge_src"].shape[0]
            npairs = counts.sum()
            edge_src, edge_dst, idx = to_sparse_nblist(
                neigh_index, d12, prev_nblist_size
            )
            graph_out["edge_src"] = edge_src
            graph_out["edge_dst"] = edge_dst
            graph_out["sparse_index"] = idx
            graph_out["overflow"] = jnp.logical_or(
                graph_out["overflow"], npairs > prev_nblist_size
            )
            sizes = update_size(sizes, "npairs", (npairs, prev_nblist_size))

        graph_out["sizes"] = sizes
        return {**inputs, self.graph_key: graph_out}


class GraphProcessor(nn.Module):
    cutoff: float
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict)
    sparse: bool = _SPARSE_DEFAULT

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph = inputs[self.graph_key]
        coords = inputs["coordinates"]
        neigh_index = graph["neigh_index"]
        edge_mask = neigh_index < coords.shape[0]
        # edge_mask = graph["edge_mask"]
        vec = coords[neigh_index] - coords[:, None, :]
        if "cells" in inputs:
            cells = inputs["cells"]
            if cells.shape[0] == 1:
                cells = cells[0].T
                vec = vec + jnp.dot(graph["pbc_shifts"], cells)
                # vec = vec + jnp.einsum("aij,abj->abi", cells, pbc_shifts_skin)
            else:
                batch_index = inputs["batch_index"]
                cells = jnp.swapaxes(cells, -2, -1)[batch_index]
                # vec = vec + jnp.einsum("aij,abj->abi", cells, pbc_shifts_skin)
                vec = vec + jax.vmap(jnp.dot)(graph["pbc_shifts"], cells)

        vec = jnp.where(edge_mask[:, :, None], vec, self.cutoff)
        distances = jnp.linalg.norm(vec, axis=-1)
        edge_mask = distances < self.cutoff

        switch = SwitchFunction(
            **{**self.switch_params, "cutoff": self.cutoff, "graph_key": None}
        )((distances, edge_mask))

        graph_out = {
            **graph,
            "vec": vec,
            "distances": distances,
            "switch": switch,
            # "edge_mask": edge_mask,
        }
        return {**inputs, self.graph_key: graph_out}


@dataclasses.dataclass(frozen=True)
class GraphFilter:
    cutoff: float
    parent_graph: str
    graph_key: str
    add_neigh: int = 5
    remove_hydrogens: int = False
    switch_params: FrozenDict = dataclasses.field(default_factory=FrozenDict)
    k_space: bool = False
    kmax: int = 30
    kthr: float = 1e-6
    sparse: bool = _SPARSE_DEFAULT
    mult_size: float = 1.05

    def init(self):
        return FrozenDict(
            {
                "max_neigh": self.add_neigh,
                "add_neigh": self.add_neigh,
                "nblist_mult_size": self.mult_size,
            }
        )

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphFilterProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "parent_graph": self.parent_graph,
            "name": f"{self.graph_key}_Filter_{self.parent_graph}",
            "switch_params": self.switch_params,
            "sparse": self.sparse,
        }

    def get_graph_properties(self):
        return {
            self.graph_key: {
                "cutoff": self.cutoff,
                "directed": True,
                "parent_graph": self.parent_graph,
                "has_sparse": self.sparse,
            }
        }

    def __call__(self, state, inputs, parent_overflow=False):
        output = self.process(state, inputs)
        (state, _), output, parent_overflow = self.check_reallocate(
            state, output, parent_overflow
        )
        return state, output, parent_overflow

    def check_reallocate(self, state, inputs, parent_overflow=False):
        overflow = parent_overflow or inputs[self.graph_key].get("overflow", False)
        if not overflow:
            return (state, {}), inputs, False

        up_sizes = {}
        while overflow:
            sizes = jax.tree_map(int, inputs[self.graph_key]["sizes"])
            state_up = {}

            max_neigh, prev_max_neigh = sizes["max_neigh"]
            if max_neigh > prev_max_neigh:
                state_up["max_neigh"] = max_neigh + state.get(
                    "add_neigh", self.add_neigh
                )
                up_sizes["max_neigh"] = (state_up["max_neigh"], prev_max_neigh)

            if "npairs" in sizes:
                npairs, prev_npairs = sizes["npairs"]
                if npairs > prev_npairs:
                    mult_size = state.get("nblist_mult_size", self.mult_size)
                    state_up["npairs"] = int(mult_size * npairs) + 1
                    up_sizes["npairs"] = (state_up["npairs"], prev_npairs)

            state = state.copy(state_up)
            inputs[self.graph_key]["overflow"] = False
            inputs = self.process(state, inputs)
            overflow = inputs[self.graph_key].get("overflow", False)
        return (state, up_sizes), inputs, True

    def process(self, state, inputs):
        graph_in = inputs[self.parent_graph]
        if state is None:
            # skin update mode
            graph = inputs[self.graph_key]
            max_neigh = graph["neigh_index"].shape[1]
            if self.sparse:
                prev_nblist_size = graph["edge_src"].shape[0]
        else:
            max_neigh = state.get("max_neigh", 1)
            if self.sparse:
                prev_nblist_size = state.get("npairs", 1)

        d12 = graph_in["d12"]
        mask = d12 < self.cutoff**2

        count = jnp.count_nonzero(mask,axis=-1)
        max_count = count.max()
        # max_neigh = state.get("max_neigh", 1)
        overflow = max_count > max_neigh

        mask = mask[:, :max_neigh]
        # neigh_index = graph_in["neigh_index"][:, :max_neigh]
        neigh_index = jnp.where(
            mask, graph_in["neigh_index"][:, :max_neigh], d12.shape[0]
        )
        d12 = d12[:, :max_neigh]

        graph = inputs[self.graph_key] if self.graph_key in inputs else {}
        sizes = graph.get("sizes", {})
        sizes = update_size(sizes, "max_neigh", (max_count, max_neigh))
        graph_out = {
            **graph,
            # "edge_mask": edge_mask,
            "neigh_index": neigh_index,
            "d12": d12,
            "overflow": overflow,
        }

        if self.sparse:
            # prev_nblist_size = state.get("npairs", 1)
            npairs = count.sum()
            edge_src, edge_dst, idx = to_sparse_nblist(
                neigh_index, d12, prev_nblist_size
            )
            graph_out["edge_src"] = edge_src
            graph_out["edge_dst"] = edge_dst
            graph_out["sparse_index"] = idx
            graph_out["overflow"] = graph_out["overflow"] | (npairs > prev_nblist_size)
            # graph_out["sizes"]["npairs"] = (npairs, prev_nblist_size)
            sizes = update_size(sizes, "npairs", (npairs, prev_nblist_size))

        graph_out["sizes"] = sizes
        return {**inputs, self.graph_key: graph_out}

    def update_skin(self, inputs):
        return self.process(None, inputs)


class GraphFilterProcessor(nn.Module):
    cutoff: float
    graph_key: str
    parent_graph: str
    switch_params: dict = dataclasses.field(default_factory=dict)
    sparse: bool = _SPARSE_DEFAULT

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph_in = inputs[self.parent_graph]
        graph = inputs[self.graph_key]

        max_neigh = graph["neigh_index"].shape[1]
        vec = graph_in["vec"][:, :max_neigh, :]
        distances = graph_in["distances"][:, :max_neigh]

        edge_mask = distances < self.cutoff
        switch = SwitchFunction(
            **{**self.switch_params, "cutoff": self.cutoff, "graph_key": None}
        )((distances, edge_mask))

        graph_out = {
            **graph,
            "vec": vec,
            "distances": distances,
            "switch": switch,
            # "edge_mask": edge_mask,
        }
        return {**inputs, self.graph_key: graph_out}


@dataclasses.dataclass(frozen=True)
class GraphAngularExtension:
    mult_size: float = 1.05
    graph_key: str = "graph"
    sparse: bool = _ANGLE_SPARSE_DEFAULT

    def init(self):
        if self.sparse:
            return FrozenDict({"nangles": 0, "nblist_mult_size": self.mult_size})
        return FrozenDict({})

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphAngleProcessor, {
            "graph_key": self.graph_key,
            "name": f"{self.graph_key}_AngleProcessor",
        }

    def get_graph_properties(self):
        return {
            self.graph_key: {
                "has_angles": True,
                "sparse_angles": self.sparse,
            }
        }

    def __call__(self, state, inputs, parent_overflow=False):
        output = self.process(state, inputs)
        (state, _), output, parent_overflow = self.check_reallocate(
            state, output, parent_overflow
        )
        return state, output, parent_overflow

    def check_reallocate(self, state, inputs, parent_overflow=False):
        overflow = parent_overflow or inputs[self.graph_key].get(
            "angle_overflow", False
        )
        if not overflow:
            return (state, {}), inputs, False
        if not self.sparse:
            return (state, {}), inputs, True

        up_sizes = {}
        while overflow:
            sizes = jax.tree_map(int, inputs[self.graph_key]["sizes"])
            nangles, prev_nangles = sizes["nangles"]
            if nangles > prev_nangles:
                mult_size = state.get("nblist_mult_size", self.mult_size)
                nangles = int(mult_size * nangles) + 1
                state = state.copy({"nangles": nangles})
                up_sizes["nangles"] = (nangles, prev_nangles)

            inputs[self.graph_key]["angle_overflow"] = False
            inputs = self.process(state, inputs)
            overflow = inputs[self.graph_key].get("angle_overflow", False)
        return (state, up_sizes), inputs, True

    def process(self, state, inputs):

        if not self.sparse:
            if state is None:
                return inputs

            graph = inputs[self.graph_key]
            neigh_index = graph["neigh_index"]
            angle_src, angle_dst = jnp.triu_indices(neigh_index.shape[1], 1)
            return {
                **inputs,
                self.graph_key: {
                    **graph,
                    "angle_src": angle_src,
                    "angle_dst": angle_dst,
                },
            }

        graph = inputs[self.graph_key]
        neigh_index = graph["neigh_index"]
        angle_src, angle_dst = np.triu_indices(neigh_index.shape[1], 1)

        central_atom = jnp.asarray(
            np.broadcast_to(
                np.arange(neigh_index.shape[0], dtype=np.int32)[:, None],
                (neigh_index.shape[0], angle_src.shape[0]),
            ).flatten()
        )
        angle_src_sparse = jnp.asarray(
            np.broadcast_to(
                angle_src[None, :], (neigh_index.shape[0], angle_src.shape[0])
            ).flatten()
        )
        angle_dst_sparse = jnp.asarray(
            np.broadcast_to(
                angle_dst[None, :], (neigh_index.shape[0], angle_src.shape[0])
            ).flatten()
        )

        # edge_mask = graph["edge_mask"]
        # angle_mask = (edge_mask[:, angle_src] & edge_mask[:, angle_dst]).flatten()
        nat = inputs["species"].shape[0]
        mask1 = neigh_index[:, angle_src].flatten() < nat
        mask2 = neigh_index[:, angle_dst].flatten() < nat
        angle_mask = mask1 & mask2
        nangles = jnp.count_nonzero(angle_mask) #.sum()
        if state is None:
            prev_nangles = graph["angle_src_sparse"].shape[0]
        else:
            prev_nangles = state.get("nangles", 0)

        overflow = graph.get("angle_overflow", False) | (nangles > prev_nangles)

        idx = jnp.argsort(~angle_mask)[:prev_nangles]
        angle_src_sparse = angle_src_sparse[idx]
        angle_dst_sparse = angle_dst_sparse[idx]
        central_atom = central_atom[idx]

        sizes = graph["sizes"]
        sizes = update_size(sizes, "nangles", (nangles, prev_nangles))
        output = {
            **inputs,
            self.graph_key: {
                **graph,
                "angle_src_sparse": angle_src_sparse,
                "angle_dst_sparse": angle_dst_sparse,
                "central_atom": central_atom,
                "angle_overflow": overflow,
                "sizes": sizes
            },
        }

        return output

    def update_skin(self, inputs):
        return self.process(None, inputs)


class GraphAngleProcessor(nn.Module):
    graph_key: str

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        vec = graph["vec"]
        angle_src = graph["angle_src_sparse"]
        angle_dst = graph["angle_dst_sparse"]
        central_atom = graph["central_atom"]
        # neigh_index = graph["neigh_index"]

        # nat = inputs["species"].shape[0]

        # angle_at1 = neigh_index[:, angle_src]
        # angle_at2 = neigh_index[:, angle_dst]
        # angle_mask = (angle_at1 < nat) & (angle_at2 < nat)

        d1 = distances[central_atom, angle_src]
        d2 = distances[central_atom, angle_dst]
        vec1 = vec[central_atom, angle_src, :]
        vec2 = vec[central_atom, angle_dst, :]

        cos_angles = (vec1 * vec2).sum(axis=-1) / jnp.clip(d1 * d2, a_min=1e-10)
        angles = jnp.arccos(0.95 * cos_angles)

        return {
            **inputs,
            self.graph_key: {
                **graph,
                "cos_angles": cos_angles,
                "angles": angles,
                # "angle_mask": angle_mask,
            },
        }


@dataclasses.dataclass(frozen=True)
class AtomPadding:
    mult_size: float = 1.2

    def init(self):
        return {"prev_nat": 0}

    def __call__(self, state, inputs: Dict) -> Union[dict, jax.Array]:
        species = inputs["species"]
        nat = species.shape[0]

        prev_nat = state.get("prev_nat", 0)
        prev_nat_ = prev_nat
        if nat > prev_nat_:
            prev_nat_ = int(self.mult_size * nat) + 1

        nsys = len(inputs["natoms"])

        add_atoms = prev_nat_ - nat
        output = {**inputs}
        if add_atoms > 0:
            batch_index = inputs["batch_index"]
            for k, v in inputs.items():
                if isinstance(v, np.ndarray):
                    if v.shape[0] == nat:
                        output[k] = np.append(
                            v,
                            np.zeros((add_atoms, *v.shape[1:]), dtype=v.dtype),
                            axis=0,
                        )
                    elif v.shape[0] == nsys:
                        if k == "cells":
                            output[k] = np.append(
                                v,
                                np.eye(3, dtype=v.dtype)[None, :, :],
                                axis=0,
                            )
                        else:
                            output[k] = np.append(
                                v, np.zeros((1, *v.shape[1:]), dtype=v.dtype), axis=0
                            )
            output["natoms"] = np.append(inputs["natoms"], 0)
            output["species"] = np.append(
                species, -1 * np.ones(add_atoms, dtype=species.dtype)
            )
            output["batch_index"] = np.append(
                batch_index, np.array([nsys] * add_atoms, dtype=batch_index.dtype)
            )

        output["true_atoms"] = output["species"] > 0
        output["true_sys"] = np.arange(len(output["natoms"])) < nsys

        state = {**state, "prev_nat": prev_nat_}

        return state, output


def atom_unpadding(inputs: Dict[str, Any]) -> Dict[str, Any]:
    if "true_atoms" not in inputs:
        return inputs

    species = inputs["species"]
    true_atoms = inputs["true_atoms"]
    true_sys = inputs["true_sys"]
    natall = species.shape[0]
    nat = np.argmax(species <= 0)
    if nat == 0:
        return inputs

    natoms = inputs["natoms"]
    nsysall = len(natoms)

    output = {**inputs}
    for k, v in inputs.items():
        if isinstance(v, jax.Array) or isinstance(v, np.ndarray):
            if v.ndim == 0:
                continue
            if v.shape[0] == natall:
                output[k] = v[true_atoms]
            elif v.shape[0] == nsysall:
                output[k] = v[true_sys]
    del output["true_sys"]
    del output["true_atoms"]
    return output


def check_input(inputs):
    assert "species" in inputs, "species must be provided"
    assert "coordinates" in inputs, "coordinates must be provided"
    species = inputs["species"]
    ifake = np.argmax(species <= 0)
    if ifake > 0:
        assert np.all(species[:ifake] > 0), "species must be positive"
    nat = inputs["species"].shape[0]

    natoms = inputs.get("natoms", np.array([nat], dtype=np.int64))
    batch_index = inputs.get(
        "batch_index", np.repeat(np.arange(len(natoms), dtype=np.int64), natoms)
    )
    output = {**inputs, "natoms": natoms, "batch_index": batch_index}
    if "cells" in inputs:
        cells = inputs["cells"]
        if cells.ndim == 2:
            cells = cells[None, :, :]
        if "reciprocal_cells" not in inputs:
            reciprocal_cells = np.linalg.inv(cells)
        else:
            reciprocal_cells = inputs["reciprocal_cells"]
        if cells.ndim == 2:
            cells = cells[None, :, :]
        if reciprocal_cells.ndim == 2:
            reciprocal_cells = reciprocal_cells[None, :, :]
        output["cells"] = cells
        output["reciprocal_cells"] = reciprocal_cells

    return output


def convert_to_jax(data):
    def convert(x):
        if isinstance(x, np.ndarray):
            # if x.dtype == np.float64:
            #     return jnp.asarray(x, dtype=jnp.float32)
            return jnp.asarray(x)
        return x

    return jax.tree_util.tree_map(convert, data)


class JaxConverter(nn.Module):
    def __call__(self, data):
        return convert_to_jax(data)


@dataclasses.dataclass(frozen=True)
class PreprocessingChain:
    layers: Tuple[Callable[..., Dict[str, Any]]]
    use_atom_padding: bool = False
    atom_padder: AtomPadding = AtomPadding()

    def __post_init__(self):
        if not isinstance(self.layers, Sequence):
            raise ValueError(
                f"'layers' must be a sequence, got '{type(self.layers).__name__}'."
            )
        if not self.layers:
            raise ValueError(f"Error: no Preprocessing layers were provided.")

    def __call__(self, state, inputs: Dict[str, Any]) -> Dict[str, Any]:
        do_check_input = state.get("check_input", True)
        if do_check_input:
            inputs = check_input(inputs)
        new_state = []
        layer_state = state["layers_state"]
        i = 0
        if self.use_atom_padding:
            s,inputs = self.atom_padder(layer_state[0],inputs)
            new_state.append(s)
            i += 1
        if do_check_input:
            inputs = convert_to_jax(inputs)
        parent_overflow = False
        for layer in self.layers:
            s, inputs, parent_overflow = layer(layer_state[i], inputs, parent_overflow)
            new_state.append(s)
            i += 1
        return FrozenDict({**state, "layers_state": tuple(new_state)}), inputs

    def check_reallocate(self, state, inputs):
        new_state = []
        state_up = []
        layer_state = state["layers_state"]
        i = 0
        if self.use_atom_padding:
            new_state.append(layer_state[0])
            i += 1
        parent_overflow = False
        for layer in self.layers:
            (s, s_up), inputs, parent_overflow = layer.check_reallocate(
                layer_state[i], inputs, parent_overflow
            )
            new_state.append(s)
            state_up.append(s_up)
            i += 1

        if not parent_overflow:
            return (state, {}), inputs, False
        return (
            (FrozenDict({**state, "layers_state": tuple(new_state)}), state_up),
            inputs,
            True,
        )

    @partial(jax.jit, static_argnums=(0, 1))
    def process(self, state, inputs):
        layer_state = state["layers_state"]
        i = 1 if self.use_atom_padding else 0
        for layer in self.layers:
            inputs = layer.process(layer_state[i], inputs)
            i += 1
        return inputs

    @partial(jax.jit, static_argnums=(0))
    def update_skin(self, inputs):
        for layer in self.layers:
            inputs = layer.update_skin(inputs)
        return inputs

    def init(self):
        state = []
        if self.use_atom_padding:
            state.append(self.atom_padder.init())
        for layer in self.layers:
            state.append(layer.init())
        return FrozenDict({"check_input": True, "layers_state": state})

    def init_with_output(self, inputs):
        state = self.init()
        return self(state, inputs)

    def get_processors(self, return_list=False):
        processors = []
        for layer in self.layers:
            if hasattr(layer, "get_processor"):
                processors.append(layer.get_processor())
        if return_list:
            return processors
        return FENNIXModules(processors)

    def get_graphs_properties(self):
        properties = {}
        for layer in self.layers:
            if hasattr(layer, "get_graph_properties"):
                properties = deep_update(properties, layer.get_graph_properties())
        return properties


PREPROCESSING = {
    "GRAPH": GraphGenerator,
    # "GRAPH_FIXED": GraphGeneratorFixed,
    "GRAPH_FILTER": GraphFilter,
    "GRAPH_ANGULAR_EXTENSION": GraphAngularExtension,
    # "GRAPH_DENSE_EXTENSION": GraphDenseExtension,
}
