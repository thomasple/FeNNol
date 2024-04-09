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
from ..utils import deep_update, mask_filter_1d
from .modules import FENNIXModules
from .misc.misc import SwitchFunction

# from ..utils.nblist import to_sparse_nblist


_SPARSE_DEFAULT = False
_ANGLE_SPARSE_DEFAULT = True


def minmaxone(x, name=""):
    print(name, x.min(), x.max(), (x**2).mean())


@dataclasses.dataclass(frozen=True)
class GraphGenerator:
    cutoff: float
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict, hash=False)
    kmax: int = 30
    kthr: float = 1e-6
    k_space: bool = False
    mult_size: float = 1.05
    # covalent_cutoff: bool = False

    def init(self):
        return FrozenDict(
            {
                "max_nat": 1,
                "npairs": 1,
                "nblist_mult_size": self.mult_size,
            }
        )

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "switch_params": self.switch_params,
            "name": f"{self.graph_key}_Processor",
        }

    def get_graph_properties(self):
        return {
            self.graph_key: {
                "cutoff": self.cutoff,
                "directed": True,
            }
        }

    def __call__(self, state, inputs, return_state_update=False,add_margin=False):
        """ build a nblist on cpu with numpy and dynamic shapes + store max shapes """
        coords = np.array(inputs["coordinates"])
        natoms = np.array(inputs["natoms"])
        batch_index = np.array(inputs["batch_index"])

        new_state = {**state}
        state_up = {}

        max_nat = state.get("max_nat", round(coords.shape[0] / natoms.shape[0]))
        true_max_nat = np.max(natoms)
        if true_max_nat > max_nat:
            state_up["max_nat"] = (true_max_nat, max_nat)
            new_state["max_nat"] = true_max_nat

        ### compute indices of all pairs
        p1, p2 = np.triu_indices(true_max_nat, 1)
        p1, p2 = p1.astype(np.int32), p2.astype(np.int32)
        pbc_shifts = None
        if natoms.shape[0] > 1:
            ## batching => mask irrelevant pairs
            mask_p12 = (
                (p1[None, :] < natoms[:, None]) * (p2[None, :] < natoms[:, None])
            ).flatten()
            shift = np.concatenate((np.array([0]), np.cumsum(natoms[:-1])))
            p1 = np.where(mask_p12, (p1[None, :] + shift[:, None]).flatten(), -1)
            p2 = np.where(mask_p12, (p2[None, :] + shift[:, None]).flatten(), -1)

        ## compute vectors
        vec = coords[p2] - coords[p1]

        ## apply PBC (minimum image convention only for now)
        if "cells" in inputs:
            cells = np.array(inputs["cells"])
            reciprocal_cells = np.array(inputs["reciprocal_cells"])

            if cells.shape[0] == 1:
                vecpbc = np.dot(vec, reciprocal_cells[0].T)
                pbc_shifts = -np.round(vecpbc)
                vec = vec + np.dot(pbc_shifts, cells[0].T)
            else:
                batch_index_vec = batch_index[p1]
                cells = np.swapaxes(cells, -2, -1)[batch_index_vec]
                reciprocal_cells = np.swapaxes(reciprocal_cells, -2, -1)[
                    batch_index_vec
                ]
                vecpbc = np.einsum("aj,aji->ai", vec, reciprocal_cells)
                pbc_shifts = -np.round(vecpbc)
                vec = vec + np.einsum("aj,aji->ai", pbc_shifts, cells)

        ## compute distances
        cutoff_skin = self.cutoff + state.get("nblist_skin", 0.0)
        d12 = (vec**2).sum(axis=-1)
        if natoms.shape[0] > 1:
            d12 = np.where(mask_p12, d12, cutoff_skin**2)

        ## filter pairs
        max_pairs = state.get("npairs", 1)
        mask = d12 < cutoff_skin**2
        idx = np.nonzero(mask)[0]
        npairs = idx.shape[0]
        if npairs > max_pairs or add_margin:
            mult_size = state.get("nblist_mult_size", self.mult_size)
            prev_max_pairs = max_pairs
            max_pairs = int(mult_size * max(npairs,max_pairs)) + 1
            state_up["npairs"] = (max_pairs, prev_max_pairs)
            new_state["npairs"] = max_pairs

        nat = coords.shape[0]
        edge_src = np.full(max_pairs, nat, dtype=np.int32)
        edge_dst = np.full(max_pairs, nat, dtype=np.int32)
        d12_ = np.full(max_pairs, cutoff_skin**2)
        edge_src[:npairs] = p1[idx]
        edge_dst[:npairs] = p2[idx]
        d12_[:npairs] = d12[idx]
        d12 = d12_
        if "cells" in inputs:
            pbc_shifts_ = np.full((max_pairs, 3), 0.0)
            pbc_shifts_[:npairs] = pbc_shifts[idx]
            pbc_shifts = pbc_shifts_

        if "nblist_skin" in state:
            # edge_mask_skin = edge_mask
            edge_src_skin = edge_src
            edge_dst_skin = edge_dst
            if "cells" in inputs:
                pbc_shifts_skin = pbc_shifts
            max_pairs_skin = state.get("npairs_skin", 1)
            mask = d12 < self.cutoff**2
            idx = np.nonzero(mask)[0]
            npairs_skin = idx.shape[0]
            if npairs_skin > max_pairs_skin or add_margin:
                mult_size = state.get("nblist_mult_size", self.mult_size)
                prev_max_pairs_skin = max_pairs_skin
                max_pairs_skin = int(mult_size * max(npairs_skin,max_pairs_skin)) + 1
                state_up["npairs_skin"] = (max_pairs_skin, prev_max_pairs_skin)
                new_state["npairs_skin"] = max_pairs_skin
            edge_src = np.full(max_pairs_skin, nat, dtype=np.int32)
            edge_dst = np.full(max_pairs_skin, nat, dtype=np.int32)
            d12_ = np.full(max_pairs_skin, self.cutoff**2)
            edge_src[:npairs_skin] = edge_src_skin[idx]
            edge_dst[:npairs_skin] = edge_dst_skin[idx]
            d12_[:npairs_skin] = d12[idx]
            d12 = d12_
            if "cells" in inputs:
                pbc_shifts = np.full((max_pairs_skin, 3), 0.0)
                pbc_shifts[:npairs_skin] = pbc_shifts_skin[idx]

        ## symmetrize
        edge_src, edge_dst = np.concatenate((edge_src, edge_dst)), np.concatenate(
            (edge_dst, edge_src)
        )
        d12 = np.concatenate((d12, d12))
        if "cells" in inputs:
            pbc_shifts = np.concatenate((pbc_shifts, -pbc_shifts))

        graph = inputs[self.graph_key] if self.graph_key in inputs else {}
        graph_out = {
            **graph,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "d12": d12,
            "overflow": False,
            "pbc_shifts": pbc_shifts,
        }
        if "nblist_skin" in state:
            graph_out["edge_src_skin"] = edge_src_skin
            graph_out["edge_dst_skin"] = edge_dst_skin
            if "cells" in inputs:
                graph_out["pbc_shifts_skin"] = pbc_shifts_skin

        output = {**inputs, self.graph_key: graph_out}

        if return_state_update:
            return FrozenDict(new_state), output, state_up
        return FrozenDict(new_state), output

    def check_reallocate(self, state, inputs, parent_overflow=False):
        """ check for overflow and reallocate nblist if necessary """
        overflow = parent_overflow or inputs[self.graph_key].get("overflow", False)
        if not overflow:
            return state,{}, inputs, False

        add_margin = inputs[self.graph_key].get("overflow", False)
        state, inputs, state_up = self(state, inputs, return_state_update=True,add_margin=add_margin)
        return state, state_up, inputs, True

    @partial(jax.jit, static_argnums=(0, 1))
    def process(self, state, inputs):
        """ build a nblist on acceleratir with jax and precomputed shapes """
        coords = inputs["coordinates"]
        natoms = inputs["natoms"]
        batch_index = inputs["batch_index"]

        max_nat = state.get("max_nat", round(coords.shape[0] / natoms.shape[0]))

        ### compute indices of all pairs
        p1, p2 = np.triu_indices(max_nat, 1)
        p1, p2 = p1.astype(np.int32), p2.astype(np.int32)
        pbc_shifts = None
        if natoms.shape[0] > 1:
            ## batching => mask irrelevant pairs
            mask_p12 = (
                (p1[None, :] < natoms[:, None]) * (p2[None, :] < natoms[:, None])
            ).flatten()
            shift = jnp.concatenate((jnp.array([0]), jnp.cumsum(natoms[:-1])))
            p1 = jnp.where(mask_p12, (p1[None, :] + shift[:, None]).flatten(), -1)
            p2 = jnp.where(mask_p12, (p2[None, :] + shift[:, None]).flatten(), -1)

        ## compute vectors
        vec = coords[p2] - coords[p1]

        ## apply PBC (minimum image convention only for now)
        if "cells" in inputs:
            cells = inputs["cells"]
            reciprocal_cells = inputs["reciprocal_cells"]

            def compute_pbc(vec, reciprocal_cell, cell):
                vecpbc = jnp.dot(vec, reciprocal_cell)
                pbc_shifts = -jnp.round(vecpbc)
                return vec + jnp.dot(pbc_shifts, cell), pbc_shifts

            if cells.shape[0] == 1:
                cells = cells[0].T
                vec, pbc_shifts = compute_pbc(vec, reciprocal_cells[0].T, cells)
            else:
                batch_index_vec = batch_index[p1]
                cells = jnp.swapaxes(cells, -2, -1)[batch_index_vec]
                reciprocal_cells = jnp.swapaxes(reciprocal_cells, -2, -1)[
                    batch_index_vec
                ]
                vec, pbc_shifts = jax.vmap(compute_pbc)(vec, reciprocal_cells, cells)

        ## compute distances
        cutoff_skin = self.cutoff + state.get("nblist_skin", 0.0)
        d12 = (vec**2).sum(axis=-1)
        if natoms.shape[0] > 1:
            d12 = jnp.where(mask_p12, d12, cutoff_skin**2)

        ## filter pairs
        max_pairs = state.get("npairs", 1)
        mask = d12 < cutoff_skin**2
        (edge_src, edge_dst, d12), scatter_idx, npairs = mask_filter_1d(
            mask,
            max_pairs,
            (jnp.asarray(p1, dtype=jnp.int32), coords.shape[0]),
            (jnp.asarray(p2, dtype=jnp.int32), coords.shape[0]),
            (d12, cutoff_skin**2),
        )
        if "cells" in inputs:
            pbc_shifts = (
                jnp.full((max_pairs, 3), 0.0)
                .at[scatter_idx]
                .set(pbc_shifts, mode="drop")
            )

        ## check for overflow
        if natoms.shape[0] == 1:
            true_max_nat = coords.shape[0]
        else:
            true_max_nat = jnp.max(natoms)
        overflow_count = npairs > max_pairs
        overflow_at = true_max_nat > max_nat
        overflow = overflow_count | overflow_at

        if "nblist_skin" in state:
            # edge_mask_skin = edge_mask
            edge_src_skin = edge_src
            edge_dst_skin = edge_dst
            if "cells" in inputs:
                pbc_shifts_skin = pbc_shifts
            max_pairs_skin = state.get("npairs_skin", 1)
            mask = d12 < self.cutoff**2
            (edge_src, edge_dst, d12), scatter_idx, npairs_skin = mask_filter_1d(
                mask,
                max_pairs_skin,
                (edge_src, coords.shape[0]),
                (edge_dst, coords.shape[0]),
                (d12, self.cutoff**2),
            )
            if "cells" in inputs:
                pbc_shifts = (
                    jnp.full((max_pairs_skin, 3), 0.0)
                    .at[scatter_idx]
                    .set(pbc_shifts, mode="drop")
                )
            overflow = overflow | (npairs_skin > max_pairs_skin)

        ## symmetrize
        edge_src, edge_dst = jnp.concatenate((edge_src, edge_dst)), jnp.concatenate(
            (edge_dst, edge_src)
        )
        d12 = jnp.concatenate((d12, d12))
        if "cells" in inputs:
            pbc_shifts = jnp.concatenate((pbc_shifts, -pbc_shifts))

        graph = inputs[self.graph_key] if self.graph_key in inputs else {}
        graph_out = {
            **graph,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "d12": d12,
            "overflow": overflow,
            "pbc_shifts": pbc_shifts,
        }
        if "nblist_skin" in state:
            graph_out["edge_src_skin"] = edge_src_skin
            graph_out["edge_dst_skin"] = edge_dst_skin
            if "cells" in inputs:
                graph_out["pbc_shifts_skin"] = pbc_shifts_skin

        return {**inputs, self.graph_key: graph_out}

    @partial(jax.jit, static_argnums=(0,))
    def update_skin(self, inputs):
        """ update the nblist without recomputing the full nblist """
        graph = inputs[self.graph_key]

        edge_src_skin = graph["edge_src_skin"]
        edge_dst_skin = graph["edge_dst_skin"]
        coords = inputs["coordinates"]
        vec = coords.at[edge_dst_skin].get(
            mode="fill", fill_value=self.cutoff
        ) - coords.at[edge_src_skin].get(mode="fill", fill_value=0.0)

        if "cells" in inputs:
            pbc_shifts_skin = graph["pbc_shifts_skin"]
            cells = inputs["cells"]
            if cells.shape[0] == 1:
                cells = cells[0].T
                vec = vec + jnp.dot(pbc_shifts_skin, cells)
            else:
                batch_index_vec = inputs["batch_index"][edge_src_skin]
                cells = jnp.swapaxes(cells, -2, -1)[batch_index_vec]
                vec = vec + jax.vmap(jnp.dot)(pbc_shifts_skin, cells)

        nat = coords.shape[0]
        d12 = jnp.sum(vec**2, axis=-1)
        mask = d12 < self.cutoff**2
        max_pairs = graph["edge_src"].shape[0] // 2
        (edge_src, edge_dst, d12), scatter_idx, npairs = mask_filter_1d(
            mask,
            max_pairs,
            (edge_src_skin, nat),
            (edge_dst_skin, nat),
            (d12, self.cutoff**2),
        )
        if "cells" in inputs:
            pbc_shifts = (
                jnp.full((max_pairs, 3), 0.0).at[scatter_idx].set(pbc_shifts_skin)
            )

        overflow = graph.get("overflow", False) | (npairs > max_pairs)
        graph_out = {
            **graph,
            "edge_src": jnp.concatenate((edge_src, edge_dst)),
            "edge_dst": jnp.concatenate((edge_dst, edge_src)),
            "d12": jnp.concatenate((d12, d12)),
            "overflow": overflow,
        }
        if "cells" in inputs:
            graph_out["pbc_shifts"] = jnp.concatenate((pbc_shifts, -pbc_shifts))

        return {**inputs, self.graph_key: graph_out}


class GraphProcessor(nn.Module):
    cutoff: float
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph = inputs[self.graph_key]
        coords = inputs["coordinates"]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        # edge_mask = edge_src < coords.shape[0]
        vec = coords.at[edge_dst].get(mode="fill", fill_value=self.cutoff) - coords.at[
            edge_src
        ].get(mode="fill", fill_value=0.0)
        if "cells" in inputs:
            cells = inputs["cells"]
            if cells.shape[0] == 1:
                cells = cells[0].T
                vec = vec + jnp.dot(graph["pbc_shifts"], cells)
            else:
                batch_index_vec = inputs["batch_index"][edge_src]
                cells = jnp.swapaxes(cells, -2, -1)[batch_index_vec]
                vec = vec + jax.vmap(jnp.dot)(graph["pbc_shifts"], cells)

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
    remove_hydrogens: int = False
    switch_params: FrozenDict = dataclasses.field(default_factory=FrozenDict)
    k_space: bool = False
    kmax: int = 30
    kthr: float = 1e-6
    mult_size: float = 1.05

    def init(self):
        return FrozenDict(
            {
                "npairs": 1,
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
        }

    def get_graph_properties(self):
        return {
            self.graph_key: {
                "cutoff": self.cutoff,
                "directed": True,
                "parent_graph": self.parent_graph,
            }
        }

    def __call__(self, state, inputs, return_state_update=False,add_margin=False):
        """ filter a nblist on cpu with numpy and dynamic shapes + store max shapes """
        graph_in = inputs[self.parent_graph]
        nat = inputs["species"].shape[0]

        new_state = {**state}
        state_up = {}

        edge_src = np.array(graph_in["edge_src"])
        d12 = np.array(graph_in["d12"])
        if self.remove_hydrogens:
            species = inputs["species"]
            src_idx = (edge_src < nat).nonzero()[0]  
            mask = np.zeros(edge_src.shape[0], dtype=bool)
            mask[src_idx] = (species > 1)[edge_src[src_idx]]
            d12 = np.where(mask, d12, self.cutoff**2)
        mask = d12 < self.cutoff**2

        max_pairs = state.get("npairs", 1)
        idx = np.nonzero(mask)[0]
        npairs = idx.shape[0]
        if npairs > max_pairs or add_margin:
            mult_size = state.get("nblist_mult_size", self.mult_size)
            prev_max_pairs = max_pairs
            max_pairs = int(mult_size * max(npairs,max_pairs)) + 1
            state_up["npairs"] = (max_pairs, prev_max_pairs)
            new_state["npairs"] = max_pairs

        filter_indices = np.full(max_pairs, edge_src.shape[0], dtype=np.int32)
        edge_src = np.full(max_pairs, nat, dtype=np.int32)
        edge_dst = np.full(max_pairs, nat, dtype=np.int32)
        d12_ = np.full(max_pairs, self.cutoff**2)
        filter_indices[:npairs] = idx
        edge_src[:npairs] = graph_in["edge_src"][idx]
        edge_dst[:npairs] = graph_in["edge_dst"][idx]
        d12_[:npairs] = d12[idx]
        d12 = d12_

        graph = inputs[self.graph_key] if self.graph_key in inputs else {}
        graph_out = {
            **graph,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "filter_indices": filter_indices,
            "d12": d12,
            "overflow": False,
        }

        output = {**inputs, self.graph_key: graph_out}
        if return_state_update:
            return FrozenDict(new_state), output, state_up
        return FrozenDict(new_state), output

    def check_reallocate(self, state, inputs, parent_overflow=False):
        """ check for overflow and reallocate nblist if necessary"""
        overflow = parent_overflow or inputs[self.graph_key].get("overflow", False)
        if not overflow:
            return state, {}, inputs, False

        add_margin = inputs[self.graph_key].get("overflow", False)
        state, inputs, state_up = self(state, inputs, return_state_update=True,add_margin=add_margin)
        return state, state_up, inputs, True

    @partial(jax.jit, static_argnums=(0, 1))
    def process(self, state, inputs):
        """ filter a nblist on accelerator with jax and precomputed shapes """
        graph_in = inputs[self.parent_graph]
        if state is None:
            # skin update mode
            graph = inputs[self.graph_key]
            max_pairs = graph["edge_src"].shape[0]
        else:
            max_pairs = state.get("npairs", 1)

        max_pairs_in = graph_in["edge_src"].shape[0]
        nat = inputs["species"].shape[0]

        edge_src = graph_in["edge_src"]
        d12 = graph_in["d12"]
        if self.remove_hydrogens:
            species = inputs["species"]
            mask = (species > 1)[edge_src]
            d12 = jnp.where(mask, d12, self.cutoff**2)
        mask = d12 < self.cutoff**2

        (edge_src, edge_dst, d12, filter_indices), _, npairs = mask_filter_1d(
            mask,
            max_pairs,
            (edge_src, nat),
            (graph_in["edge_dst"], nat),
            (d12, self.cutoff**2),
            (jnp.arange(max_pairs_in, dtype=jnp.int32), max_pairs_in),
        )

        graph = inputs[self.graph_key] if self.graph_key in inputs else {}
        overflow = graph.get("overflow", False) | (npairs > max_pairs)
        graph_out = {
            **graph,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "filter_indices": filter_indices,
            "d12": d12,
            "overflow": overflow,
        }

        return {**inputs, self.graph_key: graph_out}

    @partial(jax.jit, static_argnums=(0,))
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

        if graph_in["vec"].shape[0] == 0:
            vec = graph_in["vec"]
            distances = graph_in["distances"]
            filter_indices = jnp.asarray([], dtype=jnp.int32)
        else:
            filter_indices = graph["filter_indices"]
            vec = (
                graph_in["vec"]
                .at[filter_indices]
                .get(mode="fill", fill_value=self.cutoff)
            )
            distances = (
                graph_in["distances"]
                .at[filter_indices]
                .get(mode="fill", fill_value=self.cutoff)
            )

        edge_mask = distances < self.cutoff
        switch = SwitchFunction(
            **{**self.switch_params, "cutoff": self.cutoff, "graph_key": None}
        )((distances, edge_mask))

        graph_out = {
            **graph,
            "vec": vec,
            "distances": distances,
            "switch": switch,
            "filter_indices": filter_indices,
            # "edge_mask": edge_mask,
        }
        return {**inputs, self.graph_key: graph_out}


@dataclasses.dataclass(frozen=True)
class GraphAngularExtension:
    mult_size: float = 1.05
    add_neigh: int = 5
    graph_key: str = "graph"

    def init(self):
        return FrozenDict(
            {
                "nangles": 0,
                "nblist_mult_size": self.mult_size,
                "max_neigh": self.add_neigh,
                "add_neigh": self.add_neigh,
            }
        )

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphAngleProcessor, {
            "graph_key": self.graph_key,
            "name": f"{self.graph_key}_AngleProcessor",
        }

    def get_graph_properties(self):
        return {
            self.graph_key: {
                "has_angles": True,
            }
        }

    def __call__(self, state, inputs, return_state_update=False,add_margin=False):
        """ build angle nblist on cpu with numpy and dynamic shapes + store max shapes """
        graph = inputs[self.graph_key]
        edge_src = np.array(graph["edge_src"])

        new_state = {**state}
        state_up = {}

        ### count number of neighbors
        nat = inputs["species"].shape[0]
        count = np.zeros(nat + 1, dtype=np.int32)
        np.add.at(count, edge_src, 1)
        max_count = np.max(count[:-1])

        ### get sizes
        max_neigh = state.get("max_neigh", self.add_neigh)
        if max_count > max_neigh or add_margin:
            prev_max_neigh = max_neigh
            max_neigh = max(max_count,max_neigh) + state.get("add_neigh", self.add_neigh)
            state_up["max_neigh"] = (max_neigh, prev_max_neigh)
            new_state["max_neigh"] = max_neigh

        max_neigh_arr = np.empty(max_neigh, dtype=bool)

        nedge = edge_src.shape[0]

        ### sort edge_src
        idx_sort = np.argsort(edge_src)
        edge_src_sorted = edge_src[idx_sort]

        ### map sparse to dense nblist
        offset = np.tile(np.arange(max_count), nat)
        if max_count * nat >= nedge:
            offset = np.tile(np.arange(max_count), nat)[:nedge]
        else:
            offset = np.zeros(nedge, dtype=jnp.int32)
            offset[: max_count * nat] = np.tile(np.arange(max_count), nat)

        # offset = jnp.where(edge_src_sorted < nat, offset, 0)
        mask = edge_src_sorted < nat
        indices = edge_src_sorted * max_count + offset
        indices = indices[mask]
        idx_sort = idx_sort[mask]
        edge_idx = np.full(nat * max_count, nedge, dtype=np.int32)
        edge_idx[indices] = idx_sort
        edge_idx = edge_idx.reshape(nat, max_count)

        ### find all triplet for each atom center
        local_src, local_dst = np.triu_indices(max_count, 1)
        angle_src = edge_idx[:, local_src].flatten()
        angle_dst = edge_idx[:, local_dst].flatten()

        ### mask for valid angles
        mask1 = angle_src < nedge
        mask2 = angle_dst < nedge
        angle_mask = mask1 & mask2

        max_angles = state.get("nangles", 0)
        idx = np.nonzero(angle_mask)[0]
        nangles = idx.shape[0]
        if nangles > max_angles or add_margin:
            mult_size = state.get("nblist_mult_size", self.mult_size)
            max_angles_prev = max_angles
            max_angles = int(mult_size * max(nangles,max_angles)) + 1
            state_up["nangles"] = (max_angles, max_angles_prev)
            new_state["nangles"] = max_angles

        ## filter angles to sparse representation
        angle_src_ = np.full(max_angles, nedge, dtype=np.int32)
        angle_dst_ = np.full(max_angles, nedge, dtype=np.int32)
        angle_src_[:nangles] = angle_src[idx]
        angle_dst_[:nangles] = angle_dst[idx]

        central_atom = np.full(max_angles, nat, dtype=np.int32)
        central_atom[:nangles] = edge_src[angle_src_[:nangles]]

        ## update graph
        output = {
            **inputs,
            self.graph_key: {
                **graph,
                "angle_src": angle_src_,
                "angle_dst": angle_dst_,
                "central_atom": central_atom,
                "angle_overflow": False,
                "max_neigh": max_neigh,
                "__max_neigh_array": max_neigh_arr,
            },
        }

        if return_state_update:
            return FrozenDict(new_state), output, state_up
        return FrozenDict(new_state), output

    def check_reallocate(self, state, inputs, parent_overflow=False):
        """ check for overflow and reallocate nblist if necessary """
        overflow = parent_overflow or inputs[self.graph_key]["angle_overflow"]
        if not overflow:
            return state, {}, inputs, False

        add_margin = inputs[self.graph_key]["angle_overflow"]
        state, inputs, state_up = self(state, inputs, return_state_update=True,add_margin=add_margin)
        return state, state_up, inputs, True

    @partial(jax.jit, static_argnums=(0, 1))
    def process(self, state, inputs):
        """ build angle nblist on accelerator with jax and precomputed shapes """
        graph = inputs[self.graph_key]
        edge_src = graph["edge_src"]

        ### count number of neighbors
        nat = inputs["species"].shape[0]
        count = jnp.zeros(nat, dtype=jnp.int32).at[edge_src].add(1, mode="drop")
        max_count = jnp.max(count)

        ### get sizes
        if state is None:
            max_neigh_arr = graph["__max_neigh_array"]
            max_neigh = max_neigh_arr.shape[0]
            prev_nangles = graph["angle_src"].shape[0]
        else:
            max_neigh = state.get("max_neigh", self.add_neigh)
            max_neigh_arr = jnp.empty(max_neigh, dtype=bool)
            prev_nangles = state.get("nangles", 0)

        nedge = edge_src.shape[0]

        ### sort edge_src
        idx_sort = jnp.argsort(edge_src)
        edge_src_sorted = edge_src[idx_sort]

        ### map sparse to dense nblist
        offset = jnp.tile(jnp.arange(max_neigh), nat)
        if max_neigh * nat >= nedge:
            offset = offset[:nedge]
        else:
            offset = jax.lax.dynamic_update_slice(
                jnp.zeros(nedge, dtype=jnp.int32), offset, (0,)
            )
        # offset = jnp.where(edge_src_sorted < nat, offset, 0)
        indices = edge_src_sorted * max_neigh + offset
        edge_idx = (
            jnp.full(nat * max_neigh, nedge, dtype=jnp.int32)
            .at[indices]
            .set(idx_sort, mode="drop")
            .reshape(nat, max_neigh)
        )

        ### find all triplet for each atom center
        local_src, local_dst = np.triu_indices(max_neigh, 1)
        angle_src = edge_idx[:, local_src].flatten()
        angle_dst = edge_idx[:, local_dst].flatten()

        ### mask for valid angles
        mask1 = angle_src < nedge
        mask2 = angle_dst < nedge
        angle_mask = mask1 & mask2

        ## filter angles to sparse representation
        (angle_src, angle_dst), _, nangles = mask_filter_1d(
            angle_mask,
            prev_nangles,
            (angle_src, nedge),
            (angle_dst, nedge),
        )
        ## find central atom
        central_atom = edge_src[angle_src]

        ## check for overflow
        angle_overflow = nangles > prev_nangles
        neigh_overflow = max_count > max_neigh
        overflow = graph.get("angle_overflow", False) | angle_overflow | neigh_overflow

        ## update graph
        output = {
            **inputs,
            self.graph_key: {
                **graph,
                "angle_src": angle_src,
                "angle_dst": angle_dst,
                "central_atom": central_atom,
                "angle_overflow": overflow,
                "max_neigh": max_neigh,
                "__max_neigh_array": max_neigh_arr,
            },
        }

        return output

    @partial(jax.jit, static_argnums=(0,))
    def update_skin(self, inputs):
        return self.process(None, inputs)


class GraphAngleProcessor(nn.Module):
    graph_key: str

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        vec = graph["vec"]
        angle_src = graph["angle_src"]
        angle_dst = graph["angle_dst"]

        d1 = distances.at[angle_src].get(mode="fill", fill_value=1.0)
        d2 = distances.at[angle_dst].get(mode="fill", fill_value=1.0)
        vec1 = vec.at[angle_src].get(mode="fill", fill_value=1.0)
        vec2 = vec.at[angle_dst].get(mode="fill", fill_value=0.0)

        cos_angles = (vec1 * vec2).sum(axis=-1) / jnp.clip(d1 * d2, a_min=1e-10)
        angles = jnp.arccos(0.95 * cos_angles)

        return {
            **inputs,
            self.graph_key: {
                **graph,
                # "cos_angles": cos_angles,
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
    species = inputs["species"].astype(np.int32)
    ifake = np.argmax(species <= 0)
    if ifake > 0:
        assert np.all(species[:ifake] > 0), "species must be positive"
    nat = inputs["species"].shape[0]

    natoms = inputs.get("natoms", np.array([nat], dtype=np.int32)).astype(np.int32)
    batch_index = inputs.get(
        "batch_index", np.repeat(np.arange(len(natoms), dtype=np.int32), natoms)
    ).astype(np.int32)
    output = {**inputs, "natoms": natoms, "batch_index": batch_index}
    if "cells" in inputs:
        cells = inputs["cells"]
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
            s, inputs = self.atom_padder(layer_state[0], inputs)
            new_state.append(s)
            i += 1
        for layer in self.layers:
            s, inputs = layer(layer_state[i], inputs, return_state_update=False)
            new_state.append(s)
            i += 1
        return FrozenDict({**state, "layers_state": tuple(new_state)}), convert_to_jax(
            inputs
        )

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
            s, s_up, inputs, parent_overflow = layer.check_reallocate(
                layer_state[i], inputs, parent_overflow
            )
            new_state.append(s)
            state_up.append(s_up)
            i += 1

        if not parent_overflow:
            return state, {}, inputs, False
        return (
            FrozenDict({**state, "layers_state": tuple(new_state)}),
            state_up,
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
