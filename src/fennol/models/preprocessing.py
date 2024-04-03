import flax.linen as nn
from typing import Sequence, Callable, Union, Dict, Any
import jax.numpy as jnp
import jax
import numpy as np
from typing import Optional, Tuple
import numba
import dataclasses

from ..utils.activations import chain
from ..utils import deep_update
from .modules import FENNIXModules
from .misc.misc import SwitchFunction
from ..utils.nblist import (
    compute_nblist_flatbatch,
    angular_nblist,
    compute_nblist_fixed,
    compute_nblist_flatbatch_minimage,
    compute_nblist_flatbatch_fullpbc,
    compute_nblist_fixed_minimage,
    get_reciprocal_space_parameters,
    compute_nblist_ase,
    compute_nblist_ase_pbc,
)


def minmaxone(x, name=""):
    print(name, x.min(), x.max(), (x**2).mean())


@dataclasses.dataclass(frozen=True)
class GraphGenerator:
    cutoff: float
    mult_size: float = 1.1
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict)
    kmax: int = 30
    kthr: float = 1e-6
    k_space: bool = False
    # covalent_cutoff: bool = False

    def init(self):
        return {
            "prev_nblist_size": 1,
            "nblist_mult_size": self.mult_size,
        }

    def __call__(self, inputs: Dict, state={}):
        cutoff = self.cutoff  # + state.get("nblist_skin", 0.0)

        mult_size = float(state.get("nblist_mult_size", self.mult_size))

        coords = np.array(inputs["coordinates"])
        batch_index = np.array(inputs["batch_index"])
        natoms = np.array(inputs["natoms"])
        padding_value = coords.shape[0]
        if "true_atoms" in inputs:
            true_atoms = np.array(inputs["true_atoms"], dtype=bool)
            true_sys = np.array(inputs["true_sys"], dtype=bool)
            coords = coords[true_atoms]
            batch_index = batch_index[true_atoms]
            natoms = natoms[true_sys]

        prev_nblist_size = state.get("prev_nblist_size", 0)

        ase_nblist = inputs.get("ase_nblist", False)
        if "cells" in inputs:
            cells = np.array(inputs["cells"], dtype=coords.dtype)
            if cells.ndim == 2:
                cells = cells[None, :, :]
            reciprocal_cells = np.linalg.inv(cells)
            minimage = inputs.get("minimum_image", False)
            if ase_nblist:
                compute_nblist = compute_nblist_ase_pbc
            elif minimage:
                compute_nblist = compute_nblist_flatbatch_minimage
            else:
                compute_nblist = compute_nblist_flatbatch_fullpbc

            (
                edge_src,
                edge_dst,
                d12,
                pbc_shifts,
                prev_nblist_size_,
                isym,
            ) = compute_nblist(
                coords,
                cutoff,
                batch_index,
                natoms,
                mult_size,
                cells,
                reciprocal_cells,
                prev_nblist_size,
                padding_value=padding_value,
            )
        else:
            if ase_nblist:
                compute_nblist = compute_nblist_ase
            else:
                compute_nblist = compute_nblist_flatbatch
            edge_src, edge_dst, d12, prev_nblist_size_, isym = compute_nblist(
                coords,
                cutoff,
                batch_index,
                natoms,
                mult_size,
                prev_nblist_size,
                padding_value=padding_value,
            )
            pbc_shifts = None

        state = {**state, "prev_nblist_size": prev_nblist_size_}

        out = {
            **inputs,
            self.graph_key: {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "cutoff": self.cutoff,
                "pbc_shifts": pbc_shifts,
                "overflow": False,
                # "isym": isym,
            },
        }
        if "cells" in inputs:
            out["cells"] = cells
            out["reciprocal_cells"] = reciprocal_cells
            if self.k_space:
                in_graph = inputs.get(self.graph_key, {})
                if "k_points" in in_graph:
                    ks = in_graph["k_points"]
                    bewald = in_graph["b_ewald"]
                else:
                    ks, _, _, bewald = get_reciprocal_space_parameters(
                        reciprocal_cells, self.cutoff, self.kmax, self.kthr
                    )
                out[self.graph_key]["k_points"] = ks
                out[self.graph_key]["b_ewald"] = bewald

        return out, state

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "switch_params": self.switch_params,
            "name": f"{self.graph_key}_Processor",
        }

    def get_updater(self) -> Tuple[nn.Module, Dict]:
        raise NotImplementedError(
            "GraphGenerator does not have an updater. Use GraphGeneratorFixed instead."
        )

    def get_skin_updater(self):
        raise NotImplementedError(
            "GraphGenerator does not have a skin updater. Use GraphGeneratorFixed instead."
        )

    def get_graph_properties(self):
        return {self.graph_key: {"cutoff": self.cutoff, "directed": True}}


@dataclasses.dataclass(frozen=True)
class GraphGeneratorFixed:
    cutoff: float
    mult_size: float = 1.1
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict)
    kmax: int = 30
    kthr: float = 1e-6
    k_space: bool = False

    def init(self):
        return {
            "prev_nblist_size": 1,
            "nblist_mult_size": self.mult_size,
        }

    def __call__(self, inputs: Dict, state={}) -> Union[dict, jax.Array]:
        cutoff = self.cutoff + inputs.get("nblist_skin", 0.0)

        mult_size = float(state.get("nblist_mult_size", self.mult_size))

        coords = inputs["coordinates"]
        batch_index = inputs["batch_index"]
        natoms = inputs["natoms"]
        padding_value = coords.shape[0]
        if "true_atoms" in inputs:
            true_atoms = inputs["true_atoms"]
            true_sys = inputs["true_sys"]
            coords = coords[true_atoms]
            batch_index = batch_index[true_atoms]
            natoms = natoms[true_sys]

        prev_nblist_size = state.get("prev_nblist_size", 0)
        prev_nblist_size_ = prev_nblist_size
        max_nat = int(np.max(natoms))

        if "cells" in inputs:
            cells = np.asarray(inputs["cells"], dtype=coords.dtype)
            if cells.ndim == 2:
                cells = cells[None, :, :]
            reciprocal_cells = np.linalg.inv(cells)

            minimage = inputs.get("minimum_image", True)
            assert minimage, "Fixed nblist only works with minimum image convention"

            edge_src, edge_dst, d12, pbc_shifts, npairs, p12 = (
                compute_nblist_fixed_minimage(
                    coords,
                    cutoff,
                    batch_index,
                    natoms,
                    max_nat,
                    prev_nblist_size_,
                    padding_value,
                    cells,
                    reciprocal_cells,
                )
            )
        else:
            edge_src, edge_dst, d12, npairs, p12 = compute_nblist_fixed(
                coords,
                cutoff,
                batch_index,
                natoms,
                max_nat,
                prev_nblist_size_,
                padding_value,
            )

        if npairs > prev_nblist_size:
            prev_nblist_size_ = int(mult_size * npairs) + 1

        edge_src = edge_src[:prev_nblist_size_]
        edge_dst = edge_dst[:prev_nblist_size_]
        d12 = d12[:prev_nblist_size_]
        if "cells" in inputs:
            pbc_shifts = pbc_shifts[:prev_nblist_size_]

        if "nblist_skin" in inputs:
            edge_src_skin = edge_src
            edge_dst_skin = edge_dst
            if "cells" in inputs:
                pbc_shifts_skin = pbc_shifts
            mask = d12 < self.cutoff**2
            npairs_skin = mask.sum()
            prev_nblist_size_skin = state.get("prev_nblist_size_skin", 0)
            if npairs_skin > prev_nblist_size_skin:
                prev_nblist_size_skin = int(mult_size * npairs_skin) + 1
            state["prev_nblist_size_skin"] = prev_nblist_size_skin
            mask = mask[:prev_nblist_size_skin]
            edge_src = np.where(
                mask, edge_src_skin[:prev_nblist_size_skin], coords.shape[0]
            )
            edge_dst = np.where(
                mask, edge_dst_skin[:prev_nblist_size_skin], coords.shape[0]
            )
            d12 = np.where(mask, d12[:prev_nblist_size_skin], self.cutoff**2)

            if "cells" in inputs:
                pbc_shifts = np.where(
                    mask[:, None], pbc_shifts_skin[:prev_nblist_size_skin], 0.0
                )
        # iedge = np.arange(len(edge_src))
        # isym = np.concatenate((iedge + iedge.shape[0], iedge))
        edge_src, edge_dst = np.concatenate((edge_src, edge_dst)), np.concatenate(
            (edge_dst, edge_src)
        )

        d12 = np.concatenate((d12, d12))
        if "cells" in inputs:
            pbc_shifts = np.concatenate((pbc_shifts, -pbc_shifts))
        else:
            pbc_shifts = None

        state["prev_nblist_size"] = prev_nblist_size_

        out = {
            **inputs,
            self.graph_key: {
                "p12": p12,
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "cutoff": self.cutoff,
                "pbc_shifts": pbc_shifts,
                "overflow": False,
                # "isym": isym,
            },
        }
        if "nblist_skin" in inputs:
            out[self.graph_key]["edge_src_skin"] = edge_src_skin
            out[self.graph_key]["edge_dst_skin"] = edge_dst_skin
            if "cells" in inputs:
                out[self.graph_key]["pbc_shifts_skin"] = pbc_shifts_skin

        if "cells" in inputs:
            out["cells"] = cells
            out["reciprocal_cells"] = reciprocal_cells
            if self.k_space:
                in_graph = inputs.get(self.graph_key, {})
                if "k_points" in in_graph:
                    ks = in_graph["k_points"]
                    bewald = in_graph["b_ewald"]
                else:
                    print("Computing k-points", self.cutoff, self.kmax, self.kthr)
                    ks, _, _, bewald = get_reciprocal_space_parameters(
                        reciprocal_cells, self.cutoff, self.kmax, self.kthr
                    )
                    print("n k-points", ks.shape)
                out[self.graph_key]["k_points"] = ks
                out[self.graph_key]["b_ewald"] = bewald

        return out, state

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "switch_params": self.switch_params,
            "name": f"{self.graph_key}_Processor",
        }

    def get_updater(self):
        return get_graph_updater(
            **{
                "cutoff": self.cutoff,
                "graph_key": self.graph_key,
                "switch_params": self.switch_params,
            }
        )

    def get_skin_updater(self):
        return get_graph_skin_updater(
            **{
                "cutoff": self.cutoff,
                "graph_key": self.graph_key,
                "switch_params": self.switch_params,
            }
        )

    def get_graph_properties(self):
        return {self.graph_key: {"cutoff": self.cutoff, "directed": True}}


# class GraphUpdater(nn.Module):
#     cutoff: float
#     graph_key: str = "graph"
#     switch_params: dict = dataclasses.field(default_factory=dict)

#     @nn.compact
#     def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):


def get_graph_updater(cutoff, graph_key, switch_params):
    def graph_updater(inputs):
        graph = {**inputs[graph_key]}

        coords = inputs["coordinates"]
        if "p12" in graph:
            p1, p2, mask_p12 = graph["p12"]
        else:
            assert inputs["natoms"].shape[0] == 1, "Only one system is supported if p12 is not present"
            p1,p2=jnp.triu_indices(coords.shape[0], 1)
            mask_p12 = None
            
        if "nblist_skin" in inputs:
            max_pairs = graph["edge_src_skin"].shape[0]
        else:
            max_pairs = graph["edge_src"].shape[0] // 2
            
        vec = coords[p2] - coords[p1]
        if "cells" in inputs:
            batch_indexvec = inputs["batch_index"][p1]
            cells = inputs["cells"]
            reciprocal_cells = inputs["reciprocal_cells"]
            vecpbc = jnp.einsum("sij,sj->si", reciprocal_cells[batch_indexvec], vec)
            pbc_shifts = -jnp.round(vecpbc)
            vec = vec + jnp.einsum("sij,sj->si", cells[batch_indexvec], pbc_shifts)
        # vec = jnp.where((p1 < coords.shape[0])[:,None], vec, cutoff)
        d12 = jnp.sum(vec**2, axis=-1)
        cutoff_skin = cutoff + inputs.get("nblist_skin", 0.0)
        if mask_p12 is not None:
            d12 = jnp.where(mask, d12, cutoff_skin**2)
        mask = d12 < cutoff_skin**2
        npairs = mask.sum()
        overflow = npairs > max_pairs
        idx = jnp.argsort(d12)[:max_pairs]
        mask = mask[idx]
        edge_src = p1[idx]
        edge_dst = p2[idx]
        d12 = d12[idx]
        if "cells" in inputs:
            pbc_shifts = pbc_shifts[idx]
        # edge_src = jnp.where(mask, p1[idx][:max_pairs], coords.shape[0])
        # edge_dst = jnp.where(mask, p2[idx][:max_pairs], coords.shape[0])
        # d12 = jnp.where(mask, d12[idx][:max_pairs], cutoff_skin**2)
        # if "cells" in inputs:
        #     pbc_shifts = jnp.where(mask[:, None], pbc_shifts[idx][:max_pairs], 0)

        d12_padding = cutoff_skin**2
        if "nblist_skin" in inputs:
            edge_src_skin = edge_src
            edge_dst_skin = edge_dst
            if "cells" in inputs:
                pbc_shifts_skin = pbc_shifts
            max_pairs_skin = graph["edge_src"].shape[0] // 2
            mask = d12 < cutoff**2
            npairs_skin = mask.sum()
            mask = mask[:max_pairs_skin]
            edge_src = edge_src_skin[:max_pairs_skin]
            edge_dst = edge_dst_skin[:max_pairs_skin]
            d12 = d12[:max_pairs_skin]
            if "cells" in inputs:
                pbc_shifts = pbc_shifts_skin[:max_pairs_skin]
            # edge_src = jnp.where(mask, edge_src_skin[:max_pairs_skin], coords.shape[0])
            # edge_dst = jnp.where(mask, edge_dst_skin[:max_pairs_skin], coords.shape[0])
            # d12 = jnp.where(mask, d12[:max_pairs_skin], cutoff**2)
            # if "cells" in inputs:
            #     pbc_shifts = jnp.where(
            #         mask[:, None], pbc_shifts_skin[:max_pairs_skin], 0
            #     )
            overflow = jnp.logical_or(overflow, npairs_skin > max_pairs_skin)
            d12_padding = cutoff**2


        edge_src = jnp.where(mask, edge_src, coords.shape[0])
        edge_dst = jnp.where(mask, edge_dst, coords.shape[0])
        d12 = jnp.where(mask, d12, d12_padding)
        if "cells" in inputs:
            pbc_shifts = jnp.where(mask[:, None], pbc_shifts, 0)

        edge_src, edge_dst = jnp.concatenate((edge_src, edge_dst)), jnp.concatenate(
            (edge_dst, edge_src)
        )
        d12 = jnp.concatenate((d12, d12))
        if "cells" in inputs:
            pbc_shifts = jnp.concatenate((pbc_shifts, -pbc_shifts))

        graph_out = {
            **graph,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "d12": d12,
            "overflow": overflow,
        }
        if "nblist_skin" in inputs:
            graph_out["edge_src_skin"] = edge_src_skin
            graph_out["edge_dst_skin"] = edge_dst_skin
            if "cells" in inputs:
                graph_out["pbc_shifts_skin"] = pbc_shifts_skin

        if "cells" in inputs:
            graph_out["pbc_shifts"] = pbc_shifts
        return {**inputs, graph_key: graph_out}

    return graph_updater


def get_graph_skin_updater(cutoff, graph_key, switch_params):
    def graph_updater(inputs):
        graph = inputs[graph_key]
        natoms = inputs["species"].shape[0]
        max_pairs = graph["edge_src"].shape[0] // 2

        edge_src_skin = graph["edge_src_skin"]
        edge_dst_skin = graph["edge_dst_skin"]
        coords = inputs["coordinates"]
        vec = coords.at[edge_dst_skin].get(mode="fill", fill_value=cutoff) - coords.at[
            edge_src_skin
        ].get(mode="fill", fill_value=0.0)
        if "cells" in inputs:
            pbc_shifts_skin = graph["pbc_shifts_skin"]
            batch_indexvec = (
                inputs["batch_index"].at[edge_src_skin].get(mode="fill", fill_value=-1)
            )
            # reciprocal_cells = inputs["reciprocal_cells"][batch_indexvec]
            # vecpbc = jnp.einsum("sij,sj->si", reciprocal_cells, vec)
            # pbc_shifts_skin = -jnp.round(vecpbc)
            cells = inputs["cells"][batch_indexvec]
            vec = vec + jnp.einsum("sij,sj->si", cells, pbc_shifts_skin)
        d12 = jnp.sum(vec**2, axis=-1)
        mask = d12 < cutoff**2
        npairs = mask.sum()

        filter_indices = jnp.argsort(d12)[:max_pairs]

        mask = mask[filter_indices]
        edge_src = jnp.where(mask, edge_src_skin[filter_indices], natoms)
        edge_dst = jnp.where(mask, edge_dst_skin[filter_indices], natoms)
        d12 = jnp.where(mask, d12[filter_indices], cutoff**2)
        if "cells" in inputs:
            pbc_shifts = jnp.where(mask[:, None], pbc_shifts_skin[filter_indices], 0.0)

        # edge_src = edge_src[filter_indices]
        # edge_dst = graph_in["edge_dst"][filter_indices]
        # d12 = d12[filter_indices]

        overflow = npairs > max_pairs

        new_graph = {
            **graph,
            "edge_src": jnp.concatenate((edge_src, edge_dst)),
            "edge_dst": jnp.concatenate((edge_dst, edge_src)),
            "d12": jnp.concatenate((d12, d12)),
            "overflow": jnp.logical_or(overflow, graph.get("overflow", False)),
        }
        if "cells" in inputs:
            new_graph["pbc_shifts"] = jnp.concatenate((pbc_shifts, -pbc_shifts))
        return {**inputs, graph_key: new_graph}

    return graph_updater


class GraphProcessor(nn.Module):
    cutoff: float
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph = inputs[self.graph_key]
        coords = inputs["coordinates"]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        edge_mask = edge_src < coords.shape[0]
        vec = coords.at[edge_dst].get(mode="fill", fill_value=self.cutoff) - coords.at[
            edge_src
        ].get(mode="fill", fill_value=0.0)
        if "cells" in inputs:
            batch_indexvec = (
                inputs["batch_index"].at[edge_src].get(mode="fill", fill_value=-1)
            )
            cells = inputs["cells"][batch_indexvec]
            vec = vec + jnp.einsum("sij,sj->si", cells, graph["pbc_shifts"])
            # vec = vec + graph["pbc_shifts"]
        distances = jnp.linalg.norm(vec, axis=-1)
        # edge_mask = distances < self.cutoff

        switch, edge_mask = SwitchFunction(
            **{**self.switch_params, "cutoff": self.cutoff, "graph_key": None}
        )(distances)

        graph_out = {
            **graph,
            "vec": vec,
            "distances": distances,
            "switch": switch,
            "edge_mask": edge_mask,
        }
        return {**inputs, self.graph_key: graph_out}


@dataclasses.dataclass(frozen=True)
class GraphFilter:
    cutoff: float
    graph_out: str
    mult_size: float = 1.1
    graph_key: str = "graph"
    remove_hydrogens: int = False
    switch_params: dict = dataclasses.field(default_factory=dict)
    k_space: bool = False
    kmax: int = 30
    kthr: float = 1e-6

    def init(self):
        return {"prev_nblist_size": 1, "nblist_mult_size": self.mult_size}

    def __call__(self, inputs: Dict, state={}):
        c2 = self.cutoff**2

        graph_in = inputs[self.graph_key]
        d12 = graph_in["d12"]
        edge_src_in = graph_in["edge_src"]
        nblist_size_in = edge_src_in.shape[0]
        mask = d12 < c2
        indices = mask.nonzero()[0]
        edge_src = edge_src_in[indices]
        edge_dst = graph_in["edge_dst"][indices]
        d12 = d12[indices]
        if self.remove_hydrogens:
            species = np.array(inputs["species"])
            mask = species[edge_src] > 1
            edge_src = edge_src[mask]
            edge_dst = edge_dst[mask]
            d12 = d12[mask]
            indices = indices[mask]
            isym = None
        else:
            isym = graph_in.get("isym", None)
            if isym is not None:
                isym = isym[indices]

        prev_nblist_size = state.get("prev_nblist_size", 0)
        prev_nblist_size_ = prev_nblist_size

        mult_size = float(state.get("nblist_mult_size", self.mult_size))

        nattot = inputs["species"].shape[0]
        nblist_size = edge_src.shape[0]
        if nblist_size > prev_nblist_size_:
            prev_nblist_size_ = int(mult_size * nblist_size) +1

        edge_src = np.append(
            edge_src, nattot * np.ones(prev_nblist_size_ - nblist_size, dtype=np.int64)
        )
        edge_dst = np.append(
            edge_dst, nattot * np.ones(prev_nblist_size_ - nblist_size, dtype=np.int64)
        )
        d12 = np.append(
            d12, c2 * np.ones(prev_nblist_size_ - nblist_size, dtype=np.float32)
        )
        indices = np.append(
            indices,
            nblist_size_in * np.ones(prev_nblist_size_ - nblist_size, dtype=np.int64),
        )
        if isym is not None:
            isym = np.append(
                isym, -1 * np.ones(prev_nblist_size_ - nblist_size, dtype=np.int32)
            )

        state["prev_nblist_size"] = prev_nblist_size_

        out = {
            **inputs,
            self.graph_out: {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "filter_indices": indices,
                "cutoff": self.cutoff,
                "overflow": False,
            },
        }
        if "cells" in inputs and self.k_space:
            reciprocal_cells = np.array(inputs["reciprocal_cells"])
            in_graph = inputs.get(self.graph_out, {})
            if "k_points" in in_graph:
                ks = in_graph["k_points"]
                bewald = in_graph["b_ewald"]
            else:
                ks, _, _, bewald = get_reciprocal_space_parameters(
                    reciprocal_cells, self.cutoff, self.kmax, self.kthr
                )
            out[self.graph_out]["k_points"] = ks
            out[self.graph_out]["b_ewald"] = bewald

        if isym is not None:
            out[self.graph_out]["isym"] = isym

        return out, state

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphFilterProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "graph_out": self.graph_out,
            "name": f"{self.graph_out}_Filter_{self.graph_key}",
            "switch_params": self.switch_params,
        }

    def get_updater(self):
        return get_graph_filter_updater(
            **{
                "cutoff": self.cutoff,
                "graph_key": self.graph_key,
                "graph_out": self.graph_out,
                "remove_hydrogens": self.remove_hydrogens,
            }
        )

    def get_skin_updater(self):
        return self.get_updater()

    def get_graph_properties(self):
        return {
            self.graph_out: {
                "cutoff": self.cutoff,
                "directed": True,
                "original_graph": self.graph_key,
            }
        }


def get_graph_filter_updater(cutoff, graph_out, graph_key, remove_hydrogens):
    def graph_filter_updater(inputs):
        graph_in = inputs[graph_key]
        graph = inputs[graph_out]
        natoms = inputs["species"].shape[0]
        max_pairs = graph["edge_src"].shape[0]

        edge_src = graph_in["edge_src"]
        d12 = graph_in["d12"]
        if remove_hydrogens:
            species = inputs["species"]
            mask = species[edge_src] > 1
            d12 = jnp.where(mask, d12, (cutoff + 1) ** 2)
        mask = d12 < cutoff**2
        npairs = mask.sum()

        filter_indices = jnp.argsort(d12)[:max_pairs]

        mask = mask[filter_indices]
        edge_src = jnp.where(mask, edge_src[filter_indices], natoms)
        edge_dst = jnp.where(mask, graph_in["edge_dst"][filter_indices], natoms)
        d12 = jnp.where(mask, d12[filter_indices], cutoff**2)
        filter_indices = jnp.where(mask, filter_indices, graph_in["edge_src"].shape[0])

        # edge_src = edge_src[filter_indices]
        # edge_dst = graph_in["edge_dst"][filter_indices]
        # d12 = d12[filter_indices]

        overflow = npairs > max_pairs

        new_graph = {
            **graph,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "d12": d12,
            "filter_indices": filter_indices,
            "overflow": overflow,
            "filter_overflow":(npairs, max_pairs),
        }
        return {**inputs, graph_out: new_graph}

    return graph_filter_updater


class GraphFilterProcessor(nn.Module):
    cutoff: float
    graph_out: str
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph_in = inputs[self.graph_key]
        graph = inputs[self.graph_out]

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
        # edge_mask = distances < self.cutoff

        # switch = jnp.where(
        #     edge_mask, 0.5 * jnp.cos(distances * (jnp.pi / self.cutoff)) + 0.5, 0.0
        # )
        switch, edge_mask = SwitchFunction(
            **{**self.switch_params, "cutoff": self.cutoff, "graph_key": None}
        )(distances)

        graph_out = {
            **graph,
            "vec": vec,
            "distances": distances,
            "switch": switch,
            "edge_mask": edge_mask,
            "filter_indices": filter_indices,
        }
        return {**inputs, self.graph_out: graph_out}


@dataclasses.dataclass(frozen=True)
class GraphAngularExtension:
    mult_size: float = 1.1
    graph_key: str = "graph"
    max_neigh_add: int = 10

    def init(self):
        return {
            "prev_angle_size": 0,
            "max_neigh": 0,
            "nblist_mult_size": self.mult_size,
            "max_neigh_add": self.max_neigh_add,
        }

    def __call__(self, inputs: Dict, state={}) -> Union[dict, jax.Array]:

        graph = inputs[self.graph_key]

        edge_src = np.array(graph["edge_src"])
        nattot = inputs["species"].shape[0]
        # edge_src = edge_src[edge_src < nattot]

        central_atom_index, angle_src, angle_dst, max_neigh = angular_nblist(
            edge_src, nattot
        )

        prev_angle_size = state.get("prev_angle_size", 0)
        mult_size = float(state.get("nblist_mult_size", self.mult_size))

        prev_angle_size_ = prev_angle_size
        angle_size = angle_src.shape[0]
        if angle_size > prev_angle_size_:
            prev_angle_size_ = int(mult_size * angle_size) +1

        prev_max_neigh = state.get("max_neigh", 0)
        if max_neigh > prev_max_neigh:
            max_neigh_add = int(state.get("max_neigh_add", self.max_neigh_add))
            prev_max_neigh = max_neigh + max_neigh_add

        max_neigh_array = np.empty(prev_max_neigh, dtype=np.int8)

        nblist_size = edge_src.shape[0]
        angle_src = np.append(
            angle_src,
            nblist_size * np.ones(prev_angle_size_ - angle_size, dtype=np.int64),
        )
        angle_dst = np.append(
            angle_dst,
            nblist_size * np.ones(prev_angle_size_ - angle_size, dtype=np.int64),
        )
        central_atom_index = np.append(
            central_atom_index,
            nattot * np.ones(prev_angle_size_ - angle_size, dtype=np.int64),
        )

        state["prev_angle_size"] = prev_angle_size_
        state["max_neigh"] = prev_max_neigh

        out = {
            **inputs,
            self.graph_key: {
                **graph,
                "angle_src": angle_src,
                "angle_dst": angle_dst,
                "central_atom": central_atom_index,
                "max_neigh": max_neigh_array,
                "overflow": graph.get("overflow", False),
            },
        }
        return out, state

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphAngleProcessor, {
            "graph_key": self.graph_key,
            "name": f"{self.graph_key}_AngleProcessor",
        }

    def get_updater(self):
        return get_graph_angle_updater(**{"graph_key": self.graph_key})

    def get_skin_updater(self):
        return self.get_updater()

    def get_graph_properties(self):
        return {
            self.graph_key: {
                "has_angles": True,
            }
        }


# class GraphAngleUpdater(nn.Module):
#     graph_key: str

#     @nn.compact
#     def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):


def get_graph_angle_updater(graph_key):
    def graph_angle_updater(inputs):
        def cumsum_from_zero(input_):
            return jnp.concatenate([jnp.array([0]), jnp.cumsum(input_[:-1], axis=0)])

        graph = inputs[graph_key]
        edge_src = graph["edge_src"]
        natoms = inputs["species"].shape[0]
        # count the number of neighbors for each atom
        counts = jnp.zeros(natoms, dtype=jnp.int32).at[edge_src].add(1, mode="drop")
        max_neigh = jnp.max(counts)

        # compute the number of pairs of neighbors for each atom
        pair_sizes = (counts * (counts - 1)) // 2
        nangles = jnp.sum(pair_sizes)

        # compute the indices of the pairs of neighbors
        max_neigh_max = graph["max_neigh"].shape[0]
        intra_pair_indices = jnp.stack(jnp.tril_indices(max_neigh_max, -1))[
            :, None, :
        ].repeat(natoms, axis=1)
        max_angle_size = natoms * intra_pair_indices.shape[2]

        # compute the index of the central atom for each pair of neighbors
        central_atom_index = jnp.repeat(
            jnp.arange(natoms), pair_sizes, total_repeat_length=max_angle_size
        )
        angle_mask = (
            jnp.arange(max_angle_size) < nangles
        )  # mask to remove the extra pairs

        # find sorted indices of the edges forming the angles
        mask = (jnp.arange(intra_pair_indices.shape[2]) < pair_sizes[:, None]).flatten()
        idx1 = jnp.argsort(~mask)
        mask = mask[idx1]
        shifts = cumsum_from_zero(counts)[central_atom_index]
        sorted_local_index12 = (
            intra_pair_indices.reshape(intra_pair_indices.shape[0], -1)[:, idx1]
            + shifts
        )

        # revert to global edge order
        idx = jnp.argsort(edge_src)
        angle_src, angle_dst = idx[sorted_local_index12]

        # get only angles that fit in the original nblist and mask extra angles
        nangles_max = graph["angle_src"].shape[0]
        angle_mask = angle_mask[:nangles_max]
        central_atom_index = jnp.where(
            angle_mask, central_atom_index[:nangles_max], natoms
        )
        angle_src = jnp.where(angle_mask, angle_src[:nangles_max], edge_src.shape[0])
        angle_dst = jnp.where(angle_mask, angle_dst[:nangles_max], edge_src.shape[0])

        angle_overflow = nangles > nangles_max
        neigh_overflow = max_neigh > max_neigh_max
        overflow = jnp.logical_or(angle_overflow, neigh_overflow)
        # overflow = jnp.logical_or(overflow, jnp.asarray(graph.get("overflow", False)))

        return {
            **inputs,
            graph_key: {
                **graph,
                "angle_src": angle_src,
                "angle_dst": angle_dst,
                "central_atom": central_atom_index,
                "overflow": jnp.logical_or(overflow, jnp.asarray(graph.get("overflow", False))),
                "angle_overflow": (nangles, nangles_max),
                "neigh_overflow": (max_neigh, max_neigh_max),
            },
        }

    return graph_angle_updater


class GraphAngleProcessor(nn.Module):
    graph_key: str

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        vec = graph["vec"]
        angle_src = graph["angle_src"]
        angle_dst = graph["angle_dst"]
        angle_mask = angle_src < distances.shape[0]
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
                "cos_angles": cos_angles,
                "angles": angles,
                "angle_mask": angle_mask,
            },
        }


# class GraphDenseExtension(nn.Module):
#     mult_size: float = 1.1
#     graph_key: str = "graph"

#     @nn.compact
#     def __call__(self, inputs: Dict) -> Union[dict, jax.Array]:
#         graph = inputs[self.graph_key]

#         edge_src = np.array(graph["edge_src"])
#         edge_dst = np.array(graph["edge_dst"])
#         natoms = inputs["species"].shape[0]
#         # edge_src = edge_src[edge_src < natoms]
#         # edge_dst = edge_dst[edge_src < natoms]
#         Nedge = len(edge_src)

#         counts = np.zeros(natoms + 1, dtype=int)
#         np.add.at(counts, edge_src, 1)

#         max_count = np.max(counts[:-1])
#         prev_size = self.variable(
#             "preprocessing",
#             f"{self.graph_key}_prev_dense_size",
#             lambda: 0,
#         )
#         prev_size_ = prev_size.value
#         if max_count > prev_size_:
#             prev_size_ = int(self.mult_size * max_count)

#         offset = np.tile(np.arange(prev_size_), natoms)  # [:len(edge_src)]
#         if len(offset) < Nedge:
#             offset = np.append(
#                 offset, np.zeros(Nedge - len(offset), dtype=offset.dtype)
#             )
#         else:
#             offset = offset[:Nedge]
#         offset[edge_src == natoms] = 0
#         indices = edge_src * prev_size_ + offset
#         dense_idx = natoms * np.ones(((natoms + 1) * prev_size_,), dtype=np.int32)
#         dense_idx[indices] = edge_dst
#         dense_idx = dense_idx.reshape(natoms, prev_size_)

#         rev_indices = Nedge * np.ones_like(dense_idx)
#         rev_indices[indices] = np.arange(Nedge)
#         rev_indices[dense_idx == natoms] = Nedge

#         return {
#             **inputs,
#             self.graph_key: {
#                 **graph,
#                 "dense_nblist": dense_idx,
#                 "sparse2dense_indices": indices,
#                 "dense2sparse_indices": rev_indices,
#             },
#         }

#     def get_graph_properties(self):
#         return {
#             self.graph_key: {
#                 "has_dense": True,
#             }
#         }


@dataclasses.dataclass(frozen=True)
class AtomPadding:
    mult_size: float = 1.2

    def init(self):
        return {"prev_nat": 0}

    def __call__(self, inputs: Dict, state={}) -> Union[dict, jax.Array]:
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
            output["natoms"] = np.append(inputs["natoms"], add_atoms)
            output["species"] = np.append(
                species, -1 * np.ones(add_atoms, dtype=species.dtype)
            )
            output["batch_index"] = np.append(
                batch_index, np.array([nsys] * add_atoms, dtype=batch_index.dtype)
            )

        output["true_atoms"] = output["species"] > 0
        output["true_sys"] = np.arange(len(output["natoms"])) < nsys

        state = {**state, "prev_nat": prev_nat_}

        return output, state


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
    return {**inputs, "natoms": natoms, "batch_index": batch_index}


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
    layers: Sequence[Callable[..., Dict[str, Any]]]
    use_atom_padding: bool = False
    atom_padder: AtomPadding = AtomPadding()

    def __post_init__(self):
        if not isinstance(self.layers, Sequence):
            raise ValueError(
                f"'layers' must be a sequence, got '{type(self.layers).__name__}'."
            )
        if not self.layers:
            raise ValueError(f"Error: no Preprocessing layers were provided.")

    def __call__(self, inputs: Dict[str, Any], state) -> Dict[str, Any]:
        if state.get("check_input", True):
            inputs = check_input(inputs)
        new_state = []
        layer_state = state["layers_state"]
        i = 0
        if self.use_atom_padding:
            inputs, s = self.atom_padder(inputs, layer_state[0])
            new_state.append(s)
            i += 1
        for layer in self.layers:
            inputs, s = layer(inputs, layer_state[i])
            new_state.append(s)
            i += 1
        return inputs, {**state, "layers_state": new_state}

    def init(self):
        state = []
        if self.use_atom_padding:
            state.append(self.atom_padder.init())
        for layer in self.layers:
            state.append(layer.init())
        return {"check_input": True, "layers_state": state}

    def init_with_output(self, inputs):
        state = self.init()
        return self(inputs, state)

    def get_updaters(self, return_list=False):
        updaters = [convert_to_jax]
        for layer in self.layers:
            if hasattr(layer, "get_updater"):
                updaters.append(layer.get_updater())
        if return_list:
            return updaters
        return chain(*updaters)

    def get_skin_updaters(self, return_list=False):
        updaters = [convert_to_jax]
        for layer in self.layers:
            if hasattr(layer, "get_skin_updater"):
                updaters.append(layer.get_skin_updater())
        if return_list:
            return updaters
        return chain(*updaters)

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
    "GRAPH_FIXED": GraphGeneratorFixed,
    "GRAPH_FILTER": GraphFilter,
    "GRAPH_ANGULAR_EXTENSION": GraphAngularExtension,
    # "GRAPH_DENSE_EXTENSION": GraphDenseExtension,
}
