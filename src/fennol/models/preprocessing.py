import flax.linen as nn
from typing import Sequence, Callable, Union, Dict, Any
import jax.numpy as jnp
import jax
import numpy as np
from typing import Optional, Tuple
import numba
import dataclasses

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


class GraphGenerator(nn.Module):
    cutoff: float
    mult_size: float = 1.1
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict)
    kmax: int = 30
    kthr: float = 1e-6
    k_space: bool = False

    @nn.compact
    def __call__(self, inputs: Dict) -> Union[dict, jax.Array]:
        if self.graph_key in inputs:
            if inputs[self.graph_key].get("keep_graph", False):
                return inputs

        cutoff = self.cutoff
        if "nblist_skin" in inputs:
            cutoff += float(inputs["nblist_skin"])

        mult_size = float(inputs.get("nblist_mult_size", self.mult_size))

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

        prev_nblist_size = self.variable(
            "preprocessing",
            f"{self.graph_key}_prev_nblist_size",
            lambda: 0,
        )
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
                prev_nblist_size.value,
                padding_value=padding_value,
            )
        else:
            if ase_nblist:
                compute_nblist = compute_nblist_ase
            else:
                compute_nblist = compute_nblist_flatbatch
            edge_src, edge_dst, d12, prev_nblist_size_,isym = compute_nblist(
                coords,
                cutoff,
                batch_index,
                natoms,
                mult_size,
                prev_nblist_size.value,
                padding_value=padding_value,
            )
            pbc_shifts = None

        if not self.is_initializing():
            prev_nblist_size.value = prev_nblist_size_

        out = {
            **inputs,
            self.graph_key: {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "cutoff": self.cutoff,
                "pbc_shifts": pbc_shifts,
                "isym": isym,
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
                        reciprocal_cells, cutoff, self.kmax, self.kthr
                    )
                out[self.graph_key]["k_points"] = ks
                out[self.graph_key]["b_ewald"] = bewald

        return out

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "switch_params": self.switch_params,
            "name": f"{self.graph_key}_Processor",
        }

    def get_graph_properties(self):
        return {self.graph_key: {"cutoff": self.cutoff, "directed": True}}


class GraphGeneratorFixed(nn.Module):
    cutoff: float
    mult_size: float = 1.1
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict)
    kmax: int = 30
    kthr: float = 1e-6
    k_space: bool = False

    @nn.compact
    def __call__(self, inputs: Dict) -> Union[dict, jax.Array]:
        if self.graph_key in inputs:
            if inputs[self.graph_key].get("keep_graph", False):
                return inputs

        cutoff = self.cutoff
        if "nblist_skin" in inputs:
            cutoff += float(inputs["nblist_skin"])

        mult_size = float(inputs.get("nblist_mult_size", self.mult_size))

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

        prev_nblist_size = self.variable(
            "preprocessing",
            f"{self.graph_key}_prev_nblist_size",
            lambda: 1,
        )
        prev_nblist_size_ = prev_nblist_size.value
        max_nat = int(np.max(natoms))

        if "cells" in inputs:
            cells = np.asarray(inputs["cells"], dtype=coords.dtype)
            if cells.ndim == 2:
                cells = cells[None, :, :]
            reciprocal_cells = np.linalg.inv(cells)

            minimage = inputs.get("minimum_image", True)
            assert minimage, "Fixed nblist only works with minimum image convention"

            edge_src, edge_dst, d12, pbc_shifts, npairs = compute_nblist_fixed_minimage(
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
        else:
            edge_src, edge_dst, d12, npairs = compute_nblist_fixed(
                coords,
                cutoff,
                batch_index,
                natoms,
                max_nat,
                prev_nblist_size_,
                padding_value,
            )

        if npairs > prev_nblist_size.value:
            prev_nblist_size_ = int(mult_size * npairs) + 1
            if "cells" in inputs:
                (
                    edge_src,
                    edge_dst,
                    d12,
                    pbc_shifts,
                    npairs,
                ) = compute_nblist_fixed_minimage(
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
            else:
                edge_src, edge_dst, d12, npairs = compute_nblist_fixed(
                    coords,
                    cutoff,
                    batch_index,
                    natoms,
                    max_nat,
                    prev_nblist_size_,
                    padding_value,
                )

        iedge = np.arange(len(edge_src))
        isym = np.concatenate((iedge+iedge.shape[0],iedge))
        edge_src, edge_dst = np.concatenate((edge_src, edge_dst)), np.concatenate(
            (edge_dst, edge_src)
        )
        d12 = np.concatenate((d12, d12))
        if "cells" in inputs:
            pbc_shifts = np.concatenate((pbc_shifts, -pbc_shifts))
        else:
            pbc_shifts = None

        if not self.is_initializing():
            prev_nblist_size.value = prev_nblist_size_

        out = {
            **inputs,
            self.graph_key: {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "cutoff": self.cutoff,
                "pbc_shifts": pbc_shifts,
                "isym": isym,
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
                    print("Computing k-points", cutoff, self.kmax, self.kthr)
                    ks, _, _, bewald = get_reciprocal_space_parameters(
                        reciprocal_cells, cutoff, self.kmax, self.kthr
                    )
                    print("n k-points", ks.shape)
                out[self.graph_key]["k_points"] = ks
                out[self.graph_key]["b_ewald"] = bewald
        return out

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "switch_params": self.switch_params,
            "name": f"{self.graph_key}_Processor",
        }

    def get_graph_properties(self):
        return {self.graph_key: {"cutoff": self.cutoff, "directed": True}}


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


class GraphFilter(nn.Module):
    cutoff: float
    graph_out: str
    mult_size: float = 1.1
    graph_key: str = "graph"
    remove_hydrogens: int = False
    switch_params: dict = dataclasses.field(default_factory=dict)
    k_space: bool = False
    kmax: int = 30
    kthr: float = 1e-6

    @nn.compact
    def __call__(self, inputs: Dict):
        if self.graph_out in inputs:
            if inputs[self.graph_out].get("keep_graph", False):
                return inputs

        if "nblist_skin" in inputs:
            c2 = (self.cutoff + float(inputs["nblist_skin"])) ** 2
        else:
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
            isym=None
        else:
            isym = graph_in.get("isym",None)
            if isym is not None:
                isym = isym[indices]

        prev_nblist_size = self.variable(
            "preprocessing",
            f"{self.graph_out}_prev_nblist_size",
            lambda: 0,
        )
        prev_nblist_size_ = prev_nblist_size.value

        mult_size = float(inputs.get("nblist_mult_size", self.mult_size))

        nattot = inputs["species"].shape[0]
        nblist_size = edge_src.shape[0]
        if nblist_size > prev_nblist_size_:
            prev_nblist_size_ = int(mult_size * nblist_size)

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

        if not self.is_initializing():
            prev_nblist_size.value = prev_nblist_size_

        out = {
            **inputs,
            self.graph_out: {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "filter_indices": indices,
                "cutoff": self.cutoff,
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

        return out

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphFilterProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "graph_out": self.graph_out,
            "name": f"{self.graph_out}_Filter_{self.graph_key}",
            "switch_params": self.switch_params,
        }

    def get_graph_properties(self):
        return {
            self.graph_out: {
                "cutoff": self.cutoff,
                "directed": True,
                "original_graph": self.graph_key,
            }
        }


class GraphFilterProcessor(nn.Module):
    cutoff: float
    graph_out: str
    graph_key: str = "graph"
    switch_params: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph_in = inputs[self.graph_key]
        graph = inputs[self.graph_out]

        filter_indices = graph["filter_indices"]

        vec = (
            graph_in["vec"].at[filter_indices].get(mode="fill", fill_value=self.cutoff)
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
        }
        return {**inputs, self.graph_out: graph_out}


class GraphAngularExtension(nn.Module):
    mult_size: float = 1.1
    graph_key: str = "graph"

    @nn.compact
    def __call__(self, inputs: Dict) -> Union[dict, jax.Array]:
        graph = inputs[self.graph_key]

        edge_src = np.array(graph["edge_src"])
        nattot = inputs["species"].shape[0]
        # edge_src = edge_src[edge_src < nattot]

        central_atom_index, angle_src, angle_dst = angular_nblist(edge_src, nattot)

        prev_angle_size = self.variable(
            "preprocessing",
            f"{self.graph_key}_prev_angle_size",
            lambda: 0,
        )
        mult_size = float(inputs.get("nblist_mult_size", self.mult_size))

        prev_angle_size_ = prev_angle_size.value
        angle_size = angle_src.shape[0]
        if angle_size > prev_angle_size_:
            prev_angle_size_ = int(mult_size * angle_size)

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

        if not self.is_initializing():
            prev_angle_size.value = prev_angle_size_

        return {
            **inputs,
            self.graph_key: {
                **graph,
                "angle_src": angle_src,
                "angle_dst": angle_dst,
                "central_atom": central_atom_index,
            },
        }

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


class AtomPadding(nn.Module):
    mult_size: float = 1.2

    @nn.compact
    def __call__(self, inputs: Dict) -> Union[dict, jax.Array]:
        species = inputs["species"]
        nat = species.shape[0]

        prev_nat = self.variable(
            "preprocessing",
            "prev_nat",
            lambda: 0,
        )
        prev_nat_ = prev_nat.value
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

        if not self.is_initializing():
            prev_nat.value = prev_nat_

        return output


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


class JaxConverter(nn.Module):
    def __call__(self, data):
        def convert(x):
            if isinstance(x, np.ndarray):
                # if x.dtype == np.float64:
                #     return jnp.asarray(x, dtype=jnp.float32)
                return jnp.asarray(x)
            return x

        return jax.tree_util.tree_map(convert, data)


class PreprocessingChain(nn.Module):
    layers: Sequence[Callable[..., Dict[str, Any]]]
    use_atom_padding: bool = False

    def __post_init__(self):
        if not isinstance(self.layers, Sequence):
            raise ValueError(
                f"'layers' must be a sequence, got '{type(self.layers).__name__}'."
            )
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.layers:
            raise ValueError(f"Empty Sequential module {self.name}.")

        data = check_input(inputs)
        if self.use_atom_padding:
            data = AtomPadding()(data)
        for layer in self.layers:
            data = layer(data)
        return data


PREPROCESSING = {
    "GRAPH": GraphGenerator,
    "GRAPH_FIXED": GraphGeneratorFixed,
    "GRAPH_FILTER": GraphFilter,
    "GRAPH_ANGULAR_EXTENSION": GraphAngularExtension,
    # "GRAPH_DENSE_EXTENSION": GraphDenseExtension,
}
