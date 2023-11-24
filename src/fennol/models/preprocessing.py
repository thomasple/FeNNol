import flax.linen as nn
from typing import Sequence, Callable, Union, Dict, Any
import jax.numpy as jnp
import jax
import numpy as np
from typing import Optional, Tuple
import numba
from ..utils.nblist import (
    compute_nblist_flatbatch,
    angular_nblist,
    compute_nblist_fixed,
    compute_nblist_flatbatch_minimage,
    compute_nblist_fixed_minimage,
)


def minmaxone(x, name=""):
    print(name, x.min(), x.max(), (x**2).mean())


class GraphGenerator(nn.Module):
    cutoff: float
    mult_size: float = 1.1
    graph_key: str = "graph"
    switch_start: float = 0.0

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
        isys = np.array(inputs["isys"])
        natoms = np.array(inputs["natoms"])
        padding_value = coords.shape[0]
        if "true_atoms" in inputs:
            true_atoms = np.array(inputs["true_atoms"], dtype=bool)
            true_sys = np.array(inputs["true_sys"], dtype=bool)
            coords = coords[true_atoms]
            isys = isys[true_atoms]
            natoms = natoms[true_sys]

        prev_nblist_size = self.variable(
            "preprocessing",
            f"{self.graph_key}_prev_nblist_size",
            lambda: 0,
        )
        if "cells" in inputs:
            cells = np.array(inputs["cells"])
            if cells.ndim == 2:
                cells = cells[None, :, :]
            reciprocal_cells = np.linalg.inv(cells)
            (
                edge_src,
                edge_dst,
                d12,
                pbc_shifts,
                prev_nblist_size_,
            ) = compute_nblist_flatbatch_minimage(
                coords,
                cutoff,
                isys,
                natoms,
                mult_size,
                cells,
                reciprocal_cells,
                prev_nblist_size.value,
                padding_value=padding_value,
            )
        else:
            edge_src, edge_dst, d12, prev_nblist_size_ = compute_nblist_flatbatch(
                coords,
                cutoff,
                isys,
                natoms,
                mult_size,
                prev_nblist_size.value,
                padding_value=padding_value,
            )
            pbc_shifts = None

        if not self.is_initializing():
            prev_nblist_size.value = prev_nblist_size_

        return {
            **inputs,
            self.graph_key: {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "cutoff": self.cutoff,
                "pbc_shifts": pbc_shifts,
            },
        }

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "switch_start": self.switch_start,
            "name": f"{self.graph_key}_Processor",
        }

    def get_graph_properties(self):
        return {self.graph_key: {"cutoff": self.cutoff, "directed": True}}


class GraphGeneratorFixed(nn.Module):
    cutoff: float
    mult_size: float = 1.1
    graph_key: str = "graph"
    switch_start: float = 0.0

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
        isys = inputs["isys"]
        natoms = inputs["natoms"]
        padding_value = coords.shape[0]
        if "true_atoms" in inputs:
            true_atoms = inputs["true_atoms"]
            true_sys = inputs["true_sys"]
            coords = coords[true_atoms]
            isys = isys[true_atoms]
            natoms = natoms[true_sys]

        prev_nblist_size = self.variable(
            "preprocessing",
            f"{self.graph_key}_prev_nblist_size",
            lambda: 1,
        )
        prev_nblist_size_ = prev_nblist_size.value
        max_nat = int(np.max(natoms))

        if "cells" in inputs:
            cells = np.asarray(inputs["cells"])
            if cells.ndim == 2:
                cells = cells[None, :, :]
            edge_src, edge_dst, d12, pbc_shifts, npairs = compute_nblist_fixed_minimage(
                coords,
                cutoff,
                isys,
                natoms,
                max_nat,
                prev_nblist_size_,
                padding_value,
                cells,
            )
        else:
            edge_src, edge_dst, d12, npairs = compute_nblist_fixed(
                coords, cutoff, isys, natoms, max_nat, prev_nblist_size_, padding_value
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
                    isys,
                    natoms,
                    max_nat,
                    prev_nblist_size_,
                    padding_value,
                    cells,
                )
            else:
                edge_src, edge_dst, d12, npairs = compute_nblist_fixed(
                    coords,
                    cutoff,
                    isys,
                    natoms,
                    max_nat,
                    prev_nblist_size_,
                    padding_value,
                )

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

        return {
            **inputs,
            self.graph_key: {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "cutoff": self.cutoff,
                "pbc_shifts": pbc_shifts,
            },
        }

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "switch_start": self.switch_start,
            "name": f"{self.graph_key}_Processor",
        }

    def get_graph_properties(self):
        return {self.graph_key: {"cutoff": self.cutoff, "directed": True}}


class GraphProcessor(nn.Module):
    cutoff: float
    graph_key: str = "graph"
    switch_start: float = 0.0

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
            isysvec = inputs["isys"][edge_src]
            cells = inputs["cells"][isysvec]
            vec = vec + jnp.einsum("sij,sj->si",cells,graph["pbc_shifts"])
            # vec = vec + graph["pbc_shifts"]
        distances = jnp.linalg.norm(vec, axis=-1)
        edge_mask = distances < self.cutoff

        if self.switch_start > 1.e-5:
            assert self.switch_start < 1.0, "switch_start is a proportion of cutoff and must be smaller than 1."
            cutoff_in = self.switch_start * self.cutoff
            x = distances - cutoff_in
            switch = jnp.where(
                distances < cutoff_in,
                1.0,
                jnp.where(
                    edge_mask,
                    0.5 * jnp.cos(x * (jnp.pi / (self.cutoff - cutoff_in))) + 0.5,
                    0.0,
                ),
            )
        else:
            switch = jnp.where(
                edge_mask, 0.5 * jnp.cos(distances * (jnp.pi / self.cutoff)) + 0.5, 0.0
            )

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

        if not self.is_initializing():
            prev_nblist_size.value = prev_nblist_size_

        return {
            **inputs,
            self.graph_out: {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "filter_indices": indices,
                "cutoff": self.cutoff,
            },
        }

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphFilterProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "graph_out": self.graph_out,
            "name": f"{self.graph_out}_Filter_{self.graph_key}",
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

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph_in = inputs[self.graph_key]
        graph = inputs[self.graph_out]
        coords = inputs["coordinates"]

        filter_indices = graph["filter_indices"]

        vec = (
            graph_in["vec"].at[filter_indices].get(mode="fill", fill_value=self.cutoff)
        )
        distances = (
            graph_in["distances"]
            .at[filter_indices]
            .get(mode="fill", fill_value=self.cutoff)
        )
        edge_mask = distances < self.cutoff

        switch = jnp.where(
            edge_mask, 0.5 * jnp.cos(distances * (jnp.pi / self.cutoff)) + 0.5, 0.0
        )

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
            isys = inputs["isys"]
            for k, v in inputs.items():
                if isinstance(v, np.ndarray):
                    if v.shape[0] == nat:
                        output[k] = np.append(
                            v,
                            np.zeros((add_atoms, *v.shape[1:]), dtype=v.dtype),
                            axis=0,
                        )
                    elif v.shape[0] == nsys:
                        output[k] = np.append(
                            v, np.zeros((1, *v.shape[1:]), dtype=v.dtype), axis=0
                        )
            output["natoms"] = np.append(inputs["natoms"], add_atoms)
            output["species"] = np.append(
                species, -1 * np.ones(add_atoms, dtype=species.dtype)
            )
            output["isys"] = np.append(
                isys, np.array([nsys] * add_atoms, dtype=isys.dtype)
            )

        output["true_atoms"] = output["species"] > 0
        output["true_sys"] = np.arange(len(output["natoms"])) < nsys

        if not self.is_initializing():
            prev_nat.value = prev_nat_

        return output


def atom_unpadding(inputs: Dict[str, Any]) -> Dict[str, Any]:
    species = inputs["species"]
    natall = species.shape[0]
    nat = np.argmax(species <= 0)
    if nat == 0:
        return inputs

    natoms = inputs["natoms"]
    nsysall = len(natoms)

    output = {**inputs}
    for k, v in inputs.items():
        if isinstance(v, jax.Array) or isinstance(v, np.ndarray):
            if v.shape[0] == natall:
                output[k] = v[:nat]
            elif v.shape[0] == nsysall:
                output[k] = v[:-1]
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
    isys = inputs.get("isys", np.repeat(np.arange(len(natoms), dtype=np.int64), natoms))
    return {**inputs, "natoms": natoms, "isys": isys}


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
}
