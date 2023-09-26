import flax.linen as nn
from typing import Sequence, Callable, Union, Dict, Any
import jax.numpy as jnp
import jax
import numpy as np
from typing import Optional, Tuple
import numba


@numba.njit
def compute_nblist_flatbatch(
    coords, cutoff, isys, natoms, mult_size, prev_nblist_size=0
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
    src, dst = np.array(src + dst), np.array(dst + src)
    d12s = np.array(d12s + d12s)

    nblist_size = src.shape[0]
    if nblist_size > prev_nblist_size:
        prev_nblist_size = int(mult_size * nblist_size)
    src = np.append(
        src, nattot * np.ones(prev_nblist_size - nblist_size, dtype=np.int32)
    )
    dst = np.append(
        dst, nattot * np.ones(prev_nblist_size - nblist_size, dtype=np.int32)
    )
    d12s = np.append(
        d12s, c2 * np.ones(prev_nblist_size - nblist_size, dtype=np.float32)
    )
    return src, dst, d12s, prev_nblist_size


class GraphGenerator(nn.Module):
    cutoff: float
    mult_size: float = 1.1
    graph_key: str = "graph"

    @nn.compact
    def __call__(self, inputs: Dict) -> Union[dict, jax.Array]:
        if self.graph_key in inputs:
            return inputs
        coords = np.array(inputs["coordinates"])
        isys = np.array(inputs["isys"])
        natoms = np.array(inputs["natoms"])

        prev_nblist_size = self.variable(
            "preprocessing",
            f"{self.graph_key}_prev_nblist_size",
            lambda: 0,
        )
        edge_src, edge_dst, d12, prev_nblist_size_ = compute_nblist_flatbatch(
            coords, self.cutoff, isys, natoms, self.mult_size, prev_nblist_size.value
        )

        if not self.is_initializing():
            prev_nblist_size.value = prev_nblist_size_

        return {
            **inputs,
            self.graph_key: {
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "d12": d12,
                "cutoff": self.cutoff,
            },
        }

    def get_processor(self) -> Tuple[nn.Module, Dict]:
        return GraphProcessor, {
            "cutoff": self.cutoff,
            "graph_key": self.graph_key,
            "name": f"{self.graph_key}_Processor",
        }

    def get_graph_properties(self):
        return {self.graph_key: {"cutoff": self.cutoff, "directed": True}}


class GraphProcessor(nn.Module):
    cutoff: float
    graph_key: Optional[str] = "graph"

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        if self.graph_key is None:
            coords, graph = inputs
        else:
            graph = inputs[self.graph_key]
            coords = inputs["coordinates"]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        edge_mask = edge_src < coords.shape[0]
        vec = coords.at[edge_dst].get(mode="fill", fill_value=self.cutoff) - coords.at[
            edge_src
        ].get(mode="fill", fill_value=0.0)
        distances = jnp.linalg.norm(vec, axis=-1)
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
        if self.graph_key is not None:
            return {**inputs, self.graph_key: graph_out}
        return graph_out


class GraphFilter(nn.Module):
    cutoff: float
    graph_out: str
    mult_size: float = 1.1
    graph_key: str = "graph"

    @nn.compact
    def __call__(self, inputs: Dict) -> Union[dict, jax.Array]:
        if self.graph_out in inputs:
            return inputs

        c2 = self.cutoff**2

        graph_in = inputs[self.graph_key]
        d12 = graph_in["d12"]
        indices = (d12 < c2).nonzero()
        edge_src_in = graph_in["edge_src"]
        nblist_size_in = edge_src_in.shape[0]
        edge_src = edge_src_in[indices]
        edge_dst = graph_in["edge_dst"][indices]
        d12 = d12[indices]

        prev_nblist_size = self.variable(
            "preprocessing",
            f"{self.graph_out}_prev_nblist_size",
            lambda: 0,
        )
        prev_nblist_size_ = prev_nblist_size.value

        nattot = inputs["species"].shape[0]
        nblist_size = edge_src.shape[0]
        if nblist_size > prev_nblist_size_:
            prev_nblist_size_ = int(self.mult_size * nblist_size)

        edge_src = np.append(
            edge_src, nattot * np.ones(prev_nblist_size - nblist_size, dtype=np.int32)
        )
        edge_dst = np.append(
            edge_dst, nattot * np.ones(prev_nblist_size - nblist_size, dtype=np.int32)
        )
        d12 = np.append(
            d12, c2 * np.ones(prev_nblist_size - nblist_size, dtype=np.float32)
        )
        indices = np.append(
            indices,
            nblist_size_in * np.ones(prev_nblist_size - nblist_size, dtype=np.int32),
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
    graph_key: Optional[str] = "graph"

    @nn.compact
    def __call__(self, inputs: Union[dict, Tuple[jax.Array, dict]]):
        graph_in = inputs[self.graph_key]
        graph = inputs[self.graph_out]
        coords = inputs["coordinates"]

        edge_src = graph["edge_src"]
        edge_mask = edge_src < coords.shape[0]
        filter_indices = graph["filter_indices"]

        vec = (
            graph_in["vec"].at[filter_indices].get(mode="fill", fill_value=self.cutoff)
        )
        distances = (
            graph_in["distances"]
            .at[filter_indices]
            .get(mode="fill", fill_value=self.cutoff)
        )

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


@numba.njit
def angular_nblist(edge_src, natoms):
    idx = np.argsort(edge_src)
    rev_idx = np.argsort(idx)

    counts = np.zeros(natoms, dtype=np.int32)
    for i in edge_src:
        counts[i] += 1

    pair_sizes = (counts * (counts - 1)) // 2
    nangles = np.sum(pair_sizes)

    shift = 0
    p1s = np.zeros(nangles, dtype=np.int32)
    p2s = np.zeros(nangles, dtype=np.int32)
    central_atom_index = np.zeros(nangles, dtype=np.int32)
    iang = 0
    for i, c in enumerate(counts):
        if c < 2:
            shift += c
            continue
        for j in range(c):
            for k in range(j + 1, c):
                p1s[iang] = k + shift
                p2s[iang] = j + shift
                central_atom_index[iang] = i
                iang += 1
        shift += c
    angle_src = rev_idx[p1s]
    angle_dst = rev_idx[p2s]

    return central_atom_index, angle_src, angle_dst


class GraphAngularExtension(nn.Module):
    mult_size: float = 1.1
    graph_key: str = "graph"

    @nn.compact
    def __call__(self, inputs: Dict) -> Union[dict, jax.Array]:
        graph = inputs[self.graph_key]

        edge_src = graph["edge_src"]
        nattot = inputs["species"].shape[0]

        central_atom_index, angle_src, angle_dst = angular_nblist(edge_src, nattot)

        prev_angle_size = self.variable(
            "preprocessing",
            f"{self.graph_key}_prev_angle_size",
            lambda: 0,
        )
        prev_angle_size_ = prev_angle_size.value
        angle_size = angle_src.shape[0]
        if angle_size > prev_angle_size_:
            prev_angle_size_ = int(self.mult_size * angle_size)

        angle_src = np.append(
            angle_src, -1 * np.ones(prev_angle_size - angle_size, dtype=np.int32)
        )
        angle_dst = np.append(
            angle_dst, -1 * np.ones(prev_angle_size - angle_size, dtype=np.int32)
        )
        central_atom_index = np.append(
            central_atom_index,
            nattot * np.ones(prev_angle_size - angle_size, dtype=np.int32),
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
        angle_mask = angle_src >= 0
        d1, d2 = distances[angle_src], distances[angle_dst]

        cos_angles = (vec[angle_src] * vec[angle_dst]).sum(axis=-1) / jnp.clip(
            d1 * d2, a_min=1e-10
        )
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


def check_fennix_args(*args, **kwargs) -> Dict[str, Any]:
    if len(args) == 2:
        species, coordinates = args
        return {"species": species, "coordinates": coordinates, **kwargs}

    if len(args) == 1:
        assert isinstance(args[0], dict), "input must be a dictionary"
        inputs = {**args[0], **kwargs}
        assert "species" in inputs, "species must be provided"
        assert "coordinates" in inputs, "coordinates must be provided"
        return inputs

    if len(args) == 0:
        assert "species" in kwargs, "species must be provided"
        assert "coordinates" in kwargs, "coordinates must be provided"
        return kwargs

    print(
        "Invalid input, must be either a dictionary or two arrays (species, coordinates) and/or keyword arguments"
    )
    raise ValueError("invalid input")


def format_input(*args, **kwargs):
    inputs = check_fennix_args(*args, **kwargs)
    species = inputs["species"]
    if "isys" not in inputs:
        inputs["isys"] = np.zeros_like(species, dtype=np.int32)

    if "natoms" not in inputs:
        inputs["natoms"] = np.array([species.shape[0]], dtype=np.int32)

    return inputs


class JaxConverter(nn.Module):
    def __call__(self, data):
        return jax.tree_util.tree_map(
            lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, data
        )


class PreprocessingChain(nn.Module):
    layers: Sequence[Callable[..., Dict[str, Any]]]

    def __post_init__(self):
        if not isinstance(self.layers, Sequence):
            raise ValueError(
                f"'layers' must be a sequence, got '{type(self.layers).__name__}'."
            )
        super().__post_init__()

    def __call__(self, *args, **kwargs):
        if not self.layers:
            raise ValueError(f"Empty Sequential module {self.name}.")

        data = format_input(*args, **kwargs)
        for layer in self.layers:
            data = layer(data)
        return data


PREPROCESSING = {
    "GRAPH": GraphGenerator,
    "GRAPH_FILTER": GraphFilter,
    "GRAPH_ANGULAR_EXTENSION": GraphAngularExtension,
}
