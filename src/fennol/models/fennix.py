from typing import Any, Sequence, Callable, Union, Optional, Tuple, Dict
from copy import deepcopy
import dataclasses
from collections import OrderedDict

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax import serialization
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict

from .preprocessing import (
    GraphGenerator,
    PreprocessingChain,
    JaxConverter,
    atom_unpadding,
)
from .modules import MODULES, PREPROCESSING, FENNIXModules


@dataclasses.dataclass
class FENNIX:
    """
    Static wrapper for FENNIX models

    The underlying model is a `fennol.models.modules.FENNIXModules` built from the `modules` dictionary
    which references registered modules in `fennol.models.modules.MODULES` and provides the parameters for initialization.

    Since the model is static and contains variables, it must be initialized right away with either
    `example_data`, `variables` or `rng_key`. If `variables` is provided, it is used directly. If `example_data`
    is provided, the model is initialized with `example_data` and the resulting variables are stored
    in the wrapper. If only `rng_key` is provided, the model is initialized with a dummy system and the resulting.
    """

    cutoff: float | None
    modules: FENNIXModules
    variables: Dict
    preprocessing: PreprocessingChain
    _apply: Callable[[Dict, Dict], Dict]
    _total_energy: Callable[[Dict, Dict], Tuple[jnp.ndarray, Dict]]
    _energy_and_forces: Callable[[Dict, Dict], Tuple[jnp.ndarray, jnp.ndarray, Dict]]
    _input_args: Dict
    _graphs_properties: Dict
    preproc_state: Dict
    energy_terms: Optional[Sequence[str]] = None
    _initializing: bool = True
    use_atom_padding: bool = False

    def __init__(
        self,
        cutoff: float,
        modules: OrderedDict,
        preprocessing: OrderedDict = OrderedDict(),
        example_data=None,
        rng_key: Optional[jax.random.PRNGKey] = None,
        variables: Optional[dict] = None,
        energy_terms: Optional[Sequence[str]] = None,
        use_atom_padding: bool = False,
        graph_config: Dict = {},
        **kwargs,
    ) -> None:
        """ Initialize the FENNIX model
        
        Arguments:
        ----------
        cutoff: float
            The cutoff radius for the model
        modules: OrderedDict
            The dictionary defining the sequence of FeNNol modules and their parameters.
        preprocessing: OrderedDict
            The dictionary defining the sequence of preprocessing modules and their parameters.
        example_data: dict
            Example data to initialize the model. If not provided, a dummy system is generated.
        rng_key: jax.random.PRNGKey
            The random key to initialize the model. If not provided, jax.random.PRNGKey(0) is used (should be avoided).
        variables: dict
            The variables of the model (i.e. weights, biases and all other tunable parameters). 
            If not provided, the variables are initialized (usually at random)
        energy_terms: Sequence[str]
            The energy terms in the model output that will be summed to compute the total energy.
            If None, the total energy is always zero (useful for non-PES models).
        use_atom_padding: bool
            If True, the model will use atom padding for the input data. 
            This is useful when one plans to frequently change the number of atoms in the system (for example during training).
        graph_config: dict
            Edit the graph configuration. Mostly used to change a long-range cutoff as a function of a simulation box size.
        
        """
        self._input_args = {
            "cutoff": cutoff,
            "modules": OrderedDict(modules),
            "preprocessing": OrderedDict(preprocessing),
        }
        self.cutoff = cutoff
        self.energy_terms = energy_terms
        self.use_atom_padding = use_atom_padding

        # add non-differentiable/non-jittable modules
        preprocessing = deepcopy(preprocessing)
        if cutoff is None:
            preprocessing_modules = []
        else:
            prep_keys = list(preprocessing.keys())
            graph_params = {"cutoff": cutoff, "graph_key": "graph"}
            if len(prep_keys) > 0 and prep_keys[0] == "graph":
                graph_params = {
                    **graph_params,
                    **preprocessing.pop("graph"),
                }
            graph_params = {**graph_params, **graph_config}

            preprocessing_modules = [
                GraphGenerator(**graph_params),
            ]

        for name, params in preprocessing.items():
            key = str(params.pop("module_name")) if "module_name" in params else name
            key = str(params.pop("FID")) if "FID" in params else key
            mod = PREPROCESSING[key.upper()](**freeze(params))
            preprocessing_modules.append(mod)

        self.preprocessing = PreprocessingChain(
            tuple(preprocessing_modules), use_atom_padding
        )
        graphs_properties = self.preprocessing.get_graphs_properties()
        self._graphs_properties = freeze(graphs_properties)
        # add preprocessing modules that should be differentiated/jitted
        mods = [(JaxConverter, {})] + self.preprocessing.get_processors()
        # mods = self.preprocessing.get_processors(return_list=True)

        # build the model
        modules = deepcopy(modules)
        modules_names = []
        for name, params in modules.items():
            key = str(params.pop("module_name")) if "module_name" in params else name
            key = str(params.pop("FID")) if "FID" in params else key
            if name in modules_names:
                raise ValueError(f"Module {name} already exists")
            modules_names.append(name)
            params["name"] = name
            mod = MODULES[key.upper()]
            fields = [f.name for f in dataclasses.fields(mod)]
            if "_graphs_properties" in fields:
                params["_graphs_properties"] = graphs_properties
            mods.append((mod, params))

        self.modules = FENNIXModules(mods)

        self.__apply = self.modules.apply
        self._apply = jax.jit(self.modules.apply)

        self.set_energy_terms(energy_terms)

        # initialize the model

        inputs, rng_key = self.reinitialize_preprocessing(rng_key, example_data)

        if variables is not None:
            self.variables = variables
        elif rng_key is not None:
            self.variables = self.modules.init(rng_key, inputs)
        else:
            raise ValueError(
                "Either variables or a jax.random.PRNGKey must be provided for initialization"
            )

        self._initializing = False

    def set_energy_terms(self, energy_terms: Sequence[str] | None, jit=True) -> None:
        """ Set the energy terms to be computed by the model and prepare the energy and force functions."""
        object.__setattr__(self, "energy_terms", energy_terms)
        if energy_terms is None:
            def total_energy(variables, data):
                out = self.__apply(variables, data)
                coords = out["coordinates"]
                nsys = out["natoms"].shape[0]
                nat = coords.shape[0]
                dtype = coords.dtype
                e=jnp.zeros(nsys,dtype=dtype)
                eat = jnp.zeros(nat,dtype=dtype)
                out["total_energy"] = e
                out["atomic_energies"] = eat
                return e, out
            
            def energy_and_forces(variables, data):
                e,out = total_energy(variables, data)
                f=jnp.zeros_like(out["coordinates"])
                out["forces"] = f
                return e,f, out
            
            def energy_and_forces_and_virial(variables, data):
                e,f,out = energy_and_forces(variables, data)
                v=jnp.zeros((out["natoms"].shape[0],3,3),dtype=out["coordinates"].dtype)
                out["virial_tensor"] = v
                return e,f,v, out
        else:
            # build the energy and force functions
            def total_energy(variables, data):
                out = self.__apply(variables, data)
                atomic_energies = 0.0
                system_energies = 0.0
                species = out["species"]
                nsys = out["natoms"].shape[0]
                for term in self.energy_terms:
                    e = out[term]
                    if e.ndim > 1 and e.shape[-1] == 1:
                        e = jnp.squeeze(e, axis=-1)
                    if e.shape[0] == nsys and nsys != species.shape[0]:
                        system_energies += e
                        continue
                    assert e.shape == species.shape
                    atomic_energies += e
                # atomic_energies = jnp.squeeze(atomic_energies, axis=-1)
                if isinstance(atomic_energies, jnp.ndarray):
                    if "true_atoms" in out:
                        atomic_energies = jnp.where(out["true_atoms"], atomic_energies, 0.0)
                    out["atomic_energies"] = atomic_energies
                    energies = jax.ops.segment_sum(
                        atomic_energies,
                        data["batch_index"],
                        num_segments=len(data["natoms"]),
                    )
                else:
                    energies = 0.0

                if isinstance(system_energies, jnp.ndarray):
                    if "true_sys" in out:
                        system_energies = jnp.where(out["true_sys"], system_energies, 0.0)
                    out["system_energies"] = system_energies

                out["total_energy"] = energies + system_energies
                return out["total_energy"], out

            def energy_and_forces(variables, data):
                def _etot(variables, coordinates):
                    energy, out = total_energy(
                        variables, {**data, "coordinates": coordinates}
                    )
                    return energy.sum(), out

                de, out = jax.grad(_etot, argnums=1, has_aux=True)(
                    variables, data["coordinates"]
                )
                out["forces"] = -de

                return out["total_energy"], out["forces"], out

            def energy_and_forces_and_virial(variables, data):
                assert "cells" in data
                cells = data["cells"]
                batch_index = data["batch_index"]
                x = data["coordinates"]

                def _etot(variables, coordinates, cells):
                    reciprocal_cells = jnp.linalg.inv(cells)
                    energy, out = total_energy(
                        variables,
                        {
                            **data,
                            "coordinates": coordinates,
                            "cells": cells,
                            "reciprocal_cells": reciprocal_cells,
                        },
                    )
                    return energy.sum(), out

                (dedx, dedcells), out = jax.grad(_etot, argnums=(1, 2), has_aux=True)(
                    variables, x, cells
                )
                # dedx = jnp.einsum("sij,si->sj", reciprocal_cells[batch_index], deds)
                out["forces"] = -dedx
                fx = jax.ops.segment_sum(
                    dedx[:, :, None] * x[:, None, :],
                    batch_index,
                    num_segments=len(data["natoms"]),
                )

                # out["virial_tensor"] = (
                #     jnp.einsum("sik,skj->sij", dedcells, cells) + fx
                # )
                out["virial_tensor"] = jax.vmap(jnp.matmul)(dedcells, cells) + fx

                return out["total_energy"], out["forces"], out["virial_tensor"], out

        if jit:
            object.__setattr__(self, "_total_energy", jax.jit(total_energy))
            object.__setattr__(self, "_energy_and_forces", jax.jit(energy_and_forces))
            object.__setattr__(
                self,
                "_energy_and_forces_and_virial",
                jax.jit(energy_and_forces_and_virial),
            )
        else:
            object.__setattr__(self, "_total_energy", total_energy)
            object.__setattr__(self, "_energy_and_forces", energy_and_forces)
            object.__setattr__(
                self, "_energy_and_forces_and_virial", energy_and_forces_and_virial
            )

    def preprocess(self, **inputs) -> Dict[str, Any]:
        """apply preprocessing to the input data

        !!! This is not a pure function => do not apply jax transforms !!!"""
        preproc_state, out = self.preprocessing(self.preproc_state, inputs)
        object.__setattr__(self, "preproc_state", preproc_state)
        return out

    def reinitialize_preprocessing(
        self, rng_key: Optional[jax.random.PRNGKey] = None, example_data=None
    ) -> None:
        ### TODO ###
        if rng_key is None:
            rng_key_pre = jax.random.PRNGKey(0)
        else:
            rng_key, rng_key_pre = jax.random.split(rng_key)

        if example_data is None:
            rng_key_sys, rng_key_pre = jax.random.split(rng_key_pre)
            example_data = self.generate_dummy_system(rng_key_sys, n_atoms=10)

        preproc_state, inputs = self.preprocessing.init_with_output(example_data)
        object.__setattr__(self, "preproc_state", preproc_state)
        return inputs, rng_key

    def __call__(self, variables: Optional[dict] = None, **inputs) -> Dict[str, Any]:
        """Apply the FENNIX model (preprocess + modules)

        !!! This is not a pure function => do not apply jax transforms !!!
        if you want to apply jax transforms, use  self._apply(variables, inputs) which is pure and preprocess the input using self.preprocess
        """
        if variables is None:
            variables = self.variables
        inputs = self.preprocess(**inputs)
        output = self._apply(variables, inputs)
        if self.use_atom_padding:
            output = atom_unpadding(output)
        return output

    def total_energy(
        self, variables: Optional[dict] = None, **inputs
    ) -> Tuple[jnp.ndarray, Dict]:
        """compute the total energy of the system

        !!! This is not a pure function => do not apply jax transforms !!!
        if you want to apply jax transforms, use self._total_energy(variables, inputs) which is pure and preprocess the input using self.preprocess
        """
        if variables is None:
            variables = self.variables
        inputs = self.preprocess(**inputs)
        _, output = self._total_energy(variables, inputs)
        if self.use_atom_padding:
            output = atom_unpadding(output)
        return output["total_energy"], output

    def energy_and_forces(
        self, variables: Optional[dict] = None, **inputs
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """compute the total energy and forces of the system

        !!! This is not a pure function => do not apply jax transforms !!!
        if you want to apply jax transforms, use  self._energy_and_forces(variables, inputs) which is pure and preprocess the input using self.preprocess
        """
        if variables is None:
            variables = self.variables
        inputs = self.preprocess(**inputs)
        _, _, output = self._energy_and_forces(variables, inputs)
        if self.use_atom_padding:
            output = atom_unpadding(output)
        return output["total_energy"], output["forces"], output

    def energy_and_forces_and_virial(
        self, variables: Optional[dict] = None, **inputs
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """compute the total energy and forces of the system

        !!! This is not a pure function => do not apply jax transforms !!!
        if you want to apply jax transforms, use  self._energy_and_forces_and_virial(variables, inputs) which is pure and preprocess the input using self.preprocess
        """
        if variables is None:
            variables = self.variables
        inputs = self.preprocess(**inputs)
        _, _, _, output = self._energy_and_forces_and_virial(variables, inputs)
        if self.use_atom_padding:
            output = atom_unpadding(output)
        return output["total_energy"], output["forces"], output["virial_tensor"], output

    def remove_atom_padding(self, output):
        """ remove atom padding from the output """
        return atom_unpadding(output)

    def get_model(self) -> Tuple[FENNIXModules, Dict]:
        """ return the model and its variables"""
        return self.modules, self.variables

    def get_preprocessing(self) -> Tuple[PreprocessingChain, Dict]:
        """ return the preprocessing chain and its state"""
        return self.preprocessing, self.preproc_state

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "variables":
            if __value is not None:
                if not (
                    isinstance(__value, dict)
                    or isinstance(__value, OrderedDict)
                    or isinstance(__value, FrozenDict)
                ):
                    raise ValueError(f"{__name} must be a dict")
                object.__setattr__(self, __name, JaxConverter()(__value))
            else:
                raise ValueError(f"{__name} cannot be None")
        elif __name == "preproc_state":
            if __value is not None:
                if not (
                    isinstance(__value, dict)
                    or isinstance(__value, OrderedDict)
                    or isinstance(__value, FrozenDict)
                ):
                    raise ValueError(f"{__name} must be a FrozenDict")
                object.__setattr__(self, __name, freeze(JaxConverter()(__value)))
            else:
                raise ValueError(f"{__name} cannot be None")

        elif self._initializing:
            object.__setattr__(self, __name, __value)
        else:
            raise ValueError(f"{__name} attribute of FENNIX model is immutable.")

    def generate_dummy_system(
        self, rng_key: jax.random.PRNGKey, box_size=None, n_atoms: int = 10
    ) -> Dict[str, Any]:
        """
        Generate dummy system for initialization
        """
        if box_size is None:
            box_size = 2 * self.cutoff
            for g in self._graphs_properties.values():
                cutoff = g["cutoff"]
                if cutoff is not None:
                    box_size = min(box_size, 2 * g["cutoff"])
        coordinates = np.array(
            jax.random.uniform(rng_key, (n_atoms, 3), maxval=box_size)
        )
        species = np.ones((n_atoms,), dtype=np.int32)
        batch_index = np.zeros((n_atoms,), dtype=np.int32)
        natoms = np.array([n_atoms], dtype=np.int32)
        return {
            "species": species,
            "coordinates": coordinates,
            # "graph": graph,
            "batch_index": batch_index,
            "natoms": natoms,
        }

    def summarize(
        self, rng_key: jax.random.PRNGKey = None, example_data=None, **kwargs
    ) -> str:
        """ Summarize the model architecture and parameters """
        if rng_key is None:
            head = "Summarizing with example data:\n"
            rng_key = jax.random.PRNGKey(0)
        if example_data is None:
            head = "Summarizing with dummy 10 atoms system:\n"
            rng_key, rng_key_sys = jax.random.split(rng_key)
            example_data = self.generate_dummy_system(rng_key_sys, n_atoms=10)
        rng_key, rng_key_pre = jax.random.split(rng_key)
        _, inputs = self.preprocessing.init_with_output(example_data)
        return head + nn.tabulate(self.modules, rng_key, **kwargs)(inputs)

    def to_dict(self):
        """ return a dictionary representation of the model"""
        return {
            **self._input_args,
            "energy_terms": self.energy_terms,
            "variables": deepcopy(self.variables),
        }

    def save(self, filename):
        """ save the model to a file"""
        state_dict = self.to_dict()
        state_dict["preprocessing"] = [
            [k, v] for k, v in state_dict["preprocessing"].items()
        ]
        state_dict["modules"] = [[k, v] for k, v in state_dict["modules"].items()]
        with open(filename, "wb") as f:
            f.write(serialization.msgpack_serialize(state_dict))

    @classmethod
    def load(
        cls,
        filename,
        use_atom_padding=False,
        graph_config={},
    ):
        """ load a model from a file"""
        with open(filename, "rb") as f:
            state_dict = serialization.msgpack_restore(f.read())
        state_dict["preprocessing"] = {k: v for k, v in state_dict["preprocessing"]}
        state_dict["modules"] = {k: v for k, v in state_dict["modules"]}
        return cls(
            **state_dict,
            graph_config=graph_config,
            use_atom_padding=use_atom_padding,
        )
