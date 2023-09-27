import jax
import jax.numpy as jnp
import flax.linen as nn
import flax

from typing import Any, Sequence, Callable, Union, Optional, Tuple, Dict
from copy import deepcopy
import dataclasses

from .embeddings import EMBEDDINGS
from .encodings import ENCODINGS
from .nets import NETWORKS
from .preprocessing import GraphGenerator, PreprocessingChain,PREPROCESSING, JaxConverter
import numpy as np
from flax import serialization
from copy import deepcopy

_MODULES = {**EMBEDDINGS, **ENCODINGS, **NETWORKS}


class FENNIXModules(nn.Module):
    layers: Sequence[Tuple[nn.Module, Dict]]

    def __post_init__(self):
        if not isinstance(self.layers, Sequence):
            raise ValueError(
                f"'layers' must be a sequence, got '{type(self.layers).__name__}'."
            )
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs):
        if not self.layers:
            raise ValueError(f"Empty Sequential module {self.name}.")

        outputs = inputs
        for layer, prms in self.layers:
            outputs = layer(**prms)(outputs)
        return outputs


@dataclasses.dataclass
class FENNIX:
    """
    Static wrapper for FENNIX models

    The underlying model is a flax.nn.Sequential built from the `modules` dictionary
    which references registered modules in `_MODULES` and provides the parameters for initialization.

    Since the model is static and contains variables, it must be initialized right away with either
    `example_data` or `variables`. If `variables` is provided, it is used directly. If `example_data`
    is provided, the model is initialized with `example_data` and the resulting variables are stored
    in the wrapper.

    methods:
        * __call__  : calls the jitted apply method of the underlying model and sums the energy terms
                        in `energy_terms` provided in the constructor.
                        It returns the total energy (of each subsystem in the batch) and the output dict.

        * get_model : returns the underlying model and variables (useful for training)

    """

    cutoff: float
    modules: FENNIXModules
    variables: Dict
    preprocessing: PreprocessingChain
    energy_terms: Sequence[str]
    _apply: Callable[[Dict, Dict], Dict]
    _total_energy: Callable[[Dict, Dict], Tuple[jnp.ndarray, Dict]]
    _energy_and_forces: Callable[[Dict, Dict], Tuple[jnp.ndarray, jnp.ndarray, Dict]]
    _input_args: Dict
    init: bool = False
    preproc_state: Dict = None

    def __init__(
        self,
        cutoff: float,
        modules: dict,
        preprocessing: dict = dict(),
        example_data=None,
        rng_key: Optional[jax.random.PRNGKey] = None,
        variables: Optional[dict] = None,
        energy_terms=["energy"],
    ) -> None:
        
        self._input_args = {
            "cutoff": cutoff,
            "modules": deepcopy(modules),
            "preprocessing": deepcopy(preprocessing),
        }
        self.cutoff = cutoff
        self.energy_terms = energy_terms

        # add non-differentiable/non-jittable modules
        preprocessing_modules = [
            GraphGenerator(cutoff=cutoff),
        ]
        for name, params in preprocessing.items():
            key = str(params.pop("module_name")) if "module_name" in params else name
            mod = PREPROCESSING[key.upper()](**params)
            preprocessing_modules.append(mod)

        self.preprocessing = PreprocessingChain(preprocessing_modules)
        mods = [(JaxConverter,{})]
        # add preprocessing modules that should be differentiated/jitted
        # mods.append((JaxConverter, {})
        graphs_properties = {}
        for m in preprocessing_modules:
            if hasattr(m, "get_processor"):
                mods.append(m.get_processor())
            if hasattr(m, "get_graph_properties"):
                graphs_properties.update(m.get_graph_properties())

        # build the model
        modules = deepcopy(modules)
        modules_names = []
        for name, params in modules.items():
            key = str(params.pop("module_name")) if "module_name" in params else name
            if name in modules_names:
                raise ValueError(f"Module {name} already exists")
            modules_names.append(name)
            params["name"] = name
            mod = _MODULES[key.upper()]
            fields = [f.name for f in dataclasses.fields(mod)]
            if "_graphs_properties" in fields:
                params["_graphs_properties"]=graphs_properties
            mods.append((mod, params))

        self.modules = FENNIXModules(mods)

        self._apply = jax.jit(self.modules.apply)

        self.set_energy_terms(energy_terms)

        # initialize the model

        inputs,rng_key=self.reinitialize_preprocessing(rng_key, example_data)
        
        if variables is not None:
            self.variables = JaxConverter()(variables)
        elif rng_key is not None:
            self.variables = self.modules.init(rng_key, inputs)
        else:
            raise ValueError(
                "Either variables or a jax.random.PRNGKey must be provided for initialization"
            )

        self.init = True

    def set_energy_terms(self, energy_terms: Sequence[str]) -> None:
        object.__setattr__(self, "energy_terms", energy_terms)

        # build the energy and force functions
        @jax.jit
        def total_energy(variables, data):
            out = self._apply(variables, data)
            atomic_energies = 0.0
            for term in self.energy_terms:
                atomic_energies += out[term]
            atomic_energies = jnp.squeeze(atomic_energies, axis=-1)
            out["atomic_energies"] = atomic_energies
            energies = (
                jnp.zeros_like(data["natoms"], dtype=atomic_energies.dtype)
                .at[data["isys"]]
                .add(atomic_energies)
            )
            out["total_energy"] = energies
            return energies, out

        @jax.jit
        def energy_and_forces(variables, data):
            def _etot(variables, coordinates):
                energy, out = total_energy(
                    variables, {**data, "coordinates": coordinates}
                )
                return energy.sum(), out

            de, out = jax.grad(_etot, argnums=1, has_aux=True)(
                variables, data["coordinates"]
            )

            return out["total_energy"], -de, out

        object.__setattr__(self, "_total_energy", total_energy)
        object.__setattr__(self, "_energy_and_forces", energy_and_forces)

    def preprocess(self, *args, **kwargs) -> Dict[str, Any]:
        """apply preprocessing to the input data

        !!! This is not a pure function => do not apply jax transforms !!!"""
        out, preproc_state = self.preprocessing.apply(
            self.preproc_state, *args, **kwargs, mutable=["preprocessing"]
        )
        object.__setattr__(self, "preproc_state", preproc_state)
        return out

    def reinitialize_preprocessing(
        self, rng_key: Optional[jax.random.PRNGKey] = None, example_data=None, **kwargs
    ) -> None:
        ### TODO ###
        if rng_key is None:
            rng_key_pre = jax.random.PRNGKey(0)
        else:
            rng_key, rng_key_pre = jax.random.split(rng_key)

        if example_data is None:
            rng_key_sys,rng_key_pre = jax.random.split(rng_key_pre)
            example_data = self.generate_dummy_system(rng_key_sys, n_atoms=10)

        inputs,preproc_state = self.preprocessing.init_with_output(rng_key_pre, example_data)
        object.__setattr__(self, "preproc_state", preproc_state)
        return inputs,rng_key

    def __call__(
        self, *args, variables: Optional[dict] = None, **kwargs
    ) -> Dict[str, Any]:
        """Apply the FENNIX model (preprocess + modules)

        !!! This is not a pure function => do not apply jax transforms !!!
        if you want to apply jax transforms, use  self._apply(variables, inputs) which is pure and preprocess the input using self.preprocess
        """
        if variables is None:
            variables = self.variables
        inputs = self.preprocess(*args, **kwargs)
        return self._apply(variables, inputs)

    def total_energy(
        self, *args, variables: Optional[dict] = None, **kwargs
    ) -> Tuple[jnp.ndarray, Dict]:
        """compute the total energy of the system

        !!! This is not a pure function => do not apply jax transforms !!!
        if you want to apply jax transforms, use self._total_energy(variables, inputs) which is pure and preprocess the input using self.preprocess
        """
        if variables is None:
            variables = self.variables
        inputs = self.preprocess(*args, **kwargs)
        return self._total_energy(variables, inputs)

    def energy_and_forces(
        self, *args, variables: Optional[dict] = None, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """compute the total energy and forces of the system

        !!! This is not a pure function => do not apply jax transforms !!!
        if you want to apply jax transforms, use  self._energy_and_forces(variables, inputs) which is pure and preprocess the input using self.preprocess
        """
        if variables is None:
            variables = self.variables
        inputs = self.preprocess(*args, **kwargs)
        return self._energy_and_forces(variables, inputs)

    def get_model(self) -> Tuple[FENNIXModules, Dict]:
        return self.modules, self.variables

    def get_preprocessing(self) -> Tuple[PreprocessingChain, Dict]:
        return self.preprocessing, self.preproc_state

    def __setattr__(self, __name: str, __value: Any) -> None:
        if not self.init:
            object.__setattr__(self, __name, __value)
            return
        if __name == "variables":
            if __value is not None:
                if not isinstance(__value, dict):
                    raise ValueError("variables must be a dict")
                object.__setattr__(self, __name, __value)
            else:
                raise ValueError("variables cannot be None")
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
        coordinates = np.array(jax.random.uniform(rng_key, (n_atoms, 3), maxval=box_size))
        species = np.ones((n_atoms,), dtype=np.int32)
        edge_src, edge_dst = np.triu_indices(n_atoms, 1)
        edge_src, edge_dst = np.concatenate((edge_src, edge_dst)), np.concatenate(
            (edge_dst, edge_src)
        )
        graph = {"edge_src": edge_src, "edge_dst": edge_dst}
        isys = np.zeros((n_atoms,), dtype=np.int32)
        natoms = np.array([n_atoms], dtype=np.int32)
        return {
            "species": species,
            "coordinates": coordinates,
            # "graph": graph,
            "isys": isys,
            "natoms": natoms,
        }

    def summarize(
        self, rng_key: jax.random.PRNGKey = None, example_data=None, **kwargs
    ) -> str:
        if rng_key is None:
            head = "Summarizing with example data:\n"
            rng_key = jax.random.PRNGKey(0)
        if example_data is None:
            head = "Summarizing with dummy 10 atoms system:\n"
            rng_key, rng_key_sys = jax.random.split(rng_key)
            example_data = self.generate_dummy_system(rng_key_sys, n_atoms=10)
        rng_key, rng_key_pre = jax.random.split(rng_key)
        inputs, _ = self.preprocessing.init_with_output(rng_key_pre, example_data)
        return head + nn.tabulate(self.modules, rng_key, **kwargs)(inputs)
    
    def save(self,filename):
        state_dict = {
            **self._input_args,
            "energy_terms": self.energy_terms,
            "variables": self.variables,
        }
        with open(filename,'wb') as f:
            f.write(serialization.msgpack_serialize(state_dict))
    
    @classmethod
    def load(cls,filename):
        with open(filename,'rb') as f:
            state_dict = serialization.msgpack_restore(f.read())
        return cls(**state_dict)
