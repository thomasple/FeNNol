import os, io, sys
import numpy as np
from collections import defaultdict
import pickle
from flax import traverse_util
from typing import Dict, List, Tuple, Union, Optional, Callable
from .databases import DBDataset, H5Dataset
from ..models.preprocessing import AtomPadding

import json
import yaml

try:
    import tomlkit
except ImportError:
    tomlkit = None

try:
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError(
        "PyTorch is required for training. Install the CPU version from https://pytorch.org/get-started/locally/"
    )

from ..models import FENNIX

def load_configuration(config_file: str) -> Dict[str, any]:
    if config_file.endswith(".json"):
        parameters = json.load(open(config_file))
    elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
        parameters = yaml.load(open(config_file), Loader=yaml.FullLoader)
    elif tomlkit is not None and config_file.endswith(".toml"):
        parameters = tomlkit.loads(open(config_file).read())
    else:
        supported_formats = [".json", ".yaml", ".yml"]
        if tomlkit is not None:
            supported_formats.append(".toml")
        raise ValueError(
            f"Unknown config file format. Supported formats: {supported_formats}"
        )
    return parameters


def load_dataset(training_parameters, rename_refs=[], infinite_iterator=False, atom_padding=False):
    """
    Load a dataset from a pickle file and return two iterators for training and validation batches.

    Args:
        training_parameters (dict): A dictionary with the following keys:
            - 'dspath': str. Path to the pickle file containing the dataset.
            - 'batch_size': int. Number of samples per batch.
        rename_refs (list, optional): A list of strings with the names of the reference properties to rename.
            Default is an empty list.

    Returns:
        tuple: A tuple of two infinite iterators, one for training batches and one for validation batches.
            For each element in the batch, we expect a "species" key with the atomic numbers of the atoms in the system. Arrays are concatenated along the first axis and the following keys are added to distinguish between the systems:
            - 'natoms': np.ndarray. Array with the number of atoms in each system.
            - 'batch_index': np.ndarray. Array with the index of the system to which each atom
            if the keys "forces", "total_energy", "atomic_energies" or any of the elements in rename_refs are present, the keys are renamed by prepending "true_" to the key name.
    """
    # rename_refs = set(["forces", "total_energy", "atomic_energies"] + list(rename_refs))
    pbc_training = training_parameters.get("pbc_training", False)

    if pbc_training:
        print("Periodic boundary conditions are active.")
        length_nopbc = training_parameters.get("length_nopbc", 1000.0)
        minimum_image = training_parameters.get("minimum_image", False)
        ase_nblist = training_parameters.get("ase_nblist", False)

        def collate_fn_(batch):
            output = defaultdict(list)
            for i, d in enumerate(batch):
                nat = d["species"].shape[0]
                output["natoms"].append(np.asarray([nat]))
                output["batch_index"].append(np.asarray([i] * nat))
                if "total_charge" not in d:
                    output["total_charge"].append(np.asarray(0.0, dtype=np.float32))
                if "cell" not in d:
                    cell = np.asarray(
                        [
                            [length_nopbc, 0.0, 0.0],
                            [0.0, length_nopbc, 0.0],
                            [0.0, 0.0, length_nopbc],
                        ]
                    )
                else:
                    cell = d["cell"]
                output["cells"].append(cell.reshape(1, 3, 3))

                for k, v in d.items():
                    if k == "cell":
                        continue
                    output[k].append(np.asarray(v))
            for k, v in output.items():
                if v[0].ndim == 0:
                    output[k] = np.stack(v)
                else:
                    output[k] = np.concatenate(v, axis=0)
            # for key in rename_refs:
            #    if key in output:
            #        output["true_" + key] = output.pop(key)

            output["minimum_image"] = minimum_image
            output["ase_nblist"] = ase_nblist
            output["training_flag"] = True
            return output

    else:

        def collate_fn_(batch):
            output = defaultdict(list)
            for i, d in enumerate(batch):
                if "cell" in d:
                    raise ValueError(
                        "Activate pbc_training to use periodic boundary conditions."
                    )
                nat = d["species"].shape[0]
                output["natoms"].append(np.asarray([nat]))
                output["batch_index"].append(np.asarray([i] * nat))
                if "total_charge" not in d:
                    output["total_charge"].append(np.asarray([0.0], dtype=np.float32))
                for k, v in d.items():
                    output[k].append(np.asarray(v))
            for k, v in output.items():
                if v[0].ndim == 0:
                    output[k] = np.stack(v)
                else:
                    output[k] = np.concatenate(v, axis=0)
            # for key in rename_refs:
            #    if key in output:
            #        output["true_" + key] = output.pop(key)

            output["training_flag"] = True
            return output
    
    if atom_padding:
        padder = AtomPadding()
        padder_state = padder.init()
        def collate_with_padding(batch):
            output = collate_fn_(batch)
            padder_state_up,output = padder(padder_state,output)
            padder_state.update(padder_state_up)
            return output
        
        collate_fn = collate_with_padding
    else:
        collate_fn = collate_fn_

    # dspath = "dataset_ani1ccx.pkl"
    dspath = training_parameters.get("dspath", None)
    if dspath is None:
        raise ValueError("Dataset path 'training/dspath' should be specified.")
    print(f"Loading dataset from {dspath}...")
    # print(f"   the following keys will be renamed if present : {rename_refs}")
    if dspath.endswith(".db"):
        dataset = {}
        dataset["training"] = DBDataset(dspath, table="training")
        dataset["validation"] = DBDataset(dspath, table="validation")
    elif dspath.endswith(".h5") or dspath.endswith(".hdf5"):
        dataset = {}
        dataset["training"] = H5Dataset(dspath, table="training")
        dataset["validation"] = H5Dataset(dspath, table="validation")
    else:
        with open(dspath, "rb") as f:
            dataset = pickle.load(f)

    batch_size = training_parameters.get("batch_size", 16)
    shuffle = training_parameters.get("shuffle_dataset", True)
    dataloader_validation = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    dataloader_training = DataLoader(
        dataset["training"],
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

    print("Dataset loaded.")

    input_keys = [
        "species",
        "coordinates",
        "natoms",
        "batch_index",
        "training_flag",
        "total_charge",
    ]
    if pbc_training:
        input_keys += ["cells", "minimum_image"]
    if atom_padding:
        input_keys += ["true_atoms","true_sys"]
    additional_input_keys = training_parameters.get("additional_input_keys", [])
    input_keys += additional_input_keys

    def extract_inputs(batch):
        return {k: batch[k] for k in input_keys}

    if not infinite_iterator:
        return dataloader_training, dataloader_validation,extract_inputs

    def next_batch_factory(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    validation_iterator = next_batch_factory(dataloader_validation)
    training_iterator = next_batch_factory(dataloader_training)

    return training_iterator, validation_iterator, extract_inputs


def load_model(
    parameters: Dict[str, any],
    model_file: Optional[str] = None,
    rng_key: Optional[str] = None,
) -> FENNIX:
    """
    Load a FENNIX model from a file or create a new one.

    Args:
        parameters (dict): A dictionary of parameters for the model.
        model_file (str, optional): The path to a saved model file to load.

    Returns:
        FENNIX: A FENNIX model object.
    """
    print_model = parameters["training"].get("print_model", False)
    if model_file is None:
        model_file = parameters.get("model_file", None)
    if model_file is not None and os.path.exists(model_file):
        model = FENNIX.load(model_file, use_atom_padding=False)
        if print_model:
            print(model.summarize())
        print(f"Restored model from '{model_file}'.")
    else:
        assert (
            rng_key is not None
        ), "rng_key must be specified if model_file is not provided."
        model_params = parameters["model"]
        model = FENNIX(**model_params, rng_key=rng_key, use_atom_padding=False)
        if print_model:
            print(model.summarize())
    return model


def copy_parameters(variables, variables_ref, params):
    def merge_params(full_path_, v, v_ref):
        full_path = "/".join(full_path_[1:]).lower()
        status = (False, "")
        for path in params:
            if full_path.startswith(path.lower()) and len(path) > len(status[1]):
                status = (True, path)
        return v_ref if status[0] else v

    flat = traverse_util.flatten_dict(variables, keep_empty_nodes=False)
    flat_ref = traverse_util.flatten_dict(variables_ref, keep_empty_nodes=False)
    return traverse_util.unflatten_dict(
        {
            k: merge_params(k, v, flat_ref[k]) if k in flat_ref else v
            for k, v in flat.items()
        }
    )


class TeeLogger(object):
    def __init__(self, name):
        self.file = io.TextIOWrapper(open(name, "wb"), write_through=True)
        self.stdout = None

    def __del__(self):
        self.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def close(self):
        self.file.close()

    def flush(self):
        self.file.flush()

    def bind_stdout(self):
        if isinstance(sys.stdout, TeeLogger):
            raise ValueError("stdout already bound to a Tee instance.")
        if self.stdout is not None:
            raise ValueError("stdout already bound.")
        self.stdout = sys.stdout
        sys.stdout = self

    def unbind_stdout(self):
        if self.stdout is None:
            raise ValueError("stdout is not bound.")
        sys.stdout = self.stdout
