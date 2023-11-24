import os, io, sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
from collections import defaultdict
import pickle
from flax import traverse_util
import json
from copy import deepcopy
from typing import Dict, List, Tuple, Union,Optional, Callable
try:
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError(
        "PyTorch is required for training. Install the CPU version from https://pytorch.org/get-started/locally/"
    )

from ..models import FENNIX
from ..utils import AtomicUnits as au


def load_dataset(training_parameters, rename_refs=[]):
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
            - 'isys': np.ndarray. Array with the index of the system to which each atom 
            if the keys "forces", "total_energy", "atomic_energies" or any of the elements in rename_refs are present, the keys are renamed by prepending "true_" to the key name.
    """
    rename_refs = set(["forces", "total_energy", "atomic_energies"] + list(rename_refs))

    def collate_fn(batch):
        output = defaultdict(list)
        for i, d in enumerate(batch):
            nat = d["species"].shape[0]
            output["natoms"].append(np.asarray([nat]))
            output["isys"].append(np.asarray([i] * nat))
            for k, v in d.items():
                output[k].append(np.asarray(v))
        for k, v in output.items():
            if v[0].ndim == 0:
                output[k] = np.stack(v)
            else:
                output[k] = np.concatenate(v, axis=0)
        for key in rename_refs:
            if key in output:
                output["true_" + key] = output.pop(key)
        return output

    # dspath = "dataset_ani1ccx.pkl"
    dspath = training_parameters.get("dspath", None)
    if dspath is None:
        raise ValueError("Dataset path 'training/dspath' should be specified.")
    print(f"Loading dataset at {dspath}...")
    with open(dspath, "rb") as f:
        dataset = pickle.load(f)

    batch_size = training_parameters.get("batch_size", 16)
    dataloader_validation = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataloader_training = DataLoader(
        dataset["training"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    def next_batch_factory(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    validation_iterator = next_batch_factory(dataloader_validation)
    training_iterator = next_batch_factory(dataloader_training)

    print("Dataset loaded.")
    return training_iterator, validation_iterator


def load_model(parameters:Dict[str,any], model_file:Optional[str]=None) -> FENNIX:
    """
    Load a FENNIX model from a file or create a new one.

    Args:
        parameters (dict): A dictionary of parameters for the model.
        model_file (str, optional): The path to a saved model file to load.

    Returns:
        FENNIX: A FENNIX model object.
    """
    rng_seed = parameters.get("rng_seed", 0)
    key = jax.random.PRNGKey(rng_seed)
    print_model = parameters.get("print_model", False)
    if model_file is None:
        model_file = parameters.get("model_file", None)
    if model_file is not None and os.path.exists(model_file):
        model = FENNIX.load(model_file, use_atom_padding=True)
        if print_model:
            print(model.summarize())
        print(f"Restored model from '{model_file}'.")
    else:
        model_params = parameters["model"]
        model = FENNIX(**model_params, rng_key=key, use_atom_padding=True)
        if print_model:
            print(model.summarize())
    return model


def get_loss_definition(training_parameters: Dict[str, any]) -> Tuple[Dict[str, any], List[str]]:
    """
    Returns the loss definition and a list of renamed references.

    Args:
        training_parameters (dict): A dictionary containing training parameters.

    Returns:
        tuple: A tuple containing:
            - loss_definition (dict): A dictionary containing the loss definition.
            - rename_refs (list): A list of renamed references.
    """
    default_loss_type = training_parameters.get("default_loss_type", "log_cosh")
    loss_definition = deepcopy(
        training_parameters.get(
            "loss",
            {
                "energy": {
                    "key": "total_energy",
                    "ref": "e_formation_dft",
                    "type": "log_cosh",
                    "weight": 1.0,
                    "threshold": 2.0,
                    "unit": "kcalpermol",
                },
                "forces": {
                    "ref": "true_forces",
                    "type": "log_cosh",
                    "weight": 1000.0,
                    "threshold": 10.0,
                    "unit": "kcalpermol",
                },
            },
        )
    )
    rename_refs = []
    for k in loss_definition.keys():
        loss_prms = loss_definition[k]
        if "unit" in loss_definition[k]:
            loss_prms["mult"] = 1.0 / au.get_multiplier(loss_prms["unit"])
        else:
            loss_prms["mult"] = 1.0
        if "key" not in loss_prms:
            loss_prms["key"] = k
        if "type" not in loss_prms:
            loss_prms["type"] = default_loss_type
        if "weight" not in loss_prms:
            loss_prms["weight"] = 1.0
        assert loss_prms["weight"] >= 0.0, "Loss weight must be positive"
        if "ref" in loss_prms:
            if loss_prms["ref"] in [
                "forces",
                "total_energy",
                "atomic_energies",
                loss_prms["key"],
            ]:
                rename_refs.append(loss_prms["ref"])
                loss_prms["ref"] = "true_" + loss_prms["ref"]
        if "threshold" in loss_prms:
            assert loss_prms["threshold"] > 1.0, "Threshold must be greater than 1.0"

    return loss_definition, rename_refs


def get_optimizer(training_parameters: Dict[str, any], variables: Dict, initial_lr: float) -> optax.GradientTransformation:
    """
    Returns an optax.GradientTransformation object that can be used to optimize the model parameters.

    Args:
    - training_parameters: A dictionary containing the training parameters.
    - variables: A  pytree containing the model parameters.
    - initial_lr: The initial learning rate.

    Returns:
    - An optax.GradientTransformation object that can be used to optimize the model parameters.
    """

    default_status = str(training_parameters.get("default_status", "trainable")).lower()
    assert default_status in [
        "trainable",
        "frozen",
    ], f"Default status must be 'trainable' or 'frozen', got {default_status}"

    # find frozen and trainable parameters
    frozen = training_parameters.get("frozen", [])
    trainable = training_parameters.get("trainable", [])

    def training_status(full_path, v):
        full_path = "/".join(full_path[1:]).lower()
        status = (default_status, "")
        for path in frozen:
            if full_path.startswith(path.lower()) and len(path) > len(status[1]):
                status = ("frozen", path)
        for path in trainable:
            if full_path.startswith(path.lower()) and len(path) > len(status[1]):
                status = ("trainable", path)
        return status[0]

    params_partition = traverse_util.path_aware_map(training_status, variables)
    if len(frozen) > 0 or len(trainable) > 0:
        print("params partition:")
        print(json.dumps(params_partition, indent=2, sort_keys=False))

    ## Gradient preprocessing
    grad_preprocessing = []

    # weight decay
    weight_decay = training_parameters.get("weight_decay", 0.0)
    assert weight_decay >= 0.0, "Weight decay must be positive"
    decay_targets = training_parameters.get(
        "decay_targets", [""]
    )  # by default, decay all parameters with [""]

    def decay_status(full_path, v):
        full_path = "/".join(full_path).lower()
        status = False
        for path in decay_targets:
            if full_path.startswith("params/" + path.lower()):
                status = True
        return status

    decay_mask = traverse_util.path_aware_map(decay_status, variables)
    if weight_decay > 0.0:
        print("weight decay:", weight_decay)
        print(json.dumps(decay_mask, indent=2, sort_keys=False))
        grad_preprocessing.append(
            optax.add_decayed_weights(weight_decay=weight_decay, mask=decay_mask)
        )

    # gradient clipping
    clip_threshold = training_parameters.get("gradient_clipping", -1.)
    if clip_threshold > 0.0:
        print("Adaptive gradient clipping threshold:", clip_threshold)
        grad_preprocessing.append(optax.adaptive_grad_clip(clip_threshold))

    if len(grad_preprocessing) == 0:
        grad_preprocessing.append(optax.identity())
    ## define optimizer chain
    optimizer_ = optax.chain(
        optax.chain(*grad_preprocessing),
        optax.inject_hyperparams(optax.adabelief)(learning_rate=initial_lr),
    )
    partition_optimizer = {"trainable": optimizer_, "frozen": optax.set_to_zero()}
    return optax.multi_transform(partition_optimizer, params_partition)


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
