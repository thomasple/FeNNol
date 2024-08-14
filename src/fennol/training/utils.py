import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Dict, List, Tuple
import optax
from copy import deepcopy
from flax import traverse_util
import json
import re

from ..utils import deep_update, AtomicUnits as au
from ..models import FENNIX


def get_training_parameters(
    parameters: Dict[str, any], stage: int = -1
) -> Dict[str, any]:
    params = deepcopy(parameters["training"])
    if "stages" not in params:
        return params

    stages: dict = params.pop("stages")
    stage_keys = list(stages.keys())
    if stage < 0:
        stage = len(stage_keys) + stage
    assert stage >= 0 and stage < len(
        stage_keys
    ), f"Stage {stage} not found in training parameters"
    for i in range(stage + 1):
        ## remove end_event from previous stage ##
        if i > 0 and "end_event" in params:
            params.pop("end_event")
        ## incrementally update training parameters ##
        stage_params = stages[stage_keys[i]]
        params = deep_update(params, stage_params)
    return params


def get_loss_definition(
    training_parameters: Dict[str, any], model_energy_unit:str = "Ha"  # , manual_renames: List[str] = []
) -> Tuple[Dict[str, any], List[str], List[str]]:
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
    loss_definition = deepcopy(training_parameters["loss"])
    used_keys = []
    ref_keys = []
    energy_mult = au.get_multiplier(model_energy_unit)
    for k in loss_definition.keys():
        loss_prms = loss_definition[k]
        if "energy_unit" in loss_prms:
            loss_prms["mult"] = energy_mult/au.get_multiplier(loss_prms["energy_unit"])
            if "unit" in loss_prms:
                print("Warning: Both 'unit' and 'energy_unit' are defined for loss component",k, " -> using 'energy_unit'")
            loss_prms["unit"] = loss_prms["energy_unit"]
        elif "unit" in loss_prms:
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
        if "threshold" in loss_prms:
            assert loss_prms["threshold"] > 1.0, "Threshold must be greater than 1.0"
        if "ref" in loss_prms:
            ref = loss_prms["ref"]
            if not (ref.startswith("model_ref/") or ref.startswith("model/")):
                ref_keys.append(ref)
        if "ds_weight" in loss_prms:
            ref_keys.append(loss_prms["ds_weight"])

        used_keys.append(loss_prms["key"])

    # rename_refs = list(
    #     set(["forces", "total_energy", "atomic_energies"] + manual_renames + used_keys)
    # )

    # for k in loss_definition.keys():
    #     loss_prms = loss_definition[k]
    #     if "ref" in loss_prms:
    #         if loss_prms["ref"] in rename_refs:
    #             loss_prms["ref"] = "true_" + loss_prms["ref"]

    return loss_definition, list(set(used_keys)), list(set(ref_keys))


def get_optimizer(
    training_parameters: Dict[str, any], variables: Dict, initial_lr: float
) -> optax.GradientTransformation:
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
    grad_processing = []

    # zero nans
    zero_nans = training_parameters.get("zero_nans", False)
    if zero_nans:
        grad_processing.append(optax.zero_nans())

    # gradient clipping
    clip_threshold = training_parameters.get("gradient_clipping", -1.0)
    if clip_threshold > 0.0:
        print("Adaptive gradient clipping threshold:", clip_threshold)
        grad_processing.append(optax.adaptive_grad_clip(clip_threshold))

    # OPTIMIZER
    optimizer_name = training_parameters.get("optimizer", "adabelief")
    optimizer = eval(
        optimizer_name,
        {"__builtins__": None},
        {**optax.__dict__},
    )
    print("Optimizer:", optimizer_name)
    optimizer_configuration = training_parameters.get("optimizer_config", {})
    optimizer_configuration["learning_rate"] = 1.0
    grad_processing.append(optimizer(**optimizer_configuration))

    # weight decay
    weight_decay = training_parameters.get("weight_decay", 0.0)
    assert weight_decay >= 0.0, "Weight decay must be positive"
    decay_targets = training_parameters.get(
        "decay_targets", [""]
    ) 

    def decay_status(full_path, v):
        full_path = "/".join(full_path).lower()
        status = False
        # print(full_path,re.match(r'^params\/', full_path))
        for path in decay_targets:
            if re.match(r'^params/'+path.lower(), full_path):
                status = True
            # if full_path.startswith("params/" + path.lower()):
            #     status = True
        return status

    decay_mask = traverse_util.path_aware_map(decay_status, variables)
    if weight_decay > 0.0:
        print("weight decay:", weight_decay)
        print(json.dumps(decay_mask, indent=2, sort_keys=False))
        grad_processing.append(
            optax.add_decayed_weights(weight_decay=-weight_decay, mask=decay_mask)
        )
    
    if zero_nans:
        grad_processing.append(optax.zero_nans())

    # learning rate
    grad_processing.append(optax.inject_hyperparams(optax.scale)(step_size=initial_lr))

    ## define optimizer chain
    optimizer_ = optax.chain(
        *grad_processing,
    )
    partition_optimizer = {"trainable": optimizer_, "frozen": optax.set_to_zero()}
    return optax.multi_transform(partition_optimizer, params_partition)


def get_train_step_function(
    loss_definition: Dict,
    model: FENNIX,
    evaluate: Callable,
    optimizer: optax.GradientTransformation,
    ema: Optional[optax.GradientTransformation] = None,
    model_ref: Optional[FENNIX] = None,
    compute_ref_coords: bool = False,
    jit: bool = True,
):
    def train_step(
        data,
        inputs,
        variables,
        opt_st,
        variables_ema=None,
        ema_st=None,
        inputs_ref=None,
    ):

        def loss_fn(variables):
            if model_ref is not None:
                output_ref = evaluate(model_ref, model_ref.variables, inputs)
                # _, _, output_ref = model_ref._energy_and_forces(model_ref.variables, data)
            output = evaluate(model, variables, inputs)
            # _, _, output = model._energy_and_forces(variables, data)
            if compute_ref_coords:
                if inputs_ref is None:
                    raise ValueError(
                        "train_step was setup with compute_ref_coords=True but inputs_ref was not provided"
                    )
                output_data_ref = evaluate(model, variables, inputs_ref)
            loss_tot = 0.0
            for loss_prms in loss_definition.values():
                use_ref_mask = False
                predicted = output[loss_prms["key"]]
                if "remove_ref_sys" in loss_prms and loss_prms["remove_ref_sys"]:
                    assert compute_ref_coords, "compute_ref_coords must be True"
                    predicted = predicted - output_data_ref[loss_prms["key"]]
                if "ref" in loss_prms:
                    if loss_prms["ref"].startswith("model_ref/"):
                        assert model_ref is not None, "model_ref must be provided"
                        try:
                            ref = output_ref[loss_prms["ref"][10:]] * loss_prms["mult"]
                        except KeyError:
                            raise KeyError(
                                f"Reference key '{loss_prms['ref'][10:]}' not found in model_ref output. Keys available: {output_ref.keys()}"
                            )
                    elif loss_prms["ref"].startswith("model/"):
                        try:
                            ref = output[loss_prms["ref"][6:]] * loss_prms["mult"]
                        except KeyError:
                            raise KeyError(
                                f"Reference key '{loss_prms['ref'][6:]}' not found in model output. Keys available: {output.keys()}"
                            )
                    else:
                        try:
                            ref = data[loss_prms["ref"]] * loss_prms["mult"]
                        except KeyError:
                            raise KeyError(
                                f"Reference key '{loss_prms['ref']}' not found in data. Keys available: {data.keys()}"
                            )
                        if loss_prms["ref"] + "_mask" in data:
                            use_ref_mask = True
                            ref_mask = data[loss_prms["ref"] + "_mask"]
                else:
                    ref = jnp.zeros_like(predicted)

                if loss_prms["type"] in ["ensemble_nll", "ensemble_crps"]:
                    ensemble_axis = loss_prms.get("ensemble_axis", -1)
                    if "ensemble_subsample" in loss_prms and "rng_key" in output:
                        ns = min(
                            loss_prms["ensemble_subsample"],
                            predicted.shape[ensemble_axis],
                        )
                        key, subkey = jax.random.split(output["rng_key"])
                        predicted = jax.lax.slice_in_dim(
                            jax.random.permutation(
                                subkey, predicted, axis=ensemble_axis, independent=True
                            ),
                            start_index=0,
                            limit_index=ns,
                            axis=ensemble_axis,
                        )
                        output["rng_key"] = key
                    predicted_var = predicted.var(axis=ensemble_axis, ddof=1)
                    predicted = predicted.mean(axis=ensemble_axis)

                if predicted.ndim > 1 and predicted.shape[-1] == 1:
                    predicted = jnp.squeeze(predicted, axis=-1)

                if ref.ndim > 1 and ref.shape[-1] == 1:
                    ref = jnp.squeeze(ref, axis=-1)

                natoms = jnp.where(inputs["true_sys"], inputs["natoms"], 1)
                per_atom = False
                shape_mask = [ref.shape[0]] + [1] * (len(ref.shape) - 1)
                # print(loss_prms["key"],predicted.shape,loss_prms["ref"],ref.shape)
                natscale = 1.0
                if ref.shape[0] == output["batch_index"].shape[0]:
                    ## shape is number of atoms
                    truth_mask = inputs["true_atoms"]
                    if "ds_weight" in loss_prms:
                        weight_key = loss_prms["ds_weight"]
                        natscale = data[weight_key][output["batch_index"]].reshape(
                            *shape_mask
                        )
                elif ref.shape[0] == natoms.shape[0]:
                    ## shape is number of systems
                    truth_mask = inputs["true_sys"]
                    if loss_prms.get("per_atom", False):
                        per_atom = True
                        ref = ref / natoms.reshape(*shape_mask)
                        predicted = predicted / natoms.reshape(*shape_mask)
                    if "nat_pow" in loss_prms:
                        natscale = (
                            1.0 / natoms.reshape(*shape_mask) ** loss_prms["nat_pow"]
                        )
                    if "ds_weight" in loss_prms:
                        weight_key = loss_prms["ds_weight"]
                        natscale = natscale * data[weight_key].reshape(*shape_mask)
                else:
                    truth_mask = jnp.ones(ref.shape[0], dtype=bool)

                if use_ref_mask:
                    truth_mask = truth_mask * ref_mask.astype(bool)

                nel = jnp.maximum(
                    (float(np.prod(ref.shape)) / float(truth_mask.shape[0]))
                    * jnp.sum(truth_mask).astype(jnp.float32),
                    1.0,
                )
                truth_mask = truth_mask.reshape(*shape_mask)

                ref = ref * truth_mask
                predicted = predicted * truth_mask

                loss_type = loss_prms["type"]
                if loss_type == "mse":
                    loss = jnp.sum(natscale * (predicted - ref) ** 2)
                elif loss_type == "log_cosh":
                    loss = jnp.sum(natscale * optax.log_cosh(predicted, ref))
                elif loss_type == "rmse+mae":
                    loss = (
                        jnp.sum(natscale * (predicted - ref) ** 2)
                    ) ** 0.5 + jnp.sum(natscale * jnp.abs(predicted - ref))
                elif loss_type == "ensemble_nll":
                    predicted_var = predicted_var * truth_mask + (1.0 - truth_mask)
                    loss = 0.5 * jnp.sum(
                        natscale
                        * truth_mask
                        * (
                            jnp.log(predicted_var)
                            + (ref - predicted) ** 2 / predicted_var
                        )
                    )
                elif loss_type == "ensemble_crps":
                    if per_atom:
                        predicted_var = predicted_var / natoms.reshape(*shape_mask)
                    predicted_var = predicted_var * truth_mask + (1.0 - truth_mask)
                    sigma = predicted_var**0.5
                    dy = (ref - predicted) / sigma
                    Phi = 0.5 * (1.0 + jax.scipy.special.erf(dy / 2**0.5))
                    phi = jnp.exp(-0.5 * dy**2) / (2 * jnp.pi) ** 0.5
                    loss = jnp.sum(
                        natscale * truth_mask * sigma * (dy * (2 * Phi - 1.0) + 2 * phi)
                    )
                elif loss_type == "evidential":
                    evidence = loss_prms["evidence_key"]
                    nu, alpha, beta = jnp.split(output[evidence], 3, axis=-1)
                    gamma = predicted
                    nu = nu.reshape(shape_mask)
                    alpha = alpha.reshape(shape_mask)
                    beta = beta.reshape(shape_mask)
                    nu = jnp.where(truth_mask, nu, 1.0)
                    alpha = jnp.where(truth_mask, alpha, 1.0)
                    beta = jnp.where(truth_mask, beta, 1.0)
                    omega = 2 * beta * (1 + nu)
                    lg = jax.scipy.special.gammaln(alpha) - jax.scipy.special.gammaln(
                        alpha + 0.5
                    )
                    ls = 0.5 * jnp.log(jnp.pi / nu) - alpha * jnp.log(omega)
                    lt = (alpha + 0.5) * jnp.log(omega + nu * (gamma - ref) ** 2)
                    wst = (
                        (beta * (1 + nu) / (alpha * nu)) ** 0.5
                        if loss_prms.get("normalize_evidence", True)
                        else 1.0
                    )
                    lr = (
                        loss_prms.get("lambda_evidence", 1.0)
                        * jnp.abs(gamma - ref)
                        * nu
                        / wst
                    )
                    r = loss_prms.get("evidence_ratio", 1.0)
                    le = (
                        loss_prms.get("lambda_evidence_diff", 0.0)
                        * (nu - r * 2 * alpha) ** 2
                    )
                    lb = loss_prms.get("lambda_evidence_beta", 0.0) * beta
                    loss = lg + ls + lt + lr + le + lb

                    loss = jnp.sum(natscale * loss * truth_mask)
                elif loss_type == "raw":
                    loss = jnp.sum(natscale * predicted)
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")

                loss_tot = loss_tot + loss_prms["weight"] * loss / nel
            
            if compute_ref_coords:
                o = (output, output_data_ref)
            else:
                o = output

            return loss_tot, o

        (loss,o), grad = jax.value_and_grad(loss_fn,has_aux=True)(variables)
        updates, opt_st = optimizer.update(grad, opt_st, params=variables)
        variables = optax.apply_updates(variables, updates)
        if ema is not None:
            if variables_ema is None or ema_st is None:
                raise ValueError(
                    "train_step was setup with ema but either variables_ema or ema_st was not provided"
                )
            variables_ema, ema_st = ema.update(variables, ema_st)
            return loss, variables, opt_st, variables_ema, ema_st,o
        else:
            return loss, variables, opt_st,o

    if jit:
        return jax.jit(train_step)
    return train_step


def get_validation_function(
    loss_definition: Dict,
    model: FENNIX,
    evaluate: Callable,
    model_ref: Optional[FENNIX] = None,
    compute_ref_coords: bool = False,
    return_targets: bool = False,
    jit: bool = True,
):

    def validation(data, inputs, variables, inputs_ref=None):
        if model_ref is not None:
            output_ref = evaluate(model_ref, model_ref.variables, inputs)
            # _, _, output_ref = model_ref._energy_and_forces(model_ref.variables, data)
        output = evaluate(model, variables, inputs)
        # _, _, output = model._energy_and_forces(variables, data)
        if compute_ref_coords:
            if inputs_ref is None:
                raise ValueError(
                    "validation was setup with compute_ref_coords but inputs_ref was not provided"
                )
            output_data_ref = evaluate(model, variables, inputs_ref)
        rmses = {}
        maes = {}
        if return_targets:
            targets = {}

        natoms = jnp.where(inputs["true_sys"], inputs["natoms"], 1)

        for name, loss_prms in loss_definition.items():
            do_validation = loss_prms.get("validate", True)
            if not do_validation:
                continue
            predicted = output[loss_prms["key"]]
            use_ref_mask = False
            if "remove_ref_sys" in loss_prms and loss_prms["remove_ref_sys"]:
                assert compute_ref_coords, "compute_ref_coords must be True"
                predicted = predicted - output_data_ref[loss_prms["key"]]
            if "ref" in loss_prms:
                if loss_prms["ref"].startswith("model_ref/"):
                    assert model_ref is not None, "model_ref must be provided"
                    ref = output_ref[loss_prms["ref"][10:]] * loss_prms["mult"]
                elif loss_prms["ref"].startswith("model/"):
                    ref = output[loss_prms["ref"][6:]] * loss_prms["mult"]
                else:
                    ref = data[loss_prms["ref"]] * loss_prms["mult"]
                    if loss_prms["ref"] + "_mask" in data:
                        use_ref_mask = True
                        ref_mask = data[loss_prms["ref"] + "_mask"]
            else:
                ref = jnp.zeros_like(predicted)

            if loss_prms["type"] in ["ensemble_nll", "ensemble_crps"]:
                axis = loss_prms.get("ensemble_axis", -1)
                predicted = predicted.mean(axis=axis)

            if predicted.ndim > 1 and predicted.shape[-1] == 1:
                predicted = jnp.squeeze(predicted, axis=-1)

            if ref.ndim > 1 and ref.shape[-1] == 1:
                ref = jnp.squeeze(ref, axis=-1)

            shape_mask = [ref.shape[0]] + [1] * (len(predicted.shape) - 1)
            if ref.shape[0] == output["batch_index"].shape[0]:
                ## shape is number of atoms
                truth_mask = inputs["true_atoms"]
            elif ref.shape[0] == natoms.shape[0]:
                ## shape is number of systems
                truth_mask = inputs["true_sys"]
                if loss_prms.get("per_atom_validation", False):
                    ref = ref / natoms.reshape(*shape_mask)
                    predicted = predicted / natoms.reshape(*shape_mask)
            else:
                truth_mask = jnp.ones(ref.shape[0], dtype=bool)

            if use_ref_mask:
                truth_mask = truth_mask * ref_mask.astype(bool)

            nel = jnp.maximum(
                (float(np.prod(ref.shape)) / float(truth_mask.shape[0]))
                * jnp.sum(truth_mask).astype(jnp.float32),
                1.0,
            )

            truth_mask = truth_mask.reshape(*shape_mask)

            ref = ref * truth_mask
            predicted = predicted * truth_mask

            rmse = (jnp.sum((predicted - ref) ** 2) / nel) ** 0.5
            mae = jnp.sum(jnp.abs(predicted - ref)) / nel

            rmses[name] = rmse
            maes[name] = mae
            if return_targets:
                targets[name] = (predicted, ref, truth_mask)

        if return_targets:
            return rmses, maes, output, targets

        return rmses, maes, output

    if jit:
        return jax.jit(validation)
    return validation
