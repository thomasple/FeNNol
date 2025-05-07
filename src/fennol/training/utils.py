import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Dict, List, Tuple, Union, Any, NamedTuple
import optax
from copy import deepcopy
from functools import partial

from ..utils import deep_update, AtomicUnits as au
from ..models import FENNIX
        

@partial(jax.jit, static_argnums=(1,2,3,4))
def linear_schedule(step, start_value,end_value, start_step, duration):
    return start_value + jnp.clip((step - start_step) / duration, 0.0, 1.0) * (end_value - start_value)


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
    training_parameters: Dict[str, any],
    model_energy_unit: str = "Ha",  # , manual_renames: List[str] = []
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
    # loss_definition = deepcopy(training_parameters["loss"])
    used_keys = []
    ref_keys = []
    energy_mult = au.get_multiplier(model_energy_unit)
    loss_definition = {}
    for k in training_parameters["loss"]:
        # loss_prms = loss_definition[k]
        loss_prms = deepcopy(training_parameters["loss"][k])
        if "energy_unit" in loss_prms:
            loss_prms["mult"] = energy_mult / au.get_multiplier(
                loss_prms["energy_unit"]
            )
            if "unit" in loss_prms:
                print(
                    "Warning: Both 'unit' and 'energy_unit' are defined for loss component",
                    k,
                    " -> using 'energy_unit'",
                )
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

        if "weight_start" in loss_prms:
            weight_start = loss_prms["weight_start"]
            if "weight_ramp" in loss_prms:
                weight_ramp = loss_prms["weight_ramp"]
            else:
                weight_ramp = training_parameters.get("max_epochs")
            weight_ramp_start = loss_prms.get("weight_ramp_start", 0.0)
            weight_end = loss_prms["weight"]
            print(f"Weight ramp for {k} : {weight_start} -> {weight_end} in {weight_ramp} epochs")
            loss_prms["weight_schedule"] = (weight_start,weight_end,weight_ramp_start,weight_ramp)

        used_keys.append(loss_prms["key"])
        loss_definition[k] = loss_prms

    # rename_refs = list(
    #     set(["forces", "total_energy", "atomic_energies"] + manual_renames + used_keys)
    # )

    # for k in loss_definition.keys():
    #     loss_prms = loss_definition[k]
    #     if "ref" in loss_prms:
    #         if loss_prms["ref"] in rename_refs:
    #             loss_prms["ref"] = "true_" + loss_prms["ref"]

    return loss_definition, list(set(used_keys)), list(set(ref_keys))


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
        training_state,
    ):
        epoch = training_state["epoch"]

        def loss_fn(variables):
            if model_ref is not None:
                output_ref = evaluate(model_ref, model_ref.variables, inputs)
                # _, _, output_ref = model_ref._energy_and_forces(model_ref.variables, data)
            output = evaluate(model, variables, inputs)
            # _, _, output = model._energy_and_forces(variables, data)
            natoms = jnp.where(inputs["true_sys"], inputs["natoms"], 1)
            nsys = inputs["natoms"].shape[0]
            system_mask = inputs["true_sys"]
            atom_mask = inputs["true_atoms"]
            batch_index = inputs["batch_index"]
            if "system_sign" in inputs:
                system_sign = inputs["system_sign"] > 0
                system_mask = jnp.logical_and(system_mask,system_sign)
                atom_mask = jnp.logical_and(atom_mask, system_sign[batch_index])

            loss_tot = 0.0
            for loss_prms in loss_definition.values():
                use_ref_mask = False
                predicted = output[loss_prms["key"]]
                

                if "ref" in loss_prms:
                    if loss_prms["ref"].startswith("model_ref/"):
                        assert model_ref is not None, "model_ref must be provided"
                        try:
                            ref = output_ref[loss_prms["ref"][10:]] * loss_prms["mult"]
                        except KeyError:
                            raise KeyError(
                                f"Reference key '{loss_prms['ref'][10:]}' not found in model_ref output. Keys available: {output_ref.keys()}"
                            )
                        mask_key  = loss_prms["ref"][6:] + "_mask" 
                        if mask_key in data:
                            use_ref_mask = True
                            ref_mask = data[mask_key]
                        elif mask_key in output_ref:
                            use_ref_mask = True
                            ref_mask = output[mask_key]
                        elif mask_key in output:
                            use_ref_mask = True
                            ref_mask = output[mask_key]
                    elif loss_prms["ref"].startswith("model/"):
                        try:
                            ref = output[loss_prms["ref"][6:]] * loss_prms["mult"]
                        except KeyError:
                            raise KeyError(
                                f"Reference key '{loss_prms['ref'][6:]}' not found in model output. Keys available: {output.keys()}"
                            )
                        mask_key  = loss_prms["ref"][6:] + "_mask" 
                        if mask_key in data:
                            use_ref_mask = True
                            ref_mask = data[mask_key]
                        elif mask_key in output:
                            use_ref_mask = True
                            ref_mask = output[mask_key]
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

                if "norm_axis" in loss_prms and loss_prms["norm_axis"] is not None:
                    norm_axis = loss_prms["norm_axis"]
                    predicted = jnp.linalg.norm(
                        predicted, axis=norm_axis, keepdims=True
                    )
                    ref = jnp.linalg.norm(ref, axis=norm_axis, keepdims=True)

                ensemble_loss = loss_prms["type"] in ["ensemble_nll","ensemble_crps","gaussian_mixture",]
                if ensemble_loss:
                    ensemble_axis = loss_prms.get("ensemble_axis", -1)
                    ensemble_weights_key = loss_prms.get("ensemble_weights", None)
                    ensemble_size = predicted.shape[ensemble_axis]
                    if ensemble_weights_key is not None:
                        ensemble_weights = output[ensemble_weights_key]
                    else:
                        ensemble_weights = jnp.ones_like(predicted)
                    if "ensemble_subsample" in loss_prms and "rng_key" in output:
                        ns = min(
                            loss_prms["ensemble_subsample"],
                            predicted.shape[ensemble_axis],
                        )
                        key, subkey = jax.random.split(output["rng_key"])

                        def get_subsample(x):
                            return jax.lax.slice_in_dim(
                                jax.random.permutation(
                                    subkey, x, axis=ensemble_axis, independent=True
                                ),
                                start_index=0,
                                limit_index=ns,
                                axis=ensemble_axis,
                            )

                        predicted = get_subsample(predicted)
                        ensemble_weights = get_subsample(ensemble_weights)
                        output["rng_key"] = key
                    else:
                        get_subsample = lambda x: x
                    ensemble_weights = ensemble_weights / jnp.sum(
                        ensemble_weights, axis=ensemble_axis, keepdims=True
                    )
                    predicted_ensemble = predicted
                    predicted = (predicted * ensemble_weights).sum(
                        axis=ensemble_axis, keepdims=True
                    )
                    predicted_var = (
                        ensemble_weights * (predicted_ensemble - predicted) ** 2
                    ).sum(axis=ensemble_axis) * (ensemble_size / (ensemble_size - 1.0))
                    predicted = jnp.squeeze(predicted, axis=ensemble_axis)
                
                if "remove_ref_sys" in loss_prms and loss_prms["remove_ref_sys"]:
                    assert compute_ref_coords, "compute_ref_coords must be True"
                    # predicted = predicted - output_data_ref[loss_prms["key"]]
                    # assert predicted.shape[0] == nsys, "remove_ref_sys only works with system-level predictions"
                    shape_mask = [predicted.shape[0]] + [1] * (len(predicted.shape) - 1)
                    if predicted.shape[0] == nsys:
                        predicted_sign_mask = inputs["system_sign"].reshape(*shape_mask) < 0
                        predicted_shift = jax.ops.segment_sum(predicted_sign_mask*predicted,inputs["system_index"],nsys)
                        predicted = predicted - predicted_shift
                    elif predicted.shape[0] == batch_index.shape[0]:
                        predicted_sign_mask = (inputs["system_sign"]<0)[batch_index].reshape(*shape_mask)
                        natshift = jnp.concatenate([jnp.array([0]),jnp.cumsum(natoms)])
                        iat = jnp.arange(batch_index.shape[0]) - natshift[batch_index]
                        iatdest = natshift[inputs["system_index"]][batch_index]+iat
                        predicted_shift = jax.ops.segment_sum(predicted_sign_mask*predicted,iatdest,batch_index.shape[0])
                        predicted = predicted - predicted_shift
                    else:
                        raise ValueError("remove_ref_sys only works with system-level or atom-level predictions")
                    
                    if ensemble_loss:
                        predicted_ensemble = predicted_ensemble - jnp.expand_dims(predicted_shift,ensemble_axis)
                        predicted_var = (
                            ensemble_weights * (predicted_ensemble - jnp.expand_dims(predicted,ensemble_axis)) ** 2
                        ).sum(axis=ensemble_axis) * (ensemble_size / (ensemble_size - 1.0))

                if predicted.ndim > 1 and predicted.shape[-1] == 1:
                    predicted = jnp.squeeze(predicted, axis=-1)

                if ref.ndim > 1 and ref.shape[-1] == 1:
                    ref = jnp.squeeze(ref, axis=-1)

                per_atom = False
                shape_mask = [ref.shape[0]] + [1] * (len(ref.shape) - 1)
                # print(loss_prms["key"],predicted.shape,loss_prms["ref"],ref.shape)
                natscale = 1.0
                if ref.shape[0] == output["batch_index"].shape[0]:
                    ## shape is number of atoms
                    truth_mask = atom_mask
                    if "ds_weight" in loss_prms:
                        weight_key = loss_prms["ds_weight"]
                        natscale=data[weight_key]
                        if natscale.shape[0] == natoms.shape[0]:
                            natscale = natscale[output["batch_index"]]
                        natscale = natscale.reshape(*shape_mask)
                elif ref.shape[0] == natoms.shape[0]:
                    ## shape is number of systems
                    truth_mask = system_mask
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

                natscale = natscale / nel

                loss_type = loss_prms["type"]
                if loss_type == "mse":
                    loss = jnp.sum(natscale * (predicted - ref) ** 2)
                elif loss_type == "log_cosh":
                    loss = jnp.sum(natscale * optax.log_cosh(predicted, ref))
                elif loss_type == "mae":
                    loss = jnp.sum(natscale * jnp.abs(predicted - ref))
                elif loss_type == "rel_mse":
                    eps = loss_prms.get("rel_eps", 1.0e-6)
                    rel_pow = loss_prms.get("rel_pow", 2.0)
                    loss = jnp.sum(
                        natscale
                        * (predicted - ref) ** 2
                        / (eps + jnp.abs(ref) ** rel_pow)
                    )
                elif loss_type == "rmse+mae":
                    loss = (
                        jnp.sum(natscale * (predicted - ref) ** 2)
                    ) ** 0.5 + jnp.sum(natscale * jnp.abs(predicted - ref))
                elif loss_type == "crps":
                    predicted_var_key = loss_prms.get(
                        "var_key", loss_prms["key"] + "_var"
                    )
                    lamba_crps = loss_prms.get("lambda_crps", 1.0)
                    predicted_var = output[predicted_var_key]
                    if per_atom:
                        predicted_var = predicted_var / natoms.reshape(*shape_mask)**2
                    predicted_var = predicted_var * truth_mask + (1.0 - truth_mask)
                    sigma = predicted_var**0.5
                    dy = (ref - predicted) / sigma
                    Phi = 0.5 * (1.0 + jax.scipy.special.erf(dy / 2**0.5))
                    phi = jnp.exp(-0.5 * dy**2) / (2 * jnp.pi) ** 0.5
                    loss = jnp.sum(natscale * sigma * (dy * (2 * Phi - 1.0) + 2 * phi - lamba_crps/jnp.pi**0.5))
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
                        predicted_var = predicted_var / natoms.reshape(*shape_mask)**2
                    lamba_crps = loss_prms.get("lambda_crps", 1.0)
                    predicted_var = predicted_var * truth_mask + (1.0 - truth_mask)
                    sigma = predicted_var**0.5
                    dy = (ref - predicted) / sigma
                    Phi = 0.5 * (1.0 + jax.scipy.special.erf(dy / 2**0.5))
                    phi = jnp.exp(-0.5 * dy**2) / (2 * jnp.pi) ** 0.5
                    loss = jnp.sum(
                        natscale * truth_mask * sigma * (dy * (2 * Phi - 1.0) + 2 * phi - lamba_crps/jnp.pi**0.5)
                    )
                elif loss_type == "gaussian_mixture":
                    gm_w = get_subsample(output[loss_prms["key"] + "_w"])
                    gm_w = gm_w / jnp.sum(gm_w, axis=ensemble_axis, keepdims=True)
                    sigma = get_subsample(output[loss_prms["key"] + "_sigma"])
                    ref = jnp.expand_dims(ref, axis=ensemble_axis)
                    # log_likelihoods = -0.5*jnp.log(2*np.pi*gm_sigma2) - ((ref - predicted_ensemble) ** 2) / (2 * gm_sigma2)
                    # log_w =jnp.log(gm_w + 1.e-10)
                    # negloglikelihood = -jax.scipy.special.logsumexp(log_w + log_likelihoods,axis=ensemble_axis)
                    # loss = jnp.sum(natscale * truth_mask * negloglikelihood)

                    # CRPS loss
                    def compute_crps(mu, sigma):
                        dy = mu / sigma
                        Phi = 0.5 * (1.0 + jax.scipy.special.erf(dy / 2**0.5))
                        phi = jnp.exp(-0.5 * dy**2) / (2 * jnp.pi) ** 0.5
                        return sigma * (dy * (2 * Phi - 1.0) + 2 * phi - 1./jnp.pi**0.5)

                    crps1 = compute_crps(ref - predicted_ensemble, sigma)
                    gm_crps1 = (gm_w * crps1).sum(axis=ensemble_axis)

                    if ensemble_axis < 0:
                        ensemble_axis = predicted_ensemble.ndim + ensemble_axis
                    muij = jnp.expand_dims(
                        predicted_ensemble, axis=ensemble_axis
                    ) - jnp.expand_dims(predicted_ensemble, axis=ensemble_axis + 1)
                    sigma2 = sigma**2
                    sigmaij = (
                        jnp.expand_dims(sigma2, axis=ensemble_axis)
                        + jnp.expand_dims(sigma2, axis=ensemble_axis + 1)
                    ) ** 0.5
                    wij = jnp.expand_dims(gm_w, axis=ensemble_axis) * jnp.expand_dims(
                        gm_w, axis=ensemble_axis + 1
                    )
                    crps2 = compute_crps(muij, sigmaij)
                    gm_crps2 = (wij * crps2).sum(
                        axis=(ensemble_axis, ensemble_axis + 1)
                    )

                    crps = gm_crps1 - 0.5 * gm_crps2
                    loss = jnp.sum(natscale * truth_mask * crps)

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
                elif loss_type == "binary_crossentropy":
                    ref = ref.astype(predicted.dtype)
                    if loss_prms.get("raw_logits",False):
                        loss = jnp.sum(
                            natscale*truth_mask
                            * (
                                -ref * jax.nn.log_sigmoid(predicted)
                                - (1.0 - ref) * jax.nn.log_sigmoid(-predicted)
                            )
                        )
                    else:
                        loss = jnp.sum(
                            natscale*truth_mask
                            * (
                                -ref * jnp.log(predicted + 1.0e-10)
                                - (1.0 - ref) * jnp.log(1.0 - predicted + 1.0e-10)
                            )
                        )
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")

                if "weight_schedule" in loss_prms:
                    w = linear_schedule(epoch, *loss_prms["weight_schedule"])
                else:
                    w = loss_prms["weight"]

                loss_tot = loss_tot + w * loss

            return loss_tot, output

        variables = training_state["variables"]
        new_training_state = {**training_state, "step":training_state["step"]+1}

        (loss, o), grad = jax.value_and_grad(loss_fn, has_aux=True)(variables)
        updates, new_training_state["opt_state"] = optimizer.update(grad, training_state["opt_state"], params=variables)
        variables = optax.apply_updates(variables, updates)
        new_training_state["variables"] = variables

        if ema is not None:
            ema_st = training_state.get("ema_state",None)
            if ema_st is None:
                raise ValueError(
                    "train_step was setup with ema but either variables_ema or ema_st was not provided"
                )
            new_training_state["variables_ema"], new_training_state["ema_state"] = ema.update(variables, ema_st)
    
        return loss, new_training_state, o

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
        rmses = {}
        maes = {}
        if return_targets:
            targets = {}

        natoms = jnp.where(inputs["true_sys"], inputs["natoms"], 1)
        nsys = inputs["natoms"].shape[0]
        system_mask = inputs["true_sys"]
        atom_mask = inputs["true_atoms"]
        batch_index = inputs["batch_index"]
        if "system_sign" in inputs:
            system_sign = inputs["system_sign"] > 0
            system_mask = jnp.logical_and(system_mask,system_sign)
            atom_mask = jnp.logical_and(atom_mask, system_sign[batch_index])

        for name, loss_prms in loss_definition.items():
            do_validation = loss_prms.get("validate", True)
            if not do_validation:
                continue
            predicted = output[loss_prms["key"]]
            use_ref_mask = False
            
            if "remove_ref_sys" in loss_prms and loss_prms["remove_ref_sys"]:
                assert compute_ref_coords, "compute_ref_coords must be True"
                # predicted = predicted - output_data_ref[loss_prms["key"]]
                shape_mask = [predicted.shape[0]] + [1] * (len(predicted.shape) - 1)
                if predicted.shape[0] == nsys:
                    system_sign = inputs["system_sign"].reshape(*shape_mask)
                    predicted = jax.ops.segment_sum(system_sign*predicted,inputs["system_index"],nsys)
                elif predicted.shape[0] == batch_index.shape[0]:
                    atom_sign = inputs["system_sign"][batch_index].reshape(*shape_mask)
                    natshift = jnp.concatenate([jnp.array([0]),jnp.cumsum(natoms)])
                    iat = jnp.arange(batch_index.shape[0]) - natshift[batch_index]
                    iatdest = natshift[inputs["system_index"]][batch_index]+iat
                    predicted = jax.ops.segment_sum(atom_sign*predicted,iatdest,batch_index.shape[0])
                else:
                    raise ValueError("remove_ref_sys only works with system-level or atom-level predictions")

            if "ref" in loss_prms:
                if loss_prms["ref"].startswith("model_ref/"):
                    assert model_ref is not None, "model_ref must be provided"
                    ref = output_ref[loss_prms["ref"][10:]] * loss_prms["mult"]
                    mask_key  = loss_prms["ref"][6:] + "_mask" 
                    if mask_key in data:
                        use_ref_mask = True
                        ref_mask = data[mask_key]
                    elif mask_key in output_ref:
                        use_ref_mask = True
                        ref_mask = output[mask_key]
                    elif mask_key in output:
                        use_ref_mask = True
                        ref_mask = output[mask_key]
                elif loss_prms["ref"].startswith("model/"):
                    ref = output[loss_prms["ref"][6:]] * loss_prms["mult"]
                    mask_key  = loss_prms["ref"][6:] + "_mask" 
                    if mask_key in data:
                        use_ref_mask = True
                        ref_mask = data[mask_key]
                    elif mask_key in output:
                        use_ref_mask = True
                        ref_mask = output[mask_key]
                else:
                    ref = data[loss_prms["ref"]] * loss_prms["mult"]
                    if loss_prms["ref"] + "_mask" in data:
                        use_ref_mask = True
                        ref_mask = data[loss_prms["ref"] + "_mask"]
            else:
                ref = jnp.zeros_like(predicted)
            
            if "norm_axis" in loss_prms and loss_prms["norm_axis"] is not None:
                norm_axis = loss_prms["norm_axis"]
                predicted = jnp.linalg.norm(
                    predicted, axis=norm_axis, keepdims=True
                )
                ref = jnp.linalg.norm(ref, axis=norm_axis, keepdims=True)

            loss_type = loss_prms["type"].lower()
            if loss_type.startswith("ensemble"):
                axis = loss_prms.get("ensemble_axis", -1)
                ensemble_weights_key = loss_prms.get("ensemble_weights", None)
                if ensemble_weights_key is not None:
                    ensemble_weights = output[ensemble_weights_key]
                else:
                    ensemble_weights = jnp.ones_like(predicted)
                ensemble_weights = ensemble_weights / jnp.sum(
                    ensemble_weights, axis=axis, keepdims=True
                )

                # predicted = predicted.mean(axis=axis)
                predicted = (ensemble_weights * predicted).sum(axis=axis)

            if loss_prms["type"] in ["gaussian_mixture"]:
                axis = loss_prms.get("ensemble_axis", -1)
                gm_w = output[loss_prms["key"] + "_w"]
                gm_w = gm_w / jnp.sum(gm_w, axis=axis, keepdims=True)
                predicted = (predicted * gm_w).sum(axis=axis)

            if predicted.ndim > 1 and predicted.shape[-1] == 1:
                predicted = jnp.squeeze(predicted, axis=-1)

            if ref.ndim > 1 and ref.shape[-1] == 1:
                ref = jnp.squeeze(ref, axis=-1)

            shape_mask = [ref.shape[0]] + [1] * (len(predicted.shape) - 1)
            if ref.shape[0] == output["batch_index"].shape[0]:
                ## shape is number of atoms
                truth_mask = atom_mask
            elif ref.shape[0] == natoms.shape[0]:
                ## shape is number of systems
                truth_mask = system_mask
                if loss_prms.get("per_atom_validation", False):
                    ref = ref / natoms.reshape(*shape_mask)
                    predicted = predicted / natoms.reshape(*shape_mask)
            else:
                truth_mask = jnp.ones(ref.shape[0], dtype=bool)

            if use_ref_mask:
                truth_mask = truth_mask * ref_mask.astype(bool)

            nel = jnp.maximum(
                (float(np.prod(ref.shape)) / float(truth_mask.shape[0]))
                * jnp.sum(truth_mask).astype(predicted.dtype),
                1.0,
            )

            truth_mask = truth_mask.reshape(*shape_mask)

            ref = ref * truth_mask
            predicted = predicted * truth_mask

            if loss_type == "binary_crossentropy":
                ref = ref.astype(predicted.dtype)
                if loss_prms.get("raw_logits",False):
                    rmse = jnp.sum(
                            (
                                -ref * jax.nn.log_sigmoid(predicted)
                                - (1.0 - ref) * jax.nn.log_sigmoid(-predicted)
                            )*truth_mask/nel
                        )
                    predicted = truth_mask*jax.nn.sigmoid(predicted)
                else:
                    rmse = jnp.sum(
                            (
                                -ref * jnp.log(predicted + 1.0e-10)
                                - (1.0 - ref) * jnp.log(1.0 - predicted + 1.0e-10)
                            )*truth_mask/nel
                        )
                thr = loss_prms.get("class_threshold", 0.05)
                mae = jnp.sum(jnp.where(jnp.abs(predicted - ref) > thr, 1, 0)/nel)*100
            else:
                rmse = jnp.sum((predicted - ref) ** 2 / nel) ** 0.5
                mae = jnp.sum(jnp.abs(predicted - ref) / nel)

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
