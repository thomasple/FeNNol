import os
import yaml
import sys
import jax
import io
import time
import jax.numpy as jnp
import numpy as np
import optax
from collections import defaultdict
import json
from copy import deepcopy
from pathlib import Path
import argparse
import torch
import random
from flax import traverse_util
import json

from flax.core import freeze, unfreeze
from .io import (
    load_configuration,
    load_dataset,
    load_model,
    TeeLogger,
    copy_parameters,
)
from .utils import (
    get_loss_definition,
    get_train_step_function,
    get_validation_function,
    get_optimizer,
)
from ..utils import deep_update, AtomicUnits as au
from ..models.preprocessing import AtomPadding, check_input, convert_to_jax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--model_file", type=str, default=None)
    args = parser.parse_args()
    config_file = args.config_file
    model_file = args.model_file

    os.environ["OMP_NUM_THREADS"] = "1"
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )

    parameters = load_configuration(config_file)

    ### Set the device
    device: str = parameters.get("device", "cpu")
    if device == "cpu":
        device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda"):
        num = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = num
        device = "gpu"

    # output directory
    output_directory = parameters.get("output_directory", None)
    if output_directory is not None:
        if "{now}" in output_directory:
            output_directory = output_directory.replace(
                "{now}", time.strftime("%Y-%m-%d-%H-%M-%S")
            )
        output_directory = Path(output_directory).absolute()
        if not output_directory.exists():
            output_directory.mkdir(parents=True)
        output_directory = str(output_directory) + "/"
        print("Output directory:", output_directory)
    else:
        output_directory = ""

    # copy config_file to output directory
    # config_name = Path(config_file).name
    config_ext = Path(config_file).suffix
    with open(config_file) as f_in:
        config_data = f_in.read()
    with open(output_directory + "/config" + config_ext, "w") as f_out:
        f_out.write(config_data)

    # set log file
    log_file = parameters.get("log_file", None)
    if log_file is not None:
        logger = TeeLogger(output_directory + log_file)
        logger.bind_stdout()

    # set matmul precision
    matmul_precision = parameters.get("matmul_prec", "highest").lower()
    assert matmul_precision in [
        "default",
        "high",
        "highest",
    ], "matmul_prec must be one of 'default','high','highest'"
    jax.config.update("jax_default_matmul_precision", matmul_precision)

    # set device
    _device = jax.devices(device)[0]
    jax.config.update("jax_default_device", _device)

    # set random seed
    rng_seed = parameters.get("rng_seed", np.random.randint(0, 2**32 - 1))
    print(f"rng_seed: {rng_seed}")
    rng_key = jax.random.PRNGKey(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    try:
        if "stages" in parameters["training"]:
            ## train in stages ##
            params = deepcopy(parameters)
            stages = params["training"].pop("stages")
            assert isinstance(
                stages, dict
            ), "'stages' must be a dict with named stages"
            model_file_stage = model_file
            print_stages_params = params["training"].get(
                "print_stages_params", False
            )
            for i, (stage, stage_params) in enumerate(stages.items()):
                rng_key, subkey = jax.random.split(rng_key)
                print("")
                print(f"### STAGE {i+1}: {stage} ###")

                ## remove end_event from previous stage ##
                if i > 0 and "end_event" in params["training"]:
                    params["training"].pop("end_event")

                ## incrementally update training parameters ##
                params = deep_update(params, {"training": stage_params})
                if model_file_stage is not None:
                    ## load model from previous stage ##
                    params["model_file"] = model_file_stage

                if print_stages_params:
                    print("stage parameters:")
                    print(json.dumps(params, indent=2, sort_keys=False))

                ## train stage ##
                _, model_file_stage = train(
                    subkey, params, stage=i + 1, output_directory=output_directory
                )
        else:
            ## single training stage ##
            train(
                rng_key,
                parameters,
                model_file=model_file,
                output_directory=output_directory,
            )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if log_file is not None:
            logger.unbind_stdout()
            logger.close()


def train(rng_key, parameters, model_file=None, stage=None, output_directory=None):
    if output_directory is None:
        output_directory = "./"
    elif not output_directory.endswith("/"):
        output_directory += "/"
    stage_prefix = f"_stage_{stage}" if stage is not None else ""

    rng_key, model_key = jax.random.split(rng_key)
    model = load_model(parameters, model_file, rng_key=model_key)

    training_parameters = parameters.get("training", {})
    model_ref = None
    if "model_ref" in training_parameters:
        model_ref = load_model(parameters, training_parameters["model_ref"])
        print("Reference model:", training_parameters["model_ref"])

        if "ref_parameters" in training_parameters:
            ref_parameters = training_parameters["ref_parameters"]
            assert isinstance(
                ref_parameters, list
            ), "ref_parameters must be a list of str"
            print("Reference parameters:", ref_parameters)
            model.variables = copy_parameters(
                model.variables, model_ref.variables, ref_parameters
            )

    # rename_refs = training_parameters.get("rename_refs", [])
    loss_definition, used_keys, ref_keys = get_loss_definition(
        training_parameters
    )

    coordinates_ref_key = training_parameters.get("coordinates_ref_key", None)
    if coordinates_ref_key is not None:
        compute_ref_coords = True
        print("Reference coordinates:", coordinates_ref_key)
        ref_keys.append(coordinates_ref_key)
    else:
        compute_ref_coords = False
    
    training_iterator, validation_iterator = load_dataset(
        training_parameters, infinite_iterator=True, atom_padding=True, ref_keys=ref_keys
        ,split_data_inputs=True
    )
    batch_size = training_parameters.get("batch_size", 16)

    compute_forces = "forces" in used_keys

    # get optimizer parameters
    lr = training_parameters.get("lr", 1.0e-3)
    max_epochs = training_parameters.get("max_epochs", 2000)
    nbatch_per_epoch = training_parameters.get("nbatch_per_epoch", 200)
    nbatch_per_validation = training_parameters.get("nbatch_per_validation", 20)
    init_lr = training_parameters.get("init_lr", lr / 25)
    final_lr = training_parameters.get("final_lr", lr / 10000)
    peak_epoch = training_parameters.get("peak_epoch", 0.3 * max_epochs)

    schedule_type = training_parameters.get("schedule_type", "cosine_onecycle").lower()
    schedule_type = training_parameters.get("scheduler", schedule_type).lower()
    schedule_metrics = training_parameters.get("schedule_metrics", "rmse_tot")
    print("Schedule type:", schedule_type)
    if schedule_type == "cosine_onecycle":
        schedule_ = optax.cosine_onecycle_schedule(
            peak_value=lr,
            div_factor=lr / init_lr,
            final_div_factor=init_lr / final_lr,
            transition_steps=max_epochs * nbatch_per_epoch,
            pct_start=peak_epoch / max_epochs,
        )
        sch_state = {"count": 0, "best": np.inf, "lr": init_lr}

        def schedule(state, rmse=None):
            new_state = {**state}
            lr = schedule_(state["count"])
            if rmse is None:
                new_state["count"] += 1
            new_state["lr"] = lr
            return lr, new_state

    elif schedule_type == "constant":
        sch_state = {"count": 0}

        def schedule(state, rmse=None):
            new_state = {**state}
            if rmse is None:
                new_state["count"] += 1
            return lr, new_state

    elif schedule_type == "reduce_on_plateau":
        patience = training_parameters.get("patience", 10)
        factor = training_parameters.get("lr_factor", 0.5)
        patience_thr = training_parameters.get("patience_thr", 0.0)
        sch_state = {"count": 0, "best": np.inf, "lr": lr, "patience": patience}

        def schedule(state, rmse=None):
            new_state = {**state}
            if rmse is None:
                new_state["count"] += 1
                return state["lr"], new_state
            if rmse <= state["best"] * (1.0 + patience_thr):
                if rmse < state["best"]:
                    new_state["best"] = rmse
                new_state["patience"] = 0
            else:
                new_state["patience"] += 1
                if new_state["patience"] >= patience:
                    new_state["lr"] = state["lr"] * factor
                    new_state["patience"] = 0
                    print("Reducing learning rate to", new_state["lr"])
            return new_state["lr"], new_state

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    optimizer = get_optimizer(
        training_parameters, model.variables, schedule(sch_state)[0]
    )
    opt_st = optimizer.init(model.variables)

    # exponential moving average of the parameters
    ema_decay = training_parameters.get("ema_decay", -1.0)
    if ema_decay > 0.0:
        assert ema_decay < 1.0, "ema_decay must be in (0,1)"
        ema = optax.ema(decay=ema_decay)
    else:
        ema = optax.identity()
    ema_st = ema.init(model.variables)

    # end event
    end_event = training_parameters.get("end_event", None)
    if end_event is None or isinstance(end_event, str) and end_event.lower() == "none":
        is_end = lambda metrics: False
    else:
        assert len(end_event) == 2, "end_event must be a list of two elements"
        is_end = lambda metrics: metrics[end_event[0]] < end_event[1]

    print_timings = parameters.get("print_timings", False)

    if "energy_terms" in training_parameters:
        model.set_energy_terms(training_parameters["energy_terms"], jit=False)
    print("energy terms:", model.energy_terms)

    if compute_forces:
        print("Computing forces")
        def evaluate(model, variables, data):
            _, _, output = model._energy_and_forces(variables, data)
            return output
    elif model.energy_terms is not None:
        def evaluate(model, variables, data):
            _, output = model._total_energy(variables, data)
            return output
    else:
        def evaluate(model, variables, data):
            output = model.modules.apply(variables, data)
            return output

    train_step = get_train_step_function(
        loss_definition=loss_definition,
        model=model,
        model_ref=model_ref,
        compute_ref_coords=compute_ref_coords,
        evaluate=evaluate,
        optimizer=optimizer,
        ema=ema,
    )

    validation = get_validation_function(
        loss_definition=loss_definition,
        model=model,
        model_ref=model_ref,
        compute_ref_coords=compute_ref_coords,
        evaluate=evaluate,
        return_targets=False,
    )

    

    keep_all_bests = training_parameters.get("keep_all_bests", False)
    previous_best_name = None
    best_metric = np.inf
    metric_use_best = training_parameters.get("metric_best", "rmse").lower()
    assert metric_use_best in ["mae", "rmse"], "metric_best must be 'mae' or 'rmse'"

    ## configure preprocessing ##
    pbc_training = training_parameters.get("pbc_training", False)
    minimum_image = training_parameters.get("minimum_image", False)
    preproc_state = unfreeze(model.preproc_state)
    layer_state = []
    for st in preproc_state["layers_state"]:
        stnew = unfreeze(st)
        #     st["nblist_skin"] = nblist_skin
        #     if nblist_stride > 1:
        #         st["skin_stride"] = nblist_stride
        #         st["skin_count"] = nblist_stride
        if pbc_training:
            stnew["minimum_image"] = minimum_image
        if "nblist_mult_size" in training_parameters:
            stnew["nblist_mult_size"] = training_parameters["nblist_mult_size"]
        if "nblist_add_neigh" in training_parameters:
            stnew["add_neigh"] = training_parameters["nblist_add_neigh"]
        if "nblist_add_atoms" in training_parameters:
            stnew["add_atoms"] = training_parameters["nblist_add_atoms"]
        layer_state.append(freeze(stnew))
    
    preproc_state["layers_state"] = layer_state
    model.preproc_state = freeze(preproc_state)

    # inputs,data = next(training_iterator)
    # inputs = model.preprocess(**inputs)
    # print("preproc_state:",model.preproc_state)


    ### Training loop ###
    start = time.time()

    fetch_time = 0.0
    preprocess_time = 0.0
    step_time = 0.0

    maes_prev = defaultdict(lambda: np.inf)
    count = 0
    restore_count = 0
    max_restore_count = training_parameters.get("max_restore_count", 5)
    variables = deepcopy(model.variables)
    variables_save = deepcopy(variables)
    variables_ema_save = deepcopy(model.variables)

    fmetrics = open(output_directory + f"metrics{stage_prefix}.traj", "w")

    print("Starting training...")
    for epoch in range(max_epochs):
        for _ in range(nbatch_per_epoch):
            # fetch data
            s = time.time()
            inputs0,data = next(training_iterator)
            e = time.time()
            fetch_time += e - s

            # preprocess data
            s = time.time()
            inputs = model.preprocess(**inputs0)

            rng_key, subkey = jax.random.split(rng_key)
            inputs["rng_key"] = subkey
            if compute_ref_coords:
                inputs_ref = {**inputs0, "coordinates": data[coordinates_ref_key]}
                inputs_ref = model.preprocess(**inputs_ref)
            else:
                inputs_ref = None
            # if print_timings:
            #     jax.block_until_ready(inputs["coordinates"])
            e = time.time()
            preprocess_time += e - s

            # train step
            s = time.time()
            # opt_st.inner_states["trainable"].inner_state[1].hyperparams[
            #     "learning_rate"
            # ] = schedule(count)
            current_lr, sch_state = schedule(sch_state)
            opt_st.inner_states["trainable"].inner_state[-1].hyperparams[
                "step_size"
            ] = current_lr
            loss, variables, opt_st, model.variables, ema_st, output = train_step(
                data=data,
                inputs=inputs,
                variables=variables,
                variables_ema=model.variables,
                opt_st=opt_st,
                ema_st=ema_st,
                inputs_ref=inputs_ref,
            )
            count += 1
            # if print_timings:
            #     jax.block_until_ready(loss)
            e = time.time()
            step_time += e - s

        rmses_avg = defaultdict(lambda: 0.0)
        maes_avg = defaultdict(lambda: 0.0)
        for _ in range(nbatch_per_validation):
            inputs0,data = next(validation_iterator)

            inputs = model.preprocess(**inputs0)
            
            if compute_ref_coords:
                inputs_ref = {**inputs0, "coordinates": data[coordinates_ref_key]}
                inputs_ref = model.preprocess(**inputs_ref)
            else:
                inputs_ref = None
            rmses, maes, output_val = validation(
                data=data,inputs=inputs, variables=model.variables,inputs_ref=inputs_ref
            )
            for k, v in rmses.items():
                rmses_avg[k] += v
            for k, v in maes.items():
                maes_avg[k] += v
        for k in rmses_avg.keys():
            rmses_avg[k] /= nbatch_per_validation
        for k in maes_avg.keys():
            maes_avg[k] /= nbatch_per_validation

        step_time /= nbatch_per_epoch
        fetch_time /= nbatch_per_epoch
        preprocess_time /= nbatch_per_epoch

        print("")
        print(f"Epoch {epoch+1}, lr={current_lr:.3e}, loss = {loss:.3e}")
        rmse_tot = 0.0
        mae_tot = 0.0
        for k in rmses_avg.keys():
            mult = loss_definition[k]["mult"]
            rmse_tot = (
                rmse_tot + rmses_avg[k] * loss_definition[k]["weight"] ** 0.5
            )
            mae_tot = mae_tot + maes_avg[k] * loss_definition[k]["weight"]
            unit = (
                "(" + loss_definition[k]["unit"] + ")"
                if "unit" in loss_definition[k]
                else ""
            )
            if rmses_avg[k] / mult < 1.0e-2:
                print(
                    f"    rmse_{k}= {rmses_avg[k]/mult:10.3e} ; mae_{k}= {maes_avg[k]/mult:10.3e}   {unit}"
                )
            else:
                print(
                    f"    rmse_{k}= {rmses_avg[k]/mult:10.3f} ; mae_{k}= {maes_avg[k]/mult:10.3f}   {unit}"
                )

        if print_timings:
            print(
                f"    fetch time = {fetch_time:.5f}; preprocess time = {preprocess_time:.5f}; train time = {step_time:.5f}"
            )
        fetch_time = 0.0
        step_time = 0.0
        preprocess_time = 0.0
        restore = False
        reinit = False
        for k, mae in maes_avg.items():
            if np.isnan(mae):
                restore = True
                reinit = True
                print("NaN detected in mae")
                # for k, v in inputs.items():
                #     if hasattr(v,"shape"):
                #         if np.isnan(v).any():
                #             print(k,v)
                #sys.exit(1)
            if "threshold" in loss_definition[k]:
                thr = loss_definition[k]["threshold"]
                if mae > thr * maes_prev[k]:
                    restore = True
                    break

        if restore:
            restore_count += 1
            if restore_count > max_restore_count:
                if reinit:
                    raise ValueError("Model diverged and could not be restored.")
                else:
                    restore_count = 0
                    print(
                        f"{max_restore_count} unsuccessful restores, resuming training."
                    )
                    continue

            variables = deepcopy(variables_save)
            model.variables = deepcopy(variables_ema_save)
            print("Restored previous model after divergence.")
            if reinit:
                opt_st = optimizer.init(model.variables)
                ema_st = ema.init(model.variables)
                print("Reinitialized optimizer after divergence.")
            continue

        restore_count = 0
        variables_save = deepcopy(variables)
        variables_ema_save = deepcopy(model.variables)
        maes_prev = maes_avg

        model.save(output_directory + "latest_model.fnx")

        # save metrics
        metrics = {
            "epoch": epoch + 1,
            "step": count,
            "data_count": count * batch_size,
            "elapsed_time": time.time() - start,
            "lr": current_lr,
            "loss": loss,
        }
        for k in rmses_avg.keys():
            mult = loss_definition[k]["mult"]
            metrics[f"rmse_{k}"] = rmses_avg[k] / mult
            metrics[f"mae_{k}"] = maes_avg[k] / mult

        metrics["rmse_tot"] = rmse_tot
        metrics["mae_tot"] = mae_tot
        metric_for_best = metrics[metric_use_best+"_tot"]
        if metric_for_best < best_metric:
            best_metric = metric_for_best
            metrics["best_metric"] = best_metric
            if keep_all_bests:
                best_name = (
                    output_directory
                    + f"best_model{stage_prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}.fnx"
                )
                model.save(best_name)

            best_name = output_directory + f"best_model{stage_prefix}.fnx"
            model.save(best_name)
            print("New best model saved to:", best_name)

        if epoch == 0:
            headers = [f"{i+1}:{k}" for i, k in enumerate(metrics.keys())]
            fmetrics.write("# " + " ".join(headers) + "\n")
        fmetrics.write(" ".join([str(metrics[k]) for k in metrics.keys()]) + "\n")
        fmetrics.flush()

        # update learning rate using current metrics
        assert (
            schedule_metrics in metrics
        ), f"Error: cannot update lr, {schedule_metrics} not in metrics"
        current_lr, sch_state = schedule(sch_state, metrics[schedule_metrics])

        if is_end(metrics):
            print("Stage finished.")
            break

    end = time.time()
    print(f"Training time: {end-start} s")
    print("")

    fmetrics.close()

    filename = output_directory + f"final_model{stage_prefix}.fnx"
    model.save(filename)
    print("Final model saved to:", filename)

    return metrics, filename


if __name__ == "__main__":
    main()
