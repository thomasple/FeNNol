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
    linear_schedule,
)
from ..utils import deep_update, AtomicUnits as au
from ..utils.io import human_time_duration
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
    device: str = parameters.get("device", "cpu").lower()
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda") or device.startswith("gpu"):
        dsplit = device.split(":")
        num = 0 if len(dsplit) == 1 else dsplit[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = num
        jax.config.update("jax_default_device", jax.devices("gpu")[0])
    else:
        raise ValueError(f"Unknown device: {device}")
        

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


    # set random seed
    rng_seed = parameters.get("rng_seed", np.random.randint(0, 2**32 - 1))
    print(f"rng_seed: {rng_seed}")
    rng_key = jax.random.PRNGKey(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    np_rng = np.random.Generator(np.random.PCG64(rng_seed))

    try:
        if "stages" in parameters["training"]:
            ## train in stages ##
            params = deepcopy(parameters)
            stages = params["training"].pop("stages")
            assert isinstance(stages, dict), "'stages' must be a dict with named stages"
            model_file_stage = model_file
            print_stages_params = params["training"].get("print_stages_params", False)
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
                    (subkey, np_rng),
                    params,
                    stage=i + 1,
                    output_directory=output_directory,
                )
        else:
            ## single training stage ##
            train(
                (rng_key, np_rng),
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


def train(rng, parameters, model_file=None, stage=None, output_directory=None):
    if output_directory is None:
        output_directory = "./"
    elif not output_directory.endswith("/"):
        output_directory += "/"
    stage_prefix = f"_stage_{stage}" if stage is not None else ""

    if isinstance(rng, tuple):
        rng_key, np_rng = rng
    else:
        rng_key = rng
        np_rng = np.random.Generator(np.random.PCG64(np.random.randint(0, 2**32 - 1)))
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
        training_parameters, model_energy_unit=model.energy_unit
    )

    coordinates_ref_key = training_parameters.get("coordinates_ref_key", None)
    if coordinates_ref_key is not None:
        compute_ref_coords = True
        print("Reference coordinates:", coordinates_ref_key)
    else:
        compute_ref_coords = False

    dspath = training_parameters.get("dspath", None)
    if dspath is None:
        raise ValueError("Dataset path 'training/dspath' should be specified.")
    batch_size = training_parameters.get("batch_size", 16)
    training_iterator, validation_iterator = load_dataset(
        dspath=dspath,
        batch_size=batch_size,
        training_parameters=training_parameters,
        infinite_iterator=True,
        atom_padding=True,
        ref_keys=ref_keys,
        split_data_inputs=True,
        np_rng=np_rng,
        add_flags=["training"],
    )

    compute_forces = "forces" in used_keys
    compute_virial = "virial_tensor" in used_keys or "virial" in used_keys
    compute_stress = "stress_tensor" in used_keys or "stress" in used_keys

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
            new_state["lr"] = lr
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

    stochastic_scheduler = training_parameters.get("stochastic_scheduler", False)
    if stochastic_scheduler:
        schedule_ = schedule
        rng_key, scheduler_key = jax.random.split(rng_key)
        sch_state["rng_key"] = scheduler_key
        sch_state["lr_max"] = lr
        sch_state["lr_min"] = final_lr

        def schedule(state, rmse=None):
            new_state = {**state, "lr": state["lr_max"]}
            if rmse is None:
                lr_max, new_state = schedule_(new_state, rmse=rmse)
                lr_min = new_state["lr_min"]
                new_state["rng_key"], subkey = jax.random.split(new_state["rng_key"])
                lr = lr_min + (lr_max - lr_min) * jax.random.uniform(subkey)
                new_state["lr"] = lr
                new_state["lr_max"] = lr_max

            return new_state["lr"], new_state

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

    pbc_training = training_parameters.get("pbc_training", False)
    if compute_stress or compute_virial:
        virial_key = "virial" if "virial" in used_keys else "virial_tensor"
        stress_key = "stress" if "stress" in used_keys else "stress_tensor"
        if compute_stress:
            assert pbc_training, "PBC must be enabled for stress or virial training"
            print("Computing forces and stress tensor")

            def evaluate(model, variables, data):
                _, _, vir, output = model._energy_and_forces_and_virial(variables, data)
                cells = output["cells"]
                volume = jnp.abs(jnp.linalg.det(cells))
                stress = -vir / volume[:, None, None]
                output[stress_key] = stress
                output[virial_key] = vir
                return output

        else:
            print("Computing forces and virial tensor")

            def evaluate(model, variables, data):
                _, _, vir, output = model._energy_and_forces_and_virial(variables, data)
                output[virial_key] = vir
                return output

    elif compute_forces:
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

    ## configure preprocessing ##
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

    preproc_state["layers_state"] = tuple(layer_state)
    model.preproc_state = freeze(preproc_state)

    # inputs,data = next(training_iterator)
    # inputs = model.preprocess(**inputs)
    # print("preproc_state:",model.preproc_state)

    if training_parameters.get("gpu_preprocessing", False):
        print("GPU preprocessing activated.")
        def preprocessing(model,inputs):
            preproc_state = model.preproc_state
            outputs = model.preprocessing.process(
                preproc_state,
                inputs
            )
            preproc_state, state_up, outputs, overflow = (
                model.preprocessing.check_reallocate(preproc_state, outputs)
            )
            if overflow:
                print("GPU preprocessing: nblist overflow => reallocating nblist")
                print("size updates:", state_up)
            model.preproc_state = preproc_state
            return outputs

    else:
        preprocessing = lambda model,inputs: model.preprocess(**inputs) 

    fetch_time = 0.0
    preprocess_time = 0.0
    step_time = 0.0

    maes_prev = defaultdict(lambda: np.inf)
    metrics_beta = training_parameters.get("metrics_ema_decay", -1.0)
    smoothen_metrics = metrics_beta < 1.0 and metrics_beta > 0.0
    if smoothen_metrics:
        print("Computing smoothed metrics with beta =", metrics_beta)
        rmses_smooth = defaultdict(lambda: 0.0)
        maes_smooth = defaultdict(lambda: 0.0)
        rmse_tot_smooth = 0.0
        mae_tot_smooth = 0.0
        nsmooth = 0
    count = 0
    restore_count = 0
    max_restore_count = training_parameters.get("max_restore_count", 5)
    variables = deepcopy(model.variables)
    variables_save = deepcopy(variables)
    variables_ema_save = deepcopy(model.variables)

    fmetrics = open(output_directory + f"metrics{stage_prefix}.traj", "w")

    keep_all_bests = training_parameters.get("keep_all_bests", False)
    previous_best_name = None
    best_metric = np.inf
    metric_use_best = training_parameters.get("metric_best", "rmse_tot")  # .lower()
    # authorized_metrics = ["mae", "rmse"]
    # if smoothen_metrics:
    #     authorized_metrics+= ["mae_smooth", "rmse_smooth"]
    # assert metric_use_best in authorized_metrics, f"metric_best must be one of {authorized_metrics}"

    ### Training loop ###
    start = time.time()
    print("Starting training...")
    for epoch in range(max_epochs):
        s = time.time()
        for _ in range(nbatch_per_epoch):
            # fetch data
            inputs0, data = next(training_iterator)

            # preprocess data
            inputs = preprocessing(model,inputs0)
            # inputs = model.preprocess(**inputs0)

            rng_key, subkey = jax.random.split(rng_key)
            inputs["rng_key"] = subkey
            # if compute_ref_coords:
            #     inputs_ref = {**inputs0, "coordinates": data[coordinates_ref_key]}
            #     inputs_ref = model.preprocess(**inputs_ref)
            # else:
            #     inputs_ref = None
            # if print_timings:
            #     jax.block_until_ready(inputs["coordinates"])

            # train step
            # opt_st.inner_states["trainable"].inner_state[1].hyperparams[
            #     "learning_rate"
            # ] = schedule(count)
            current_lr, sch_state = schedule(sch_state)
            opt_st.inner_states["trainable"].inner_state[-1].hyperparams[
                "step_size"
            ] = current_lr
            loss, variables, opt_st, model.variables, ema_st, output = train_step(
                epoch=epoch,
                data=data,
                inputs=inputs,
                variables=variables,
                variables_ema=model.variables,
                opt_st=opt_st,
                ema_st=ema_st,
            )
            count += 1

        rmses_avg = defaultdict(lambda: 0.0)
        maes_avg = defaultdict(lambda: 0.0)
        for _ in range(nbatch_per_validation):
            inputs0, data = next(validation_iterator)

            inputs = preprocessing(model,inputs0)
            # inputs = model.preprocess(**inputs0)

            rng_key, subkey = jax.random.split(rng_key)
            inputs["rng_key"] = subkey

            # if compute_ref_coords:
            #     inputs_ref = {**inputs0, "coordinates": data[coordinates_ref_key]}
            #     inputs_ref = model.preprocess(**inputs_ref)
            # else:
            #     inputs_ref = None
            rmses, maes, output_val = validation(
                data=data,
                inputs=inputs,
                variables=model.variables,
                # inputs_ref=inputs_ref,
            )
            for k, v in rmses.items():
                rmses_avg[k] += v
            for k, v in maes.items():
                maes_avg[k] += v
        
        jax.block_until_ready(output_val)
        e = time.time()
        epoch_time = e - s
        batch_time = human_time_duration(epoch_time / (nbatch_per_epoch+nbatch_per_validation))
        epoch_time = human_time_duration(epoch_time)


        for k in rmses_avg.keys():
            rmses_avg[k] /= nbatch_per_validation
        for k in maes_avg.keys():
            maes_avg[k] /= nbatch_per_validation

        step_time /= nbatch_per_epoch
        fetch_time /= nbatch_per_epoch
        preprocess_time /= nbatch_per_epoch

        print("")
        print(f"Epoch {epoch+1}, lr={current_lr:.3e}, loss = {loss:.3e}, epoch time = {epoch_time}, batch time = {batch_time}")
        rmse_tot = 0.0
        mae_tot = 0.0
        for k in rmses_avg.keys():
            mult = loss_definition[k]["mult"]
            loss_prms = loss_definition[k]
            rmse_tot = rmse_tot + rmses_avg[k] * loss_prms["weight"] ** 0.5
            mae_tot = mae_tot + maes_avg[k] * loss_prms["weight"]
            unit = "(" + loss_prms["unit"] + ")" if "unit" in loss_prms else ""

            weight_str = ""
            if "weight_schedule" in loss_prms:
                w = linear_schedule(epoch, *loss_prms["weight_schedule"])
                weight_str = f"(w={w:.3f})"

            if rmses_avg[k] / mult < 1.0e-2:
                print(
                    f"    rmse_{k}= {rmses_avg[k]/mult:10.3e} ; mae_{k}= {maes_avg[k]/mult:10.3e}   {unit}  {weight_str}"
                )
            else:
                print(
                    f"    rmse_{k}= {rmses_avg[k]/mult:10.3f} ; mae_{k}= {maes_avg[k]/mult:10.3f}   {unit}  {weight_str}"
                )

        # if print_timings:
        #     print(
        #         f"    Timings per batch: fetch time = {fetch_time:.5f}; preprocess time = {preprocess_time:.5f}; train time = {step_time:.5f}"
        #     )
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
                # sys.exit(1)
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
        if smoothen_metrics:
            nsmooth += 1
            for k in rmses_avg.keys():
                mult = loss_definition[k]["mult"]
                rmses_smooth[k] = (
                    metrics_beta * rmses_smooth[k] + (1.0 - metrics_beta) * rmses_avg[k]
                )
                maes_smooth[k] = (
                    metrics_beta * maes_smooth[k] + (1.0 - metrics_beta) * maes_avg[k]
                )
                metrics[f"rmse_smooth_{k}"] = (
                    rmses_smooth[k] / (1.0 - metrics_beta**nsmooth) / mult
                )
                metrics[f"mae_smooth_{k}"] = (
                    maes_smooth[k] / (1.0 - metrics_beta**nsmooth) / mult
                )
            rmse_tot_smooth = (
                metrics_beta * rmse_tot_smooth + (1.0 - metrics_beta) * rmse_tot
            )
            mae_tot_smooth = (
                metrics_beta * mae_tot_smooth + (1.0 - metrics_beta) * mae_tot
            )
            metrics["rmse_smooth_tot"] = rmse_tot_smooth / (1.0 - metrics_beta**nsmooth)
            metrics["mae_smooth_tot"] = mae_tot_smooth / (1.0 - metrics_beta**nsmooth)

        assert (
            metric_use_best in metrics
        ), f"Error: metric for selectring best model '{metric_use_best}' not in metrics"
        metric_for_best = metrics[metric_use_best]
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
        ), f"Error: cannot update lr, '{schedule_metrics}' not in metrics"
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
