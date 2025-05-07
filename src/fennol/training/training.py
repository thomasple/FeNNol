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
try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for training models. Install the CPU version following instructions at https://pytorch.org/get-started/locally/"
    )
import random
from flax import traverse_util
import json
import shutil
import pickle

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
    linear_schedule,
)
from .optimizers import get_optimizer, get_lr_schedule
from ..utils import deep_update, AtomicUnits as au
from ..utils.io import human_time_duration
from ..models.preprocessing import AtomPadding, check_input, convert_to_jax

def save_restart_checkpoint(output_directory,stage,training_state,preproc_state,metrics_state):
    with open(output_directory + "/restart_checkpoint.pkl", "wb") as f:
            pickle.dump({
                "stage": stage,
                "training": training_state,
                "preproc_state": preproc_state,
                "metrics_state": metrics_state,
            }, f)

def load_restart_checkpoint(output_directory):
    restart_checkpoint_file = output_directory + "/restart_checkpoint.pkl"
    if not os.path.exists(restart_checkpoint_file):
        raise FileNotFoundError(
            f"Training state file not found: {restart_checkpoint_file}"
        )
    with open(restart_checkpoint_file, "rb") as f:
        restart_checkpoint = pickle.load(f)
    return restart_checkpoint


def main():
    parser = argparse.ArgumentParser(prog="fennol_train")
    parser.add_argument("config_file", type=str)
    parser.add_argument("--model_file", type=str, default=None)
    args = parser.parse_args()
    config_file = args.config_file
    model_file = args.model_file

    os.environ["OMP_NUM_THREADS"] = "1"
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), "wb", 0), write_through=True
    )

    restart_training = False
    if os.path.isdir(config_file):
        output_directory = Path(config_file).absolute().as_posix()
        restart_training = True
        while output_directory.endswith("/"):
            output_directory = output_directory[:-1]
        config_file = output_directory + "/config.yaml"
        backup_dir = output_directory + f"_backup_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        shutil.copytree(output_directory, backup_dir)
        print("Restarting training from", output_directory)

    parameters = load_configuration(config_file)

    ### Set the device
    device: str = parameters.get("device", "cpu").lower()
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda") or device.startswith("gpu"):
        if ":" in device:
            num = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = num
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = "gpu"

    _device = jax.devices(device)[0]
    jax.config.update("jax_default_device", _device)

    if restart_training:
        restart_checkpoint = load_restart_checkpoint(output_directory)
    else:
        restart_checkpoint = None

    # output directory
    if not restart_training:
        output_directory = parameters.get("output_directory", None)
        if output_directory is not None:
            if "{now}" in output_directory:
                output_directory = output_directory.replace(
                    "{now}", time.strftime("%Y-%m-%d-%H-%M-%S")
                )
            output_directory = Path(output_directory).absolute()
            if not output_directory.exists():
                output_directory.mkdir(parents=True)
            print("Output directory:", output_directory)
        else:
            output_directory = "."

    output_directory = str(output_directory) + "/"

    # copy config_file to output directory
    # config_name = Path(config_file).name
    config_ext = Path(config_file).suffix
    with open(config_file) as f_in:
        config_data = f_in.read()
    with open(output_directory + "/config" + config_ext, "w") as f_out:
        f_out.write(config_data)

    # set log file
    log_file = "train.log"  # parameters.get("log_file", None)
    logger = TeeLogger(output_directory + log_file)
    logger.bind_stdout()

    # set matmul precision
    enable_x64 = parameters.get("double_precision", False)
    jax.config.update("jax_enable_x64", enable_x64)
    fprec = "float64" if enable_x64 else "float32"
    parameters["fprec"] = fprec
    if enable_x64:
        print("Double precision enabled.")

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

                if restart_training and restart_checkpoint["stage"] != i + 1:
                    print(f"Skipping stage {i+1} (already completed)")
                    continue

                if print_stages_params:
                    print("stage parameters:")
                    print(json.dumps(params, indent=2, sort_keys=False))

                ## train stage ##
                _, model_file_stage = train(
                    (subkey, np_rng),
                    params,
                    stage=i + 1,
                    output_directory=output_directory,
                    restart_checkpoint=restart_checkpoint,
                )

                restart_training = False
                restart_checkpoint = None
        else:
            ## single training stage ##
            train(
                (rng_key, np_rng),
                parameters,
                model_file=model_file,
                output_directory=output_directory,
                restart_checkpoint=restart_checkpoint,
            )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if log_file is not None:
            logger.unbind_stdout()
            logger.close()


def train(
    rng,
    parameters,
    model_file=None,
    stage=None,
    output_directory=None,
    restart_checkpoint=None,
):
    if output_directory is None:
        output_directory = "./"
    elif output_directory == "":
        output_directory = "./"
    elif not output_directory.endswith("/"):
        output_directory += "/"
    stage_prefix = f"_stage_{stage}" if stage is not None else ""

    if isinstance(rng, tuple):
        rng_key, np_rng = rng
    else:
        rng_key = rng
        np_rng = np.random.Generator(np.random.PCG64(np.random.randint(0, 2**32 - 1)))

    ### GET TRAINING PARAMETERS ###
    training_parameters = parameters.get("training", {})

    #### LOAD MODEL ####
    if restart_checkpoint is not None:
        model_key = None
        model_file = output_directory + "latest_model.fnx"
    else:
        rng_key, model_key = jax.random.split(rng_key)
    model = load_model(parameters, model_file, rng_key=model_key)

    ### SAVE INITIAL MODEL ###
    save_initial_model = (
        restart_checkpoint is None 
        and (stage is None or stage == 0)
    )
    if save_initial_model:
        model.save(output_directory + "initial_model.fnx")

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

    ### SET FLOATING POINT PRECISION ###
    fprec = parameters.get("fprec", "float32")

    def convert_to_fprec(x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(fprec)
        return x

    model.variables = jax.tree_map(convert_to_fprec, model.variables)

    ### SET UP LOSS FUNCTION ###
    loss_definition, used_keys, ref_keys = get_loss_definition(
        training_parameters, model_energy_unit=model.energy_unit
    )

    coordinates_ref_key = training_parameters.get("coordinates_ref_key", None)
    if coordinates_ref_key is not None:
        compute_ref_coords = True
        print("Reference coordinates:", coordinates_ref_key)
    else:
        compute_ref_coords = False

    #### LOAD DATASET ####
    dspath = training_parameters.get("dspath", None)
    if dspath is None:
        raise ValueError("Dataset path 'training/dspath' should be specified.")
    batch_size = training_parameters.get("batch_size", 16)
    batch_size_val = training_parameters.get("batch_size_val", None)
    rename_refs = training_parameters.get("rename_refs", {})
    training_iterator, validation_iterator = load_dataset(
        dspath=dspath,
        batch_size=batch_size,
        batch_size_val=batch_size_val,
        training_parameters=training_parameters,
        infinite_iterator=True,
        atom_padding=True,
        ref_keys=ref_keys,
        split_data_inputs=True,
        np_rng=np_rng,
        add_flags=["training"],
        fprec=fprec,
        rename_refs=rename_refs,
    )

    compute_forces = "forces" in used_keys
    compute_virial = "virial_tensor" in used_keys or "virial" in used_keys
    compute_stress = "stress_tensor" in used_keys or "stress" in used_keys
    compute_pressure = "pressure" in used_keys or "pressure_tensor" in used_keys

    ### get optimizer parameters ###
    max_epochs = int(training_parameters.get("max_epochs", 2000))
    epoch_format = len(str(max_epochs))
    nbatch_per_epoch = training_parameters.get("nbatch_per_epoch", 200)
    nbatch_per_validation = training_parameters.get("nbatch_per_validation", 20)

    schedule, sch_state, schedule_metrics,adaptive_scheduler = get_lr_schedule(max_epochs,nbatch_per_epoch,training_parameters)
    optimizer = get_optimizer(
        training_parameters, model.variables, schedule(sch_state)[0]
    )
    opt_st = optimizer.init(model.variables)

    ### exponential moving average of the parameters ###
    ema_decay = training_parameters.get("ema_decay", -1.0)
    if ema_decay > 0.0:
        assert ema_decay < 1.0, "ema_decay must be in (0,1)"
        ema = optax.ema(decay=ema_decay)
    else:
        ema = optax.identity()
    ema_st = ema.init(model.variables)

    #### end event ####
    end_event = training_parameters.get("end_event", None)
    if end_event is None or isinstance(end_event, str) and end_event.lower() == "none":
        is_end = lambda metrics: False
    else:
        assert len(end_event) == 2, "end_event must be a list of two elements"
        is_end = lambda metrics: metrics[end_event[0]] < end_event[1]

    if "energy_terms" in training_parameters:
        model.set_energy_terms(training_parameters["energy_terms"], jit=False)
    print("energy terms:", model.energy_terms)

    ### MODEL EVALUATION FUNCTION ###
    pbc_training = training_parameters.get("pbc_training", False)
    if compute_stress or compute_virial or compute_pressure:
        virial_key = "virial" if "virial" in used_keys else "virial_tensor"
        stress_key = "stress" if "stress" in used_keys else "stress_tensor"
        pressure_key = "pressure" if "pressure" in used_keys else "pressure_tensor"
        if compute_stress or compute_pressure:
            assert pbc_training, "PBC must be enabled for stress or virial training"
            print("Computing forces and stress tensor")
            def evaluate(model, variables, data):
                _, _, vir, output = model._energy_and_forces_and_virial(variables, data)
                cells = output["cells"]
                volume = jnp.abs(jnp.linalg.det(cells))
                stress = vir / volume[:, None, None]
                output[stress_key] = stress
                output[virial_key] = vir
                if pressure_key == "pressure":
                    output[pressure_key] = -jnp.trace(stress, axis1=1, axis2=2) / 3.0
                else:
                    output[pressure_key] = -stress
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

    ### TRAINING FUNCTIONS ###
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

    #### CONFIGURE PREPROCESSING ####
    gpu_preprocessing = training_parameters.get("gpu_preprocessing", False)
    if gpu_preprocessing:
        print("GPU preprocessing activated.")

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
    preproc_state["check_input"] = False
    model.preproc_state = freeze(preproc_state)

    ### INITIALIZE METRICS ###
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
    max_restore_count = training_parameters.get("max_restore_count", 5)    

    fmetrics = open(
        output_directory + f"metrics{stage_prefix}.traj",
        "w" if restart_checkpoint is None else "a",
    )

    keep_all_bests = training_parameters.get("keep_all_bests", False)
    save_model_at_epochs = training_parameters.get("save_model_at_epochs", [])
    save_model_at_epochs= set([int(x) for x in save_model_at_epochs])
    save_model_every = int(training_parameters.get("save_model_every_epoch", 0))
    if save_model_every > 0:
        save_model_every = list(range(save_model_every, max_epochs+save_model_every, save_model_every))
        save_model_at_epochs = save_model_at_epochs.union(save_model_every)
    save_model_at_epochs = list(save_model_at_epochs)
    save_model_at_epochs.sort()

    ### LR factor after restoring a model that has diverged
    restore_scaling = training_parameters.get("restore_scaling", 1)
    assert restore_scaling > 0.0, "restore_scaling must be > 0.0"
    assert restore_scaling <= 1.0, "restore_scaling must be <= 1.0"
    if restore_scaling != 1:
        print("Applying LR scaling after restore:", restore_scaling)


    metric_use_best = training_parameters.get("metric_best", "rmse_tot")  # .lower()
    metrics_state = {}

    ### INITIALIZE TRAINING STATE ###
    if restart_checkpoint is not None:
        training_state = deepcopy(restart_checkpoint["training"])
        epoch_start = training_state["epoch"]
        model.preproc_state = freeze(restart_checkpoint["preproc_state"])
        if smoothen_metrics:
            metrics_state = deepcopy(restart_checkpoint["metrics_state"])
            rmses_smooth = metrics_state["rmses_smooth"]
            maes_smooth = metrics_state["maes_smooth"]
            rmse_tot_smooth = metrics_state["rmse_tot_smooth"]
            mae_tot_smooth = metrics_state["mae_tot_smooth"]
            nsmooth = metrics_state["nsmooth"]
        print("Restored training state")
    else:
        epoch_start = 1
        training_state = {
            "rng_key": rng_key,
            "opt_state": opt_st,
            "ema_state": ema_st,
            "sch_state": sch_state,
            "variables": model.variables,
            "variables_ema": model.variables,
            "step": 0,
            "restore_count": 0,
            "restore_scale": 1,
            "epoch": 0,
            "best_metric": np.inf,
        }
        if smoothen_metrics:
            metrics_state = {
                "rmses_smooth": dict(rmses_smooth),
                "maes_smooth": dict(maes_smooth),
                "rmse_tot_smooth": rmse_tot_smooth,
                "mae_tot_smooth": mae_tot_smooth,
                "nsmooth": nsmooth,
            }
    
    del opt_st, ema_st, sch_state, preproc_state
    variables_save = training_state['variables']
    variables_ema_save = training_state['variables_ema']

    ### Training loop ###
    start = time.time()
    print("Starting training...")
    for epoch in range(epoch_start, max_epochs+1):
        training_state["epoch"] = epoch
        s = time.time()
        for _ in range(nbatch_per_epoch):
            # fetch data
            inputs0, data = next(training_iterator)

            # preprocess data
            inputs = model.preprocess(verbose=True,use_gpu=gpu_preprocessing,**inputs0)

            rng_key, subkey = jax.random.split(rng_key)
            inputs["rng_key"] = subkey
            inputs["training_epoch"] = epoch

            # adjust learning rate
            current_lr, training_state["sch_state"] = schedule(training_state["sch_state"])
            current_lr *= training_state["restore_scale"]
            training_state["opt_state"].inner_states["trainable"].inner_state[-1].hyperparams[
                "step_size"
            ] = current_lr 

            # train step
            loss,training_state, _ = train_step(data,inputs,training_state)

        rmses_avg = defaultdict(lambda: 0.0)
        maes_avg = defaultdict(lambda: 0.0)
        for _ in range(nbatch_per_validation):
            inputs0, data = next(validation_iterator)

            inputs = model.preprocess(verbose=True,use_gpu=gpu_preprocessing,**inputs0)

            rng_key, subkey = jax.random.split(rng_key)
            inputs["rng_key"] = subkey
            inputs["training_epoch"] = epoch

            rmses, maes, output_val = validation(
                data=data,
                inputs=inputs,
                variables=training_state["variables_ema"],
            )
            for k, v in rmses.items():
                rmses_avg[k] += v
            for k, v in maes.items():
                maes_avg[k] += v

        jax.block_until_ready(output_val)

        ### Timings ###
        e = time.time()
        epoch_time = e - s

        elapsed_time = e - start
        if not adaptive_scheduler:
            remain_glob = elapsed_time * (max_epochs - epoch + 1) / epoch
            remain_last = epoch_time * (max_epochs - epoch + 1)
            # estimate remaining time via weighted average (put more weight on last at the beginning)
            wremain = np.sin(0.5 * np.pi * epoch / max_epochs)
            remaining_time = human_time_duration(
                remain_glob * wremain + remain_last * (1 - wremain)
            )
        elapsed_time = human_time_duration(elapsed_time)
        batch_time = human_time_duration(
            epoch_time / (nbatch_per_epoch + nbatch_per_validation)
        )
        epoch_time = human_time_duration(epoch_time)

        for k in rmses_avg.keys():
            rmses_avg[k] /= nbatch_per_validation
        for k in maes_avg.keys():
            maes_avg[k] /= nbatch_per_validation

        ### Print metrics ###
        print("")
        line = f"Epoch {epoch}, lr={current_lr:.3e}, loss = {loss:.3e}"
        line += f", epoch time = {epoch_time}, batch time = {batch_time}"
        line += f", elapsed time = {elapsed_time}"
        if not adaptive_scheduler:
            line += f", est. remaining time = {remaining_time}"
        print(line)
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

        ### CHECK IF WE SHOULD RESTORE PREVIOUS MODEL ###
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

        epoch_time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        model.variables = training_state["variables_ema"]
        if epoch in save_model_at_epochs:
            filename = output_directory + f"model{stage_prefix}_epoch{epoch:0{epoch_format}}_{epoch_time_str}.fnx"
            print("Scheduled model save :", filename)
            model.save(filename)

        if restore:
            training_state["restore_count"] += 1
            if training_state["restore_count"]  > max_restore_count:
                if reinit:
                    raise ValueError("Model diverged and could not be restored.")
                else:
                    training_state["restore_count"]  = 0
                    print(
                        f"{max_restore_count} unsuccessful restores, resuming training."
                    )

            training_state["variables"] = deepcopy(variables_save)
            training_state["variables_ema"] = deepcopy(variables_ema_save)
            model.variables = training_state["variables_ema"]
            training_state["restore_scale"] *= restore_scaling
            print(f"Restored previous model after divergence. Restore scale: {training_state['restore_scale']:.3f}")
            if reinit:
                training_state["opt_state"] = optimizer.init(model.variables)
                training_state["ema_state"] = ema.init(model.variables)
                print("Reinitialized optimizer after divergence.")
            save_restart_checkpoint(output_directory,stage,training_state,model.preproc_state,metrics_state)
            continue

        training_state["restore_count"]  = 0

        variables_save = deepcopy(training_state["variables"])
        variables_ema_save = deepcopy(training_state["variables_ema"])
        maes_prev = maes_avg

        ### SAVE MODEL ###
        model.save(output_directory + "latest_model.fnx")

        # save metrics
        step = int(training_state["step"])
        metrics = {
            "epoch": epoch,
            "step": step,
            "data_count": step * batch_size,
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

            # update metrics state
            metrics_state["rmses_smooth"] = dict(rmses_smooth)
            metrics_state["maes_smooth"] = dict(maes_smooth)
            metrics_state["rmse_tot_smooth"] = rmse_tot_smooth
            metrics_state["mae_tot_smooth"] = mae_tot_smooth
            metrics_state["nsmooth"] = nsmooth

        assert (
            metric_use_best in metrics
        ), f"Error: metric for selectring best model '{metric_use_best}' not in metrics"

        metric_for_best = metrics[metric_use_best]
        if metric_for_best <  training_state["best_metric"]:
            training_state["best_metric"] = metric_for_best
            metrics["best_metric"] = training_state["best_metric"]
            if keep_all_bests:
                best_name = (
                    output_directory
                    + f"best_model{stage_prefix}_epoch{epoch:0{epoch_format}}_{epoch_time_str}.fnx"
                )
                model.save(best_name)

            best_name = output_directory + f"best_model{stage_prefix}.fnx"
            model.save(best_name)
            print("New best model saved to:", best_name)
        else:
            metrics["best_metric"] = training_state["best_metric"]

        if epoch == 1:
            headers = [f"{i+1}:{k}" for i, k in enumerate(metrics.keys())]
            fmetrics.write("# " + " ".join(headers) + "\n")
        fmetrics.write(" ".join([str(metrics[k]) for k in metrics.keys()]) + "\n")
        fmetrics.flush()

        # update learning rate using current metrics
        assert (
            schedule_metrics in metrics
        ), f"Error: cannot update lr, '{schedule_metrics}' not in metrics"
        current_lr, training_state["sch_state"] = schedule(training_state["sch_state"], metrics[schedule_metrics])

        # update and save training state
        save_restart_checkpoint(output_directory,stage,training_state,model.preproc_state,metrics_state)

        
        if is_end(metrics):
            print("Stage finished.")
            break

    end = time.time()

    print(f"Training time: {human_time_duration(end-start)}")
    print("")

    fmetrics.close()

    filename = output_directory + f"final_model{stage_prefix}.fnx"
    model.save(filename)
    print("Final model saved to:", filename)

    return metrics, filename


if __name__ == "__main__":
    main()
