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
    import tomlkit
except ImportError:
    tomlkit = None

from .utils import (
    load_dataset,
    load_model,
    get_optimizer,
    get_loss_definition,
    TeeLogger,
)
from ..utils import deep_update, AtomicUnits as au


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

    ### Set the device
    device: str = parameters.get("device", "cpu")
    if device == "cpu":
        device = "cpu"
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

    # set log file
    log_file = parameters.get("log_file", None)
    if log_file is not None:
        logger = TeeLogger(output_directory+log_file)
        logger.bind_stdout()

    _device = jax.devices(device)[0]

    try:
        with jax.default_device(_device):
            if "stages" in parameters["training"]:
                params = deepcopy(parameters)
                stages = params["training"].pop("stages")
                assert isinstance(stages, dict), "'stages' must be a dict with named stages"
                model_file_stage = model_file
                print_stages_params = params["training"].get("print_stages_params", False)
                for i, (stage, stage_params) in enumerate(stages.items()):
                    print("")
                    print(f"### STAGE {i+1}: {stage} ###")
                    if i > 0 and "end_event" in params["training"]:
                        params["training"].pop("end_event")
                    params = deep_update(params, {"training": stage_params})
                    if model_file_stage is not None:
                        params["model_file"] = model_file_stage
                    if print_stages_params:
                        print("stage parameters:")
                        print(json.dumps(params, indent=2, sort_keys=False))
                    _, model_file_stage = train(params, stage=i + 1,output_directory=output_directory)
            else:
                train(parameters, model_file=model_file,output_directory=output_directory)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        if log_file is not None:
            logger.unbind_stdout()
            logger.close()


def train(parameters, model_file=None, stage=None,output_directory=None):
    if output_directory is None:
        output_directory = ""
    stage_prefix = f"_stage_{stage}" if stage is not None else ""

    model = load_model(parameters, model_file)

    training_parameters = parameters.get("training", {})

    loss_definition, rename_refs = get_loss_definition(training_parameters)
    training_iterator, validation_iterator = load_dataset(
        training_parameters, rename_refs
    )

    # get optimizer parameters
    lr = training_parameters.get("lr", 1.0e-3)
    max_epochs = training_parameters.get("max_epochs", 2000)
    nbatch_per_epoch = training_parameters.get("nbatch_per_epoch", 200)
    nbatch_per_validation = training_parameters.get("nbatch_per_validation", 20)
    schedule = optax.cosine_onecycle_schedule(
        peak_value=lr, transition_steps=max_epochs * nbatch_per_epoch
    )

    optimizer = get_optimizer(training_parameters, model.variables, schedule(0))
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

    @jax.jit
    def train_step(variables, variables_ema, opt_st, ema_st, data):
        
        def loss_fn(variables):
            _, _, output = model._energy_and_forces(variables, data)
            nsys = jnp.sum(data["true_sys"])
            nat = jnp.sum(data["true_atoms"])
            loss_tot = 0.0
            for loss_prms in loss_definition.values():
                predicted = output[loss_prms["key"]]
                if "ref" in loss_prms:
                    ref = output[loss_prms["ref"]] * loss_prms["mult"]
                else:
                    ref = jnp.zeros_like(predicted)

                if predicted.shape[-1] == 1:
                    predicted = jnp.squeeze(predicted, axis=-1)
                
                nel = np.prod(ref.shape)
                shape_mask=[ref.shape[0]]+[1]*(len(predicted.shape)-1)
                if ref.shape[0] == output["isys"].shape[0]:
                    nel = nel * nat / ref.shape[0]
                    ref = ref * data["true_atoms"].reshape(*shape_mask)
                    predicted = predicted * data["true_atoms"].reshape(*shape_mask)
                elif ref.shape[0] == output["natoms"].shape[0]:
                    nel = nel * nsys / ref.shape[0]
                    ref = ref * data["true_sys"].reshape(*shape_mask)
                    predicted = predicted * data["true_sys"].reshape(*shape_mask)

                loss_type = loss_prms["type"]
                if loss_type == "mse":
                    loss = jnp.sum((predicted - ref) ** 2)
                elif loss_type == "log_cosh":
                    loss = jnp.sum(optax.log_cosh(predicted, ref))
                elif loss_type == "rmse+mae":
                    loss = (jnp.sum((predicted - ref) ** 2)) ** 0.5 + jnp.sum(
                        jnp.abs(predicted - ref)
                    )
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")

                

                loss_tot = loss_tot + loss_prms["weight"] * loss / nel

            return loss_tot

        loss, grad = jax.value_and_grad(loss_fn)(variables)
        updates, opt_st = optimizer.update(grad, opt_st, params=variables)
        variables = optax.apply_updates(variables, updates)
        variables_ema, ema_st = ema.update(variables, ema_st)
        return variables, variables_ema, opt_st, ema_st, loss

    @jax.jit
    def validation(variables, data):
        _, _, output = model._energy_and_forces(variables, data)
        nsys = jnp.sum(data["true_sys"])
        nat = jnp.sum(data["true_atoms"])
        rmses = {}
        maes = {}
        for name, loss_prms in loss_definition.items():
            predicted = output[loss_prms["key"]]
            if "ref" in loss_prms:
                ref = output[loss_prms["ref"]] * loss_prms["mult"]
            else:
                ref = jnp.zeros_like(predicted)
            if predicted.shape[-1] == 1:
                predicted = jnp.squeeze(predicted, axis=-1)
            # nel = ref.shape[0]
            nel = np.prod(ref.shape)
            if ref.shape[0] == data["isys"].shape[0]:
                nel = nel * nat / ref.shape[0]
            elif ref.shape[0] == data["natoms"].shape[0]:
                nel = nel * nsys / ref.shape[0]

            rmse = (jnp.sum((predicted - ref) ** 2) / nel) ** 0.5
            mae = jnp.sum(jnp.abs(predicted - ref)) / nel

            rmses[name] = rmse
            maes[name] = mae
        return rmses, maes, output

    if "energy_terms" in training_parameters:
        model.set_energy_terms(training_parameters["energy_terms"])
    print("energy terms:", model.energy_terms)

    keep_all_bests = training_parameters.get("keep_all_bests", False)
    previous_best_name = None
    rmse_tot_best = np.inf

    ### Training loop ###
    start = time.time()

    fetch_time = 0.0
    preprocess_time = 0.0
    step_time = 0.0

    rmses_prev = defaultdict(lambda: np.inf)
    count = 0
    restore_count = 0
    max_restore_count = training_parameters.get("max_restore_count", 5)
    variables = deepcopy(model.variables)
    variables_save = deepcopy(variables)
    variables_ema_save = deepcopy(model.variables)
    print("Starting training...")
    for epoch in range(max_epochs):
        for _ in range(nbatch_per_epoch):
            # fetch data
            s = time.time()
            data = next(training_iterator)
            e = time.time()
            fetch_time += e - s

            # preprocess data
            s = time.time()
            inputs = model.preprocess(**data)
            e = time.time()
            preprocess_time += e - s

            # train step
            s = time.time()
            opt_st.inner_states["trainable"].inner_state[1].hyperparams[
                "learning_rate"
            ] = schedule(count)
            variables, model.variables, opt_st, ema_st, loss = train_step(
                variables, model.variables, opt_st, ema_st, inputs
            )
            count += 1
            # jax.block_until_ready(state)
            e = time.time()
            step_time += e - s

        rmses_avg = defaultdict(lambda: 0.0)
        maes_avg = defaultdict(lambda: 0.0)
        for _ in range(nbatch_per_validation):
            data = next(validation_iterator)
            inputs = model.preprocess(**data)
            rmses, maes, output = validation(model.variables, inputs)
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
        current_lr = schedule(count)
        print("")
        print(f"Step {epoch+1}, lr={current_lr:.3e}, loss = {loss:.3e}")
        metrics = {
            "loss": loss,
            "lr": current_lr,
            "step": epoch,
            "count": count,
            "time": time.time() - start,
        }
        rmse_tot = 0.0
        for k in rmses_avg.keys():
            mult = loss_definition[k]["mult"]
            unit = (
                "(" + loss_definition[k]["unit"] + ")"
                if "unit" in loss_definition[k]
                else ""
            )
            rmse_tot = rmse_tot + rmses_avg[k] * loss_definition[k]["weight"]**0.5
            metrics[f"rmse_{k}"] = rmses_avg[k] / mult
            metrics[f"mae_{k}"] = maes_avg[k] / mult
            if rmses_avg[k]/mult < 1.e-2:
                print(
                    f"    rmse_{k}= {rmses_avg[k]/mult:10.3e} ; mae_{k}= {maes_avg[k]/mult:10.3e}   {unit}"
                )
            else:
                print(
                    f"    rmse_{k}= {rmses_avg[k]/mult:10.3f} ; mae_{k}= {maes_avg[k]/mult:10.3f}   {unit}"
                )
        metrics["rmse_tot"] = rmse_tot
        

        # print("fetch time = {fetch_time:.5f}; preprocess time = {preprocess_time:.5f}; train time = {step_time:.5f}")
        fetch_time = 0.0
        step_time = 0.0
        preprocess_time = 0.0
        restore = False
        reinit = False
        for k, rmse in rmses_avg.items():
            if np.isnan(rmse):
                restore = True
                reinit = True
                break
            if "threshold" in loss_definition[k]:
                thr = loss_definition[k]["threshold"]
                if rmse > thr * rmses_prev[k]:
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
        rmses_prev = rmses_avg
        
        model.save(output_directory+"latest_model.fnx")

        if rmse_tot < rmse_tot_best:
            rmse_tot_best = rmse_tot
            metrics["rmse_tot_best"] = rmse_tot_best
            if keep_all_bests:
                best_name = output_directory+f"best_model{stage_prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}.fnx"
                model.save(best_name)
            
            best_name = output_directory+f"best_model{stage_prefix}.fnx"
            model.save(best_name)
            print("New best model saved to:", best_name)

        if is_end(metrics):
            print("Stage finished.")
            break

    end = time.time()
    print(f"Training time: {end-start} s")
    print("")

    filename = output_directory+f"final_model{stage_prefix}.fnx"
    model.save(filename)
    print("Final model saved to:", filename)

    return metrics, filename


if __name__ == "__main__":
    main()
