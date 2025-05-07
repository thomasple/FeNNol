from typing import Callable, Optional, Dict, List, Tuple, Union, Any, NamedTuple
import optax
import jax
import jax.numpy as jnp
import numpy as np
import operator
from flax import traverse_util
import json
import re

from optax._src import base
from optax._src import wrappers



class AddWeightDiffState(NamedTuple):
    ref_weights: Any

def add_weights_difference(
    weight_decay: Union[float, jax.Array] = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None
) -> base.GradientTransformation:
  """weight decay toward initial weights."""
  def init_fn(params):
    return AddWeightDiffState(ref_weights=params)

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree_util.tree_map(
        lambda g, p, pref: g + weight_decay * (p-pref), updates, params,state.ref_weights)
    return updates, state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return wrappers.masked(
        base.GradientTransformation(init_fn, update_fn), mask)
  return base.GradientTransformation(init_fn, update_fn)



def add_grokfast(
    alpha: float = 0.9,
    l: float = 1.,
)-> base.GradientTransformation:
    """Grokfast: amplify slow gradients by exponential moving average."""

    ema: base.GradientTransformation = optax.ema(decay=alpha,debias=False)
    def init_fn(params):
        return ema.init(params)
    
    def update_fn(updates, state, params=None):
        dupdates, state = ema.update(updates, state, params)
        # updates = updates + l*dupdates
        updates = jax.tree_util.tree_map(
            lambda g,d: g+l*d, updates, dupdates)
        return updates, state
    
    return base.GradientTransformation(init_fn, update_fn)


class PROFITState(NamedTuple):
    ref_weights: Any
    istep: int
    main_opt_state: Any
    internal_opt_state: Any

def profit(
    learning_rate: base.ScalarOrSchedule,
    nsteps_ref: int = 1,
    main_opt: str = 'adam',
    main_opt_params: Dict[str, Any] = {},
    internal_opt: str = 'sgd',
    internal_opt_params: Dict[str, Any] = {},
    **kwargs
)-> base.GradientTransformation:
    """PROFIT optimizer for fine-tuning https://arxiv.org/pdf/2412.01930"""

    main_opt_params = {'learning_rate': learning_rate,**main_opt_params}
    main_opt = eval(
        main_opt,
        {"__builtins__": None},
        {**optax.__dict__},
    )(**main_opt_params)

    internal_opt_params = {'learning_rate': .1,**internal_opt_params}
    internal_opt = eval(
        internal_opt,
        {"__builtins__": None},
        {**optax.__dict__},
    )(**internal_opt_params)


    def init_fn(params):
        return PROFITState(
            ref_weights=params,
            istep=0,
            main_opt_state=main_opt.init(params),
            internal_opt_state=internal_opt.init(params),
        )
    

    def update_main(gradients,main_opt_state,internal_opt_state,params,params_ref):
        delta = jax.tree_util.tree_map(lambda p, pref: p-pref, params, params_ref)
        dot = jax.tree.reduce(
           operator.add,
           jax.tree_util.tree_map(lambda g,d: (g*d).sum(), gradients, delta)
        )
        delta2 = jax.tree.reduce(
           operator.add,
           jax.tree_util.tree_map(lambda d: (d**2).sum(), delta)
        )
        proj = dot/(delta2+1.e-6)

        gradients = jax.lax.cond(dot>=0,
            lambda g,d: g,
            lambda g,d: jax.tree_util.tree_map(lambda x: proj*x, d),
            gradients,delta
        )
        updates,main_opt_state = main_opt.update(gradients,main_opt_state,params)
        updates = jax.tree_util.tree_map(lambda g,d: g-d, updates, delta)
        return updates,main_opt_state,internal_opt_state
    
    def update_internal(gradients,main_opt_state,internal_opt_state,params,params_ref):
        updates,internal_opt_state = internal_opt.update(gradients,internal_opt_state,params)
        return updates,main_opt_state,internal_opt_state
    
    def update_fn(gradients, state, params):
        istep = state.istep % (nsteps_ref+1)
        # jax.debug.print("{i} {j}",i=istep,j=state.istep)

        params_ref = jax.lax.cond(istep==0, lambda a,b: a, lambda a,b: b, params, state.ref_weights)

        updates,main_opt_state,internal_opt_state = jax.lax.cond(
            istep==nsteps_ref,
            update_main,
            update_internal,
            gradients,state.main_opt_state,state.internal_opt_state,params,params_ref
        )

        new_state = PROFITState(
            ref_weights=params_ref,
            istep=state.istep+1,
            main_opt_state=main_opt_state,
            internal_opt_state=internal_opt_state
        )
        return updates, new_state

    return base.GradientTransformation(init_fn, update_fn)

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
        # full_path = "/".join(full_path[1:]).lower()
        status = (default_status, "")
        for path in frozen:
            # if full_path.startswith(path.lower()) and len(path) > len(status[1]):
            if re.match(path.lower(), full_path):
                status = ("frozen", path)
        for path in trainable:
            # if full_path.startswith(path.lower()) and len(path) > len(status[1]):
            if re.match(path.lower(), full_path):
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

    use_grokfast = training_parameters.get("use_grokfast", False)
    if use_grokfast:
        print("Using Grokfast")
        alpha_grokfast = training_parameters.get("alpha_grokfast", 0.9)
        l_grokfast = training_parameters.get("l_grokfast", 1.0)
        grad_processing.append(add_grokfast(alpha=alpha_grokfast, l=l_grokfast))


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
        {**optax.__dict__,"profit":profit},
    )
    print("Optimizer:", optimizer_name)
    optimizer_configuration = training_parameters.get("optimizer_config", {})
    optimizer_configuration["learning_rate"] = 1.0
    grad_processing.append(optimizer(**optimizer_configuration))

    # weight decay
    weight_decay = training_parameters.get("weight_decay", 0.0)
    assert weight_decay >= 0.0, "Weight decay must be positive"
    decay_targets = training_parameters.get("decay_targets", [""])

    def decay_status(full_path, v):
        full_path = "/".join(full_path).lower()
        status = False
        # print(full_path,re.match(r'^params\/', full_path))
        for path in decay_targets:
            if re.match(r"^params/" + path.lower(), full_path):
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
    
    regularize_init_weight = training_parameters.get("regularize_init_weights", 0.)
    if regularize_init_weight > 0.0:
        print("Regularizing toward initial weights with L2 norm:", 
        regularize_init_weight)
        if weight_decay <=0.:
            print(json.dumps(decay_mask, indent=2, sort_keys=False))

        grad_processing.append(add_weights_difference(weight_decay=-regularize_init_weight, mask=decay_mask))

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

def get_lr_schedule(max_epochs,nbatch_per_epoch,training_parameters):
    lr = training_parameters.get("lr", 1.0e-3)
    init_lr = training_parameters.get("init_lr", lr / 25)
    final_lr = training_parameters.get("final_lr", lr / 10000)

    #### LEARNING RATE SCHEDULER ####
    schedule_type = training_parameters.get("schedule_type", "cosine_onecycle").lower()
    schedule_type = training_parameters.get("scheduler", schedule_type).lower()
    schedule_metrics = training_parameters.get("schedule_metrics", "rmse_tot")

    adaptive_scheduler = False
    print("Schedule type:", schedule_type)
    if schedule_type == "cosine_onecycle":
        transition_epochs = training_parameters.get("onecycle_epochs", max_epochs)
        peak_epoch = training_parameters.get("peak_epoch", 0.3 * transition_epochs)
        schedule_ = optax.cosine_onecycle_schedule(
            peak_value=lr,
            div_factor=lr / init_lr,
            final_div_factor=init_lr / final_lr,
            transition_steps=transition_epochs * nbatch_per_epoch,
            pct_start=peak_epoch / transition_epochs,
        )
        sch_state = {"count": 0, "best": np.inf, "lr": init_lr}

        def schedule(state, rmse=None):
            new_state = {**state}
            lr = schedule_(state["count"])
            if rmse is None:
                new_state["count"] += 1
            new_state["lr"] = lr
            return lr, new_state
    
    elif schedule_type == "piecewise_interpolate":
        schedule_params = training_parameters.get("scheduler_parameters", {})
        schedule_ = optax.piecewise_interpolate_schedule(
            **{"init_value":lr,"interpolate_type":"linear",**schedule_params}
        )
        sch_state = {"count": 0, "best": np.inf, "lr": schedule_(0)}
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
        adaptive_scheduler = True

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
    
    return schedule, sch_state, schedule_metrics, adaptive_scheduler