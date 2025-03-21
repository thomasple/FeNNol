import argparse
from pathlib import Path
import yaml
import json
import os
from flax import traverse_util
import jax
import dataclasses

from .fennix import FENNIX


class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _device = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", _device)
    ### Read the parameter file
    parser = argparse.ArgumentParser(prog="fennol_inspect")
    parser.add_argument("model_file", type=Path, help="Model file")
    parser.add_argument(
        "-s",
        "--short",
        action="store_true",
        help="Print only module names in the list of modules",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Print all module attributes and automatically added modules",
    )
    parser.add_argument(
        "-p","--prm", action="store_true", help="Print parameter shapes"
    )
    args = parser.parse_args()
    model_file = args.model_file
    # param_shapes = args.param_shapes
    # short = args.short
    # all = args.all

    ### Load the model
    if model_file.suffix == ".yaml" or model_file.suffix == ".yml":
        with open(model_file, "r") as f:
            model_dict = yaml.safe_load(f)
        model = FENNIX(**model_dict,rng_key=jax.random.PRNGKey(0))
    else:
        model = FENNIX.load(model_file)
    print(f"# filename: {model_file}")
    print(inspect_model(model, **vars(args)))


def inspect_model(model, prm=False, short=False, all=False, **kwargs):

    model_dict = model._input_args

    inspect_dict = {
        "energy_unit": model.energy_unit,
        "energy_terms": model.energy_terms,
        "cutoff": model.cutoff,
    }
    
    inspect_dict["preprocessing"] = dict(model_dict["preprocessing"])
    if not all:
        inspect_dict["modules"] = dict(model_dict["modules"])
    else:
        mods = {}
        for mod, inp in model.modules.layers:
            m = mod(**inp)
            key = m.name
            if key is None or key.startswith("_"):
                continue
            mods[key] = {}
            for field in dataclasses.fields(m):
                if field.name not in ["name", "parent"]:
                    mods[key][field.name] = getattr(m, field.name)
        inspect_dict["modules"] = mods
    
    if short:
        inspect_dict["preprocessing"] = list(inspect_dict["preprocessing"].keys())
        inspect_dict["modules"] = list(inspect_dict["modules"].keys())
    
    data = "# MODEL DESCRIPTION\n"
    data += yaml.dump(inspect_dict, sort_keys=False, Dumper=IndentDumper)

    params = model.variables["params"] if "params" in model.variables else {}

    if prm:
        shapes = traverse_util.path_aware_map(
            lambda p, v: f"[{','.join(str(i) for i in v.shape)}]",
            params,
        )

        data = (
            data
            + "\n\n# PARAMETER SHAPES\n"
            + yaml.dump(
                shapes,
                sort_keys=False,
            )
        )

    number_of_params = sum(
        jax.tree.leaves(
            traverse_util.path_aware_map(
                lambda p, v: v.size,
                params,
            )
        )
    )

    data += f"\n# NUMBER OF PARAMETERS: {number_of_params:_}"
    return data


if __name__ == "__main__":
    main()
