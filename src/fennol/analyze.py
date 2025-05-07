import sys, os, io
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from .models import FENNIX
from .utils.io import xyz_reader
from .utils.periodic_table import PERIODIC_TABLE_REV_IDX
from .utils.atomic_units import AtomicUnits as au
import yaml
import pickle
import json


def main():
    parser = argparse.ArgumentParser(prog="fennol_analyze")
    parser.add_argument(
        "input_file", type=Path, help="file containing the geometries to analyze"
    )
    parser.add_argument(
        "model_file", type=Path, help="file containing the model to use"
    )
    parser.add_argument("--output", "-o", type=Path, help="file to write the output to")
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="batch size to use for the model",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to use for the model"
    )
    parser.add_argument("-f64", action="store_true", help="use double precision")
    parser.add_argument("--periodic", action="store_true", help="use PBC")
    parser.add_argument(
        "--format", type=str, default="xyz", help="format of the input file"
    )
    parser.add_argument(
        "--output_keys",
        type=str,
        nargs="+",
        help="keys to output",
        default=["total_energy"],
    )
    args = parser.parse_args()

    # set the device
    device = args.device.lower()
    if device == "cpu":
        device = "cpu"
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

    # set the precision
    if args.f64:
        jax.config.update("jax_enable_x64", True)
        fprec = "float64"
    else:
        fprec = "float32"
    jax.config.update("jax_default_matmul_precision", "highest")


    # check the input file
    input_file = args.input_file.resolve()
    assert input_file.exists(), f"Input file {input_file} does not exist"
    # assert args.format == "xyz", f"Only xyz format is supported for now"
    assert args.format in ["xyz","arc","pkl"], f"Only xyz, arc and pkl formats are supported for now"
    output_keys = args.output_keys
    xyz_indexed = args.format == "arc"

    # load the model
    model_file: Path = args.model_file.resolve()
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=True)

    # check the output file
    if args.output is not None:
        output_file: Path = args.output.resolve()
        assert not output_file.exists(), f"Output file {args.output} already exists"
        assert output_file.suffix in [
            ".pkl",
            ".json",
            ".yaml",
        ], f"Unsupported output file format {output_file.suffix}"

    # print metadata
    metadata = {
        "input_file": str(input_file),
        "model_file": str(model_file),
        "output_keys": output_keys,
        "energy_unit": model.energy_unit,
    }
    print("metadata:")
    for k,v in metadata.items():
        print(f"  {k}: {v}")

    # define the model prediction function
    def model_predict(batch):
        natoms = np.array([frame["natoms"] for frame in batch])
        batch_index = np.concatenate([frame["batch_index"] for frame in batch])
        species = np.concatenate([frame["species"] for frame in batch])
        xyz = np.concatenate([frame["coordinates"] for frame in batch], axis=0)
        inputs = {
            "species": species,
            "coordinates": xyz,
            "batch_index": batch_index,
            "natoms": natoms,}
        if args.periodic:
            cells = np.concatenate([frame["cell"] for frame in batch], axis=0)
            inputs["cells"] = cells

        if "forces" in output_keys:
            e, f, output = model.energy_and_forces(
                **inputs
            )
        else:
            e, output = model.total_energy(
                **inputs
            )
        return output

    # define the function to process a batch
    def process_batch(batch):
        output = model_predict(batch)
        natoms = np.array([frame["natoms"] for frame in batch])
        if args.periodic:
            cells = np.array(output["cells"])
        species = np.array(output["species"])
        coordinates = np.array(output["coordinates"])
        natshift = np.concatenate([np.array([0], dtype=np.int32), np.cumsum(natoms)])
        frames_data = []
        for i in range(len(batch)):
            frame_data = {
                "species": species[natshift[i] : natshift[i + 1]].tolist(),
                "coordinates": coordinates[natshift[i] : natshift[i + 1]].tolist(),
            }
            if args.periodic:
                frame_data["cell"] = cells[i].tolist()

            for k in output_keys:
                if k not in output:
                    raise ValueError(f"Output key {k} not found")
                v = output[k]
                if v.shape[0] == species.shape[0]:
                    frame_data[k] = v[natshift[i] : natshift[i + 1]].tolist()
                elif v.shape[0] == natoms.shape[0]:
                    frame_data[k] = v[i].tolist()
                else:
                    raise ValueError(f"Output key {k} has wrong shape")

            frames_data.append(frame_data)
        return frames_data

    ### start processing the input file
    # reader = xyz_reader(input_file, has_comment_line=True, indexed=xyz_indexed)
    if args.format in ["arc","xyz"]:
        def reader():
            for symbols, xyz, comment in xyz_reader(input_file, has_comment_line=True, indexed=xyz_indexed):
                species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols])
                inputs = {"species": species, "coordinates": xyz, "natoms":species.shape[0]}
                if args.periodic:
                    cell = np.array([float(x) for x in comment.split()]).reshape(1,3,3)
                    inputs["cell"] = cell
                yield inputs

    elif args.format == "pkl":
        with open(input_file, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            if "frames" in data:
                frames = data["frames"]
            elif "training" in data:
                frames = data["training"]
                if "validation" in data:
                    frames.extend(data["validation"])
            else:
                raise ValueError("No frames found in the input file")
        else:
            frames = data
        
        def reader():
            for frame in frames:
                species = np.array(frame["species"])
                coordinates = np.array(frame["coordinates"])
                inputs = {"species": species, "coordinates": coordinates,"natoms":species.shape[0]}
                if args.periodic:
                    cell = np.array(frame["cell"]).reshape(1,3,3)
                    inputs["cell"] = cell
                if "total_charge" in frame:
                    inputs["total_charge"] = frame["total_charge"]
                yield inputs

    batch = []
    output_data = []
    ibatch = 0
    for frame in reader():
        batch_index = np.full(frame["natoms"], ibatch, dtype=np.int32)
        frame["batch_index"] = batch_index
        batch.append(frame)
        ibatch += 1
        if len(batch) == args.batch_size:
            frames_data = process_batch(batch)
            if args.output is None:
                for frame_data in frames_data:
                    print("\n---\n")
                    for k in frame_data:
                        print(f"{k}: {frame_data[k]}")
            output_data.extend(frames_data)
            batch = []
            ibatch = 0
    # process the last batch
    if len(batch) > 0:
        frames_data = process_batch(batch)
        if args.output is None:
            for frame_data in frames_data:
                print("\n---\n")
                for k in frame_data:
                    print(f"{k}: {frame_data[k]}")
        output_data.extend(frames_data)

    # write the output to a file
    if args.output is not None:
        output_data = {
            "metadata": metadata,
            "frames": output_data,
        }
        if output_file.suffix == ".json":
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
        elif output_file.suffix == ".yaml":
            with open(args.output, "w") as f:
                yaml.dump(output_data, f, sort_keys=False)
        elif output_file.suffix == ".pkl":
            with open(args.output, "wb") as f:
                pickle.dump(output_data, f)


if __name__ == "__main__":
    main()
