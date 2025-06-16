import sys, os, io
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from .models import FENNIX
from .utils import parse_cell
from .utils.io import xyz_reader,human_time_duration
from .utils.periodic_table import PERIODIC_TABLE_REV_IDX
from .utils.atomic_units import AtomicUnits as au
import yaml
import pickle
from flax.core import freeze, unfreeze
import time
import multiprocessing as mp

def get_file_reader(input_file,file_format=None,has_comment_line=True, periodic=False):
    # check the input file
    input_file = Path(input_file).resolve()
    assert input_file.exists(), f"Input file {input_file} does not exist"
    # assert args.format == "xyz", f"Only xyz format is supported for now"
    if file_format is None:
        file_format = input_file.suffix[1:]  # remove the dot
    file_format = file_format.lower()
    if file_format == "pickle":
        file_format = "pkl"

    assert file_format in [
        "xyz",
        "arc",
        "pkl",
    ], f"Only xyz, arc and pkl formats are supported for now"
    xyz_indexed = file_format == "arc"

    if file_format in ["arc", "xyz"]:

        def reader():
            for symbols, xyz, comment in xyz_reader(
                input_file, has_comment_line=has_comment_line, indexed=xyz_indexed
            ):
                species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols])
                inputs = {
                    "species": species,
                    "coordinates": xyz,
                    "natoms": species.shape[0],
                    "total_charge": 0,  # default total charge
                }
                if periodic:
                    box = np.array(comment.split(), dtype=float)
                    cell = parse_cell(box)
                    inputs["cell"] = cell
                yield inputs

    elif file_format == "pkl":
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
                inputs = {
                    "species": species,
                    "coordinates": coordinates,
                    "natoms": species.shape[0],
                    "total_charge": frame.get("total_charge", 0),
                }
                if periodic:
                    cell_key = "cells" if "cells" in frame else "cell"
                    cell = np.array(frame[cell_key]).reshape(3, 3)
                    inputs["cell"] = cell
                yield inputs
    
    return reader


def fennix_analyzer(input_file, model, output_keys,file_format=None,periodic=False,has_comment_line=True,batch_size=1):
    
    reader = get_file_reader(input_file,file_format=file_format,periodic=periodic,has_comment_line=has_comment_line)

    # define the model prediction function
    def model_predict(batch):
        natoms = np.array([frame["natoms"] for frame in batch], dtype=np.int32)
        batch_index = np.concatenate([frame["batch_index"] for frame in batch])
        species = np.concatenate([frame["species"] for frame in batch])
        xyz = np.concatenate([frame["coordinates"] for frame in batch], axis=0)
        total_charge = np.array([frame["total_charge"] for frame in batch], dtype=np.int32)
        inputs = {
            "species": species,
            "coordinates": xyz,
            "batch_index": batch_index,
            "natoms": natoms,
            "total_charge": total_charge,
        }
        if periodic:
            cells = np.stack([frame["cell"] for frame in batch], axis=0)
            inputs["cells"] = cells

        if "forces" in output_keys:
            e, f, output = model.energy_and_forces(**inputs,gpu_preprocessing=True)
        else:
            e, output = model.total_energy(**inputs,gpu_preprocessing=True)
        return output

    # define the function to process a batch
    def process_batch(batch):
        output = model_predict(batch)
        natoms = np.array([frame["natoms"] for frame in batch])
        if periodic:
            cells = np.array(output["cells"])
        species = np.array(output["species"])
        coordinates = np.array(output["coordinates"])
        natshift = np.concatenate([np.array([0], dtype=np.int32), np.cumsum(natoms)])
        frames_data = []
        for i in range(len(batch)):
            frame_data = {
                "species": species[natshift[i] : natshift[i + 1]],
                "coordinates": coordinates[natshift[i] : natshift[i + 1]],
                "total_charge": int(output["total_charge"][i]),
            }
            if periodic:
                frame_data["cell"] = cells[i]

            for k in output_keys:
                if k not in output:
                    raise ValueError(f"Output key {k} not found")
                v = np.asarray(output[k])
                # if scalar or only 1 element, convert to float
                if v.ndim == 0:
                    v = float(v)
                if v.size == 1:
                    v = v.flatten()[0]
                if v.shape[0] == species.shape[0]:
                    frame_data[k] = v[natshift[i] : natshift[i + 1]]
                elif v.shape[0] == natoms.shape[0]:
                    frame_data[k] = v[i]
                else:
                    raise ValueError(f"Output key {k} has wrong shape {v.shape} {natoms.shape} {species.shape}")

            frames_data.append(frame_data)
        
        return frames_data

    batch = []
    iframe = 0
    for frame in reader():
        batch_index = np.full(frame["natoms"], iframe, dtype=np.int32)
        frame["batch_index"] = batch_index
        batch.append(frame)
        iframe += 1
        if len(batch) == batch_size:
            frames_data = process_batch(batch)
            for frame_data in frames_data:
                yield frame_data
            # output_data.extend(frames_data)
            batch = []
            iframe = 0
    # process the last batch
    if len(batch) > 0:
        frames_data = process_batch(batch)
        for frame_data in frames_data:
            yield frame_data

def main():
    parser = argparse.ArgumentParser(prog="fennol_analyze")
    parser.add_argument(
        "input_file", type=Path, help="file containing the geometries to analyze"
    )
    parser.add_argument(
        "model_file", type=Path, help="file containing the model to use"
    )
    parser.add_argument(
        "-o", "--outfile", type=Path, help="file to write the output to"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="batch size to use for the model",
    )
    parser.add_argument("--device", type=str, help="device to use for the model")
    parser.add_argument("-f64", action="store_true", help="use double precision")
    parser.add_argument("--periodic", action="store_true", help="use PBC")
    parser.add_argument(
        "--format",
        type=str,
        help="format of the input file. Default: auto-detect from file extension",
    )
    parser.add_argument(
        "-c",
        "--nocomment",
        action="store_true",
        help="flag to indicate that the input file does not have a comment line. Only used for xyz and arc formats",
    )
    parser.add_argument(
        "--output_keys",
        type=str,
        nargs="+",
        help="keys to output",
        default=["total_energy"],
    )
    parser.add_argument(
        "--nblist",
        nargs=3,
        metavar=("mult_size", "add_neigh", "add_atoms"),
        help="neighbour list parameters: mult_size, add_neigh, add_atoms. If not provided, the default values from the model will be used.",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        action="store_true",
        help="add metadata as a first frame.",
    )
    args = parser.parse_args()

    # set the device
    if args.device:
        device = args.device.lower()
    elif "FENNOL_DEVICE" in os.environ:
        device = os.environ["FENNOL_DEVICE"].lower()
        print(f"# Setting device from env FENNOL_DEVICE={device}")
    else:
        device = "cpu"
    if device == "cpu":
        jax.config.update("jax_platforms", "cpu")
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

    output_keys = args.output_keys

    # load the model
    model_file: Path = args.model_file.resolve()
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=True)

    preproc_state = unfreeze(model.preproc_state)
    layer_state = []
    for st in preproc_state["layers_state"]:
        stnew = unfreeze(st)
        # if "nblist_mult_size" in training_parameters:
        #     stnew["nblist_mult_size"] = training_parameters["nblist_mult_size"]
        # if "nblist_add_neigh" in training_parameters:
        #     stnew["add_neigh"] = training_parameters["nblist_add_neigh"]
        # if "nblist_add_atoms" in training_parameters:
        #     stnew["add_atoms"] = training_parameters["nblist_add_atoms"]
        if args.nblist is not None:
            mult_size, add_neigh, add_atoms = args.nblist
            stnew["nblist_mult_size"] = float(mult_size)
            stnew["add_neigh"] = int(add_neigh)
            stnew["add_atoms"] = int(add_atoms)
        layer_state.append(freeze(stnew))

    preproc_state["layers_state"] = tuple(layer_state)
    preproc_state["check_input"] = False
    model.preproc_state = freeze(preproc_state)
    # model.preproc_state = model.preproc_state.copy({"check_input": False})

    # check the output file
    if args.outfile is not None:
        output_file: Path = args.outfile.resolve()
        assert not output_file.exists(), f"Output file {args.outfile} already exists"
        out_format = output_file.suffix.lower()[1:]
        if out_format == "yml":
            out_format = "yaml"
        if out_format == "pickle":
            out_format = "pkl"
        if out_format == "hdf5":
            out_format = "h5"
        if out_format == "msgpack":
            out_format = "mpk"
        assert out_format in [
            "pkl",
            "yaml",
            "h5",
            "mpk",
        ], f"Unsupported output file format {output_file.suffix}"

    # print metadata
    metadata = {
        "input_file": str(args.input_file),
        "model_file": str(model_file),
        "output_keys": output_keys,
        "energy_unit": model.energy_unit,
    }
    print("metadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    def make_dumpable(data):
        for k in data:
            if isinstance(data[k], np.ndarray):
                data[k] = data[k].tolist()
        return data
    
    def queue_reader(queue):
        if args.outfile is not None:
            if out_format == "yaml":
                of = open(args.outfile, "w")
                def dump(data):
                    of.write("\n---\n")
                    i=yaml.dump(make_dumpable(data), of, sort_keys=False)
                    of.flush()
                    return i
            elif out_format == "pkl":
                of = open(args.outfile, "wb")
                def dump(data):
                    i=pickle.dump(data,of)
                    of.flush()
                    return i
            elif out_format == "h5":
                import h5py
                of = h5py.File(args.outfile, "w")
                global iframeh5
                iframeh5 = 0  # start from frame 1
                def dump(data):
                    global iframeh5
                    of.create_group(f'{iframeh5}', track_order=True)
                    for k in data:
                        of[f"{iframeh5}/{k}"] = data[k]
                    iframeh5 += 1
                    return None
            elif out_format == "mpk":
                import msgpack
                of = open(args.outfile, "wb")
                def dump(data):
                    i=msgpack.pack(make_dumpable(data), of)
                    of.flush()
                    return i
            else:
                raise ValueError(f"Unsupported output file format {out_format}")
        else:
            def dump(data):
                print("\n---\n")
                data = make_dumpable(data)
                for k in data:
                    print(f"{k}: {data[k]}")
                return None
        while True:
            data = queue.get()
            if data is None:
                break
            dump(data)
        if args.outfile is not None:
            of.close()
    # create a multiprocessing queue to dump the output
    queue = mp.Queue()
    p = mp.Process(target=queue_reader, args=(queue,))
    p.start()
    if args.metadata and args.outfile is not None:
        queue.put(metadata)

    error = None
    try:
        time_start = time.time()
        ibatch = 0
        for iframe,frame in enumerate(fennix_analyzer(
            args.input_file,
            model,
            output_keys=output_keys,
            file_format=args.format,
            periodic=args.periodic,
            has_comment_line=not args.nocomment,
            batch_size=args.batch_size,
        )):
            queue.put(frame)
            if (iframe+1) % args.batch_size == 0 and args.outfile is not None:
                ibatch += 1
                elapsed = time.time() - time_start
                print(f"# Processed batch {ibatch}. Elapsed time: {human_time_duration(elapsed)}")
                # output_data.extend(frames_data)
    except KeyboardInterrupt:
        print("# Interrupted by user. Exiting...")
    except Exception as error:
        print(f"# Exiting with Exception: {error}")

    # wait for the process to finish
    queue.put(None)
    print("# Waiting for the writing process to finish...")
    p.join()
    print("# All done in ", human_time_duration(time.time() - time_start))

    if error is not None:
        raise error
    


if __name__ == "__main__":
    main()
