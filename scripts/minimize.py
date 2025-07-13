#!/usr/bin/env python3
import sys, os, io
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from fennol.models import FENNIX
from fennol.utils import parse_cell, cell_vectors_to_lengths_angles
from fennol.utils.io import last_xyz_frame, write_xyz_frame, write_arc_frame
from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX
from fennol.utils.atomic_units import AtomicUnits as au
from fennol.md.utils import optimize_fire2
from scipy.optimize import minimize


def main():
    parser = argparse.ArgumentParser(
        description="Minimize the energy of a molecular geometry using a FENNIX model"
    )
    parser.add_argument(
        "xyz_file",
        type=Path,
        help="file containing the geometry to optimize. If the file has multiple frames, only the last frame is used.",
    )
    parser.add_argument(
        "model_file", type=Path, help="file containing the model to use"
    )
    parser.add_argument("--device", type=str, help="device to use for the model")
    parser.add_argument("-f64", action="store_true", help="use double precision")
    parser.add_argument(
        "--cell",
        type=float,
        nargs="+",
        help="PBC cell vectors. If 9 floats, it corresponds to the sequence of the three cell vectors. If 6 floats, it corresponds to the lengths and angles (in degrees) of the cell vectors. If 3 floats, it corresponds to the lengths of the cell vectors. If 1 float, it corresponds to the length of a cubic cell.",
    )
    parser.add_argument(
        "-c",
        "--nocomment",
        action="store_true",
        help="flag to indicate that the xyz file does not have a comment line",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="output file for the optimized geometry. If not provided, the output will be written to <xyz_file>.opt.xyz",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Initial time step for the optimization (default: 0.002)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum number of optimization steps (default: 10000)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="Convergence tolerance on forces (default: 1e-2)",
    )
    parser.add_argument(
        "--keep-every",
        type=int,
        default=-1,
        help="Keep every Nth frame during optimization. If -1, only the final frame is kept.",
    )
    parser.add_argument(
        "--dxmax",
        type=float,
        help="Maximum displacement per step during optimization (default: not applied)",
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
        print("# No device specified, using CPU as default")

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

    # check the input file
    input_file = args.xyz_file.resolve()
    assert input_file.exists(), f"Input file {input_file} does not exist"
    has_comment_line = not args.nocomment

    # load the model
    model_file: Path = args.model_file.resolve()
    assert model_file.exists(), f"Model file {model_file} does not exist"
    model = FENNIX.load(model_file, use_atom_padding=False)
    convert = au.KCALPERMOL / model.Ha_to_model_energy

    # read the coordinates from the xyz file
    print(f"Reading coordinates from: {input_file}")
    symbols, coordinates, comment = last_xyz_frame(
        input_file, has_comment_line=has_comment_line
    )
    print(f"Read {len(symbols)} atoms.")
    coordinates = np.array(coordinates, dtype=fprec)
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)
    nat = len(species)
    inputs = {
        "species": species,
        "natoms": np.array([nat], dtype=np.int32),
        "batch_index": np.array([0] * nat, dtype=np.int32),
    }

    ## get cell vectors
    cell = parse_cell(args.cell)
    if cell is None and comment:
        try:
            box = np.array(comment.split(), dtype=float)
            cell = parse_cell(box)
        except:
            cell = None
    if cell is not None:
        print("cell matrix:")
        for l in cell:
            print("  ", l)
        inputs["cells"] = cell.reshape(1, 3, 3)

    def energy_force_fn(coordinates):
        e, f, _ = model.energy_and_forces(
            **inputs, coordinates=coordinates, gpu_preprocessing=True
        )
        e = float(e[0]) * convert / nat
        f = np.array(f) * convert
        return e, f

    keep_every = args.keep_every
    dxmax = args.dxmax

    print("Starting geometry optimization...")
    results = optimize_fire2(
        coordinates,
        energy_force_fn,
        atol=args.tol,
        dt=args.dt,
        Nmax=args.max_steps,
        logoutput=True,
        keep_every=keep_every,
        max_disp=dxmax,  # convert Angstroms to Bohr
    )
    coordinates = results[0]
    success = results[1]
    print("#######################################################")
    if success:
        print("Optimization converged successfully!")
    else:
        print("Optimization did not converge... Writing the last frame anyway.")
    # write the output
    output_file = args.outfile if args.outfile else input_file.with_suffix(".opt.xyz")
    with open(output_file, "w") as f:
        write_xyz_frame(f, symbols, coordinates, cell=cell)
    print(f"Final configuration written to {output_file}")

    if keep_every > 0:
        output_arc_file = input_file.with_suffix(".opt.arc")
        frames = results[2]
        with open(output_arc_file, "w") as f:
            for frame in frames:
                write_arc_frame(f, symbols, frame, cell=cell)


if __name__ == "__main__":
    main()
