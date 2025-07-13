#!/usr/bin/env python3
import numpy as np
import argparse
from pathlib import Path

from fennol.utils.periodic_table import PERIODIC_TABLE_REV_IDX
from fennol.utils.io import last_xyz_frame
from fennol.utils import parse_cell, detect_topology


def main():
    parser = argparse.ArgumentParser(description="detect topology from an xyz file")
    parser.add_argument(
        "xyz_file", type=Path, help="file containing the geometry. If the file has multiple frames, only the last frame is used."
    )
    parser.add_argument(
        "--cell",
        type=float,
        nargs="+",
        help="PBC cell. If 9 floats, it corresponds to the sequence of the three cell vectors. If 6 floats, it corresponds to the lengths and angles (in degrees) of the cell vectors. If 3 floats, it corresponds to the lengths of the cell vectors. If 1 float, it corresponds to the length of a cubic cell.",
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
        help="output file for the topology data. defaults to <stem(xyz_file)>.topo",
    )
    args = parser.parse_args()

    # check the input file
    input_file = args.xyz_file.resolve()
    assert input_file.exists(), f"Input file {input_file} does not exist"
    system_name = input_file.stem
    has_comment_line = not args.nocomment

    print(f"Reading coordinates from: {input_file}")
    symbols, coordinates, comment = last_xyz_frame(
        input_file, has_comment_line=has_comment_line
    )
    print(f"Read {len(symbols)} atoms.")
    coordinates = np.array(coordinates)
    species = np.array([PERIODIC_TABLE_REV_IDX[s] for s in symbols], dtype=np.int32)

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

    print("Detecting topology...")
    topology = detect_topology(species, coordinates, cell=cell)

    outfile = args.outfile if args.outfile else system_name + ".topo"
    np.savetxt(outfile, topology + 1, fmt="%d")
    print(f"Topology saved to {outfile}")


if __name__ == "__main__":
    main()
