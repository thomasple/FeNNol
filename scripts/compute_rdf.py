#!/usr/bin/env python3
import sys
import time
import numpy as np
import argparse
from numpy import arccos, cos, dot, pi, sin, sqrt
from numpy.linalg import norm

from typing import Union, List

from fennol.utils import parse_cell
from fennol.utils.io import xyz_reader


def apply_pbc(xyz, pairs, cell, rmax, minimum_image):
    cellinv = np.linalg.inv(cell)
    if minimum_image:
        vec = xyz[pairs[1]] - xyz[pairs[0]]
        shift = -np.round(np.dot(vec, cellinv))
        vec = vec + np.dot(shift, cell)
    else:
        shift = -np.floor(np.dot(xyz, cellinv))
        xyz_pbc = xyz + np.dot(shift, cell)

        inv_distances = np.sum(cellinv**2, axis=1) ** 0.5
        num_repeats = np.ceil(rmax * inv_distances).astype(np.int32)
        cell_shift_pbc = np.array(
            np.meshgrid(*[np.arange(-n, n + 1) for n in num_repeats])
        ).T.reshape(-1, 3)
        dvec = np.dot(cell_shift_pbc, cell)

        vec = (
            (xyz_pbc[pairs[1]] - xyz_pbc[pairs[0]])[:, None, :] + dvec[None, :, :]
        ).reshape(-1, 3)
    return vec


def get_pairs(species: Union[List[str], np.ndarray], sp1=None, sp2=None):
    if isinstance(species, list):
        species = np.array(species, dtype=str)
    if sp1 is None:
        if sp2 is None:
            return np.array(np.triu_indices(len(species), 1)), 0.5 * len(species) ** 2
        sp1 = sp2
    id1 = np.flatnonzero(species == sp1)
    num1 = len(id1)
    if sp2 == sp1 or sp2 is None:
        pairsloc = np.triu_indices(len(id1), 1)
        return np.array([id1[pairsloc[0]], id1[pairsloc[1]]]), 0.5 * num1**2
    id2 = np.flatnonzero(species == sp2)
    num2 = len(id2)
    return np.array([np.repeat(id1, len(id2)), np.tile(id2, len(id1))]), num1 * num2


def radial_density(
    species,
    xyz,
    cell=None,
    rmax=10.0,
    dr=0.1,
    sp1=None,
    sp2=None,
    pairs=None,
    timer=False,
    minimum_image=False,
):
    # nat = len(species)
    if timer:
        print(sp1, sp2)
        time0 = time.time()
    if pairs is None:
        pairs, n_pairs_norm = get_pairs(species, sp1, sp2)
    # n_pairs = pairs.shape[1]
    # mult = 1 if sp1 != sp2 else 2
    # print(mult*n_pairs,mult)
    if timer:
        print("pairs:", time.time() - time0)
        time0 = time.time()
    if cell is not None:
        vec = apply_pbc(xyz, pairs, cell, rmax, minimum_image)
    else:
        vec = xyz[pairs[0]] - xyz[pairs[1]]
    dist = np.linalg.norm(vec, axis=-1)
    if timer:
        print("dists:", time.time() - time0)
        time0 = time.time()
    n_bins = int(rmax / dr)
    bins = np.linspace(0, rmax, n_bins + 1)
    r, _ = np.histogram(dist, bins=n_bins, range=(0, rmax))
    if timer:
        print("hist:", time.time() - time0)
    expect = n_pairs_norm * 4.0 / 3.0 * np.pi * ((bins[1:]) ** 3 - bins[:-1] ** 3)
    if cell is not None:
        volume = np.abs(np.linalg.det(cell))
        expect /= volume
    return r / expect


def radial_from_file(
    xyzfile,
    pairs,
    outfile="gr.dat",
    rmax=10,
    dr=0.02,
    thermalize=0,
    stride=1,
    has_comment_line=False,
    watch=False,
    cell=None,
    minimum_image=False,
    weightfile=None,
):
    try:
        # frames = list(read_arc_file(arcfile,max_frames=205))
        # print(len(frames))
        reader = xyz_reader(
            xyzfile,
            has_comment_line=has_comment_line,
            start=thermalize + 1,
            step=stride,
            stream=watch,
        )
        n_bins = int(rmax / dr)
        centers = (np.arange(n_bins) + 0.5) * dr
        nframe = 0
        variable_cell = cell is None
        if not variable_cell:
            # cell=np.array([float(c) for c in cell.split()]).reshape((3,3),order="F")
            cell = np.array(cell).reshape((3, 3))
            print("cell vectors=")
            print(cell)
        # setup_figures(["g(r)"])
        # plt.pause(1.)
        pairs = [p.strip().replace("-", " ").replace("_", " ") for p in pairs]
        header = " ".join(["r"] + ["-".join(p.split()) for p in pairs])
        box = None
        gr_i = np.zeros((n_bins, len(pairs)))
        gr_avg = np.zeros((n_bins, len(pairs)))
        if weightfile is not None:
            fw = open(weightfile, "r")
            grw_avg = None
            w_avg = None
        for frame in reader:
            nframe += 1
            species, xyz, comment = frame
            if variable_cell and comment is not None:
                try:
                    box = np.array(comment.split(), dtype=float)
                    cell = parse_cell(box)
                    print(nframe, "cell=", cell.flatten().tolist())
                except:
                    cell = None
                    print(nframe)
            else:
                print(nframe)

            for i, pair in enumerate(pairs):
                sp1, sp2 = pair.split()
                gr = radial_density(
                    species,
                    xyz,
                    cell,
                    rmax,
                    dr,
                    sp1=sp1,
                    sp2=sp2,
                    timer=False,
                    minimum_image=minimum_image,
                )
                # gr=radial_density_numba(species,xyz,box,rmax,dr,sp1=sp1,sp2=sp2)
                gr_i[:, i] = gr
                gr_avg[:, i] += gr

            # cleanup_ax(plt.gca())
            # plt.plot(centers,gr_avg/nframe)
            # plt.pause(1.)
            np.savetxt(
                outfile, np.column_stack((centers, gr_avg / nframe)), header=header
            )

            if weightfile is not None:
                while True:
                    line = fw.readline()
                    if line or not watch:
                        break
                    time.sleep(2)
                if not line:
                    raise Exception("Error: premature end of weight file!")
                weights = np.array(line.split(), dtype=float)
                if grw_avg is None:
                    grw_avg = np.zeros((n_bins, len(pairs), weights.shape[0]))
                    w_avg = np.zeros(weights.shape[0])
                w_avg += weights
                grw_avg[:, :, :] += gr_i[:, :, None] * weights[None, None, :]
                nens = weights.shape[0]

                gr_corr = (
                    -(gr_avg[:, :, None] / nframe) * (w_avg[None, None, :] / nframe)
                    + grw_avg / nframe
                )
                gr_reweight = gr_avg[:, :, None] / nframe + gr_corr  # / nens**0.5
                for i in range(weights.shape[0]):
                    np.savetxt(
                        outfile + f".ens{i}",
                        np.column_stack((centers, gr_reweight[:, :, i])),
                        header=header,
                    )

    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        if "fw" in locals():
            fw.close()


def main():
    parser = argparse.ArgumentParser(description="compute radial distribution function")
    parser.add_argument("xyz_file", type=str, help="arc file")
    parser.add_argument(
        "pairs",
        nargs="+",
        type=str,
        help="pairs of species to compute the RDF for (e.g. H-O O-O)",
    )
    parser.add_argument(
        "-t", "--thermalize", type=int, default=0, help="number of frames to skip"
    )
    parser.add_argument("-w", "--watch", action="store_true", help="watch file")
    parser.add_argument(
        "-c",
        "--nocomment",
        action="store_true",
        help="flag to indicate that the xyz file does not have a comment line",
    )
    parser.add_argument(
        "--cell",
        type=float,
        nargs="+",
        help="PBC cell. If 9 floats, it corresponds to the sequence of the three cell vectors. If 6 floats, it corresponds to the lengths and angles (in degrees) of the cell vectors. If 3 floats, it corresponds to the lengths of the cell vectors. If 1 float, it corresponds to the length of a cubic cell.",
    )

    parser.add_argument("-m", "--minimage", action="store_true", help="minimum image")
    parser.add_argument("--rmax", type=float, default=10.0, help="max radius")
    parser.add_argument("--dr", type=float, default=0.05, help="discretization")
    parser.add_argument(
        "--weightfile", "--wf", type=str, help="weight file", default=None
    )
    parser.add_argument(
        "--outfile",
        "-o",
        type=str,
        default="gr.dat",
        help="output file for the RDF data",
    )

    args = parser.parse_args()
    xyzfile = args.xyz_file
    pairs = args.pairs
    thermalize = args.thermalize
    watch = args.watch
    comment = not args.nocomment
    rmax = args.rmax
    dr = args.dr
    weightfile = args.weightfile
    outfile = args.outfile
    minimum_image = args.minimage

    cell = args.cell
    cell = parse_cell(cell) if cell is not None else None

    radial_from_file(
        xyzfile,
        pairs,
        outfile=outfile,
        cell=cell,
        rmax=rmax,
        dr=dr,
        has_comment_line=comment,
        watch=watch,
        thermalize=thermalize,
        minimum_image=minimum_image,
        weightfile=weightfile,
    )


if __name__ == "__main__":
    main()
