#!/usr/bin/env python3

# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from math import ceil
from pathlib import Path

from progressbar import ProgressBar

import numpy as np

import soundfile as sf


class EmptySample(Exception):
    pass


def make_dataset(args, unk_args):
    import h5py
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("================================")
        print("Parameters")
        print("--------------------------------")
        for key, val in args.__dict__.items():
            print(f"{key:20s}: {val}")
        print("================================")

    # obtain file list
    if rank == 0:
        mix_paths = list(args.data_dir.rglob("mix.wav"))
        mix_paths += (ceil(len(mix_paths) / size) * size - len(mix_paths)) * [None]
    else:
        mix_paths = None
    mix_paths = comm.bcast(mix_paths, root=0)

    hdf_name = args.hdf5_output_path
    with h5py.File(hdf_name, "w", driver="mpio", comm=comm) as f:
        pbar = ProgressBar(redirect_stdout=True) if rank == 0 else lambda x: x
        for idx, mix_path in enumerate(pbar(mix_paths[rank::size])):
            # load spectrogram
            if mix_path is None:
                raise EmptySample()

            wav, sr = sf.read(mix_path, always_2d=True)
            duration, n_mic = wav.shape

            ref = []
            for inst in args.source_name:
                ref_path = mix_path.parent / f"{inst}.wav"
                ref_inst, sr = sf.read(ref_path, always_2d=True)
                ref.append(ref_inst)
            ref = np.stack(ref, axis=0)  # [4src, 2ch, n_samples]

            grp_names = []
            grp_names.append(f"{size*idx + rank:08d}")

            # initialize datasets
            all_grp_names = sum(comm.allgather(grp_names), [])
            for grp_name in all_grp_names:
                g = f.create_group(grp_name)
                g.create_dataset("wav", [n_mic, duration], "float32")
                g.create_dataset("ref", [len(args.source_name), n_mic, duration], "float32")

                f[f"{grp_name}/wav"][:] = wav.T
                f[f"{grp_name}/ref"][:] = np.swapaxes(ref, -1, -2)


def main():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--hdf5_output_path", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=["tr", "cv", "tt"], required=True)
    parser.add_argument("--source_name", type=list, default=["speech", "music", "sfx"])
    parser.add_argument("--mono", action="store_true")

    args, unk_args = parser.parse_known_args()
    make_dataset(args, unk_args)


if __name__ == "__main__":
    main()
