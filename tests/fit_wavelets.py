import torch
import glob
import h5py
import os
from learnable_wavelets import learnable_wavelets


def get_data(folder):
    print("FOLDER", folder, len(glob.glob(os.path.join(folder, "*.jld2"))))
    for fname in glob.glob(os.path.join(folder, "*.jld2")):
        with h5py.File(fname, 'r') as file:
            print("KEYS", file.keys())
        asdf

get_data('./data')