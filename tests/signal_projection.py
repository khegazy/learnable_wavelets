import torch
import numpy as np
from matplotlib import pyplot as plt
from learnable_wavelets import learnable_wavelets

time_len = 81
num_basis_sets = 2
num_daughters = 4
num_wavelets = num_daughters + 1
wave = learnable_wavelets(
    num_basis_sets,
    num_daughters, 
    time_len, 
    mother_num_freqs=16)

# Setting mother
waveform = torch.sin(
    torch.pi/(wave.mother_num_freqs-1)*torch.arange(
        wave.mother_num_freqs)
)**2
mother = torch.concatenate(
    [waveform.unsqueeze(0), waveform.unsqueeze(0)**9],
    dim=0)
wave.set_mother_spectrum(mother, mother)

# Getting wavelets
wavelets = wave.get_wavelets()

# Use wavelets as data to showcase orthogonality
data = wavelets.reshape((-1, time_len))
data = torch.concat(
    [torch.zeros(data.shape[0], 30), 
	data,
	torch.zeros(data.shape[0], 30)],
    dim=-1)
result = wave(data)

# Plotting
X,Y = np.meshgrid(
    np.arange(result.shape[-1]+1) - result.shape[-1]//2 + 1,
    np.arange(data.shape[0]+1))
result = np.reshape(
    result.numpy(),
    (data.shape[0], data.shape[0], result.shape[-1]))

for idx_d in range(data.shape[0]):
    fig, axs = plt.subplots(2, 1, figsize=(10,5), sharex=True)
    axs[0].plot(
        np.arange(data.shape[1]) - data.shape[1]//2 + 2,
        data[idx_d]
    )
    axs[1].pcolormesh(X, Y, result[idx_d])
    axs[1].set_xlabel("Time [bins]", fontsize=13)
    axs[1].set_ylabel("Wavelet", fontsize=13)
    basis_set = idx_d//num_daughters
    basis = idx_d%num_daughters
    fig.savefig(F"projection_wavelet_{basis_set}_{basis}.png")
