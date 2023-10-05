import os
import torch
from matplotlib import pyplot as plt
from learnable_wavelets import learnable_wavelets

def run_test():
    N_time = 81
    wave = learnable_wavelets(2, 4, N_time, mother_num_freqs=16)

    # Setting mother
    waveform = torch.sin(
        torch.pi/(wave.mother_num_freqs-1)*torch.arange(
            wave.mother_num_freqs)
    )**2
    mother = torch.concatenate(
        [waveform.unsqueeze(0), waveform.unsqueeze(0)**9],
        dim=0)
    wave.set_mother_spectrum(mother, mother)

    # Plotting
    if not os.path.exists("./tests/plots"):
        os.makedirs("./tests/plots")
    fig, axs = wave.plot_power_spectrums()
    fig.savefig("./tests/plots/test_power_spectrum.png")
    fig, axs = wave.plot_wavelets()
    fig.savefig("./tests/plots/test_wavelets.png")
