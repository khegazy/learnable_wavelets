import os
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
from learnable_wavelets import learnable_wavelets


parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_bases", type=int, default=1,
)
parser.add_argument(
    "--n_daughters", type=int, default=4,
)
parser.add_argument(
    "--wavelet_len", type=int, default=81,
)
parser.add_argument(
    "--n_mother_freqs", type=int, default=16,
)
parser.add_argument(
    "--n_train_samples", type=int, default=1000,
)
args = parser.parse_args()

wavelets = learnable_wavelets(
    args.n_bases,
    args.n_daughters,
    args.wavelet_len,
    mother_num_freqs=args.n_mother_freqs
)

label = f"nB-{args.n_bases}_nD-{args.n_daughters}_L{args.wavelet_len}_nMF{args.n_mother_freqs}"

# Setting mother
waveform = torch.sin(
    torch.pi/(wavelets.mother_num_freqs-1)*torch.arange(
        wavelets.mother_num_freqs)
)**2
mother = waveform.unsqueeze(0)
#torch.concatenate(
#    [waveform.unsqueeze(0), waveform.unsqueeze(0)**9],
#    dim=0)
wavelets.set_mother_spectrum(mother, mother)
corrs = wavelets.get_wavelet_correlations()
wavelets.plot_correlations()
asdf

# Make data directory
data_dir = os.path.join("data", "gaussian_FT_"+label)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Plotting
fig, axs = wavelets.plot_power_spectrums()
fig.savefig(os.path.join(data_dir, "power_spectrum.png"))
fig, axs = wavelets.plot_wavelets()
fig.savefig(os.path.join(data_dir, "wavelets.png"))


n_time = 126
loop = zip(
    ['train', 'valid', 'test'],
    [args.n_train_samples, args.n_train_samples//4, args.n_train_samples/4]
)
idxs_base = np.arange(args.wavelet_len) - args.wavelet_len//2
waves = wavelets.get_wavelets().detach().numpy() 
for name, n_samples in loop:
    data = []
    for i in range(int(n_samples)):
        sample = np.zeros(n_time)
        coeffs = np.expand_dims(np.random.rand(*waves.shape[:-1]), -1)
        signal = coeffs*waves
        signal = np.sum(np.sum(signal, axis=0), axis=0)
        
        t_idx = np.random.randint(0, n_time)
        idxs = t_idx + idxs_base
        mask = (idxs >= 0)*(idxs < n_time)
        signal = signal[mask]
        idxs = idxs[mask]

        sample[idxs] = signal
        data.append(sample)
        if i%100 == 0:
            plt.figure()
            plt.plot(sample)
            plt.savefig(os.path.join(data_dir, f"{name}_sample_{i}.png"))
            plt.close()
    np.save(os.path.join(data_dir, f"{name}.npy"), np.array(data))

