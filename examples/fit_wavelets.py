import sys
import glob
import h5py
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import rearrange

from learnable_wavelets import learnable_wavelets


parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_bases", type=int, default=1,
)
parser.add_argument(
    "--n_daughters", type=int, default=4,
)
parser.add_argument(
    "--stride", type=int, default=1,
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
parser.add_argument(
    "--data", type=str, default='gaussian_FT',
)
args = parser.parse_args()


def import_original(filename):
    with h5py.File(filename, 'r') as file:
        output = file['sol'][:]
    return output

def get_data(data_label, folder, run_label='', batch_size=64, eval_batch_size=512, num_workers=1, plot_example=False):
    if 'ks' in data_label.lower():
        train, valid, test = get_KS_data(folder, plot_example)
    else:
        train = np.load(os.path.join(folder, data_label+"_"+run_label, 'train.npy'))
        valid = np.load(os.path.join(folder, data_label+"_"+run_label, 'valid.npy'))
        test = np.load(os.path.join(folder, data_label+"_"+run_label, 'test.npy'))
    
    train_loader = DataLoader(
        train,
        shuffle=True,
        batch_size=batch_size,
        #num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid,
        shuffle=False,
        batch_size=eval_batch_size,
        #num_workers=num_workers
    )
    test_loader = DataLoader(
        test,
        shuffle=False,
        batch_size=eval_batch_size,
        #num_workers=num_workers
    )
    
    return train, train_loader, valid, valid_loader, test, test_loader
    

def get_KS_data(folder, plot_example):
    np_filename = 'full_data.npy'
    if not os.path.exists(os.path.join(folder, np_filename)) or plot_example:
        print("FOLDER", folder, len(glob.glob(os.path.join(folder, "*.jld2"))))
        data_files = glob.glob(os.path.join(folder, "*.jld2"))
        if plot_example:
            with h5py.File(data_files[0], 'r') as file:
                if plot_example:
                    #fig, ax = plt.subplots()
                    plt.imshow(file['sol'][:400,:100])
                    plt.savefig("test.png")
                    sys.exit()
        
        data = [import_original(fname) for fname in data_files]
        data = np.array(data)
        np.save(os.path.join(folder, np_filename), data)

    full_data = np.load(os.path.join(folder, np_filename))
    full_data = np.reshape(full_data, (-1, full_data.shape[-1]))
    n_samples = len(full_data)
    n_train = n_samples//2
    n_valid = n_samples//4
    train_data = full_data[:n_train]
    valid_data = full_data[n_train:n_train+n_valid]
    test_data = full_data[n_train+n_valid:]

    return train_data, valid_data, test_data






torch.autograd.set_detect_anomaly(True)
label = f"nB-{args.n_bases}_nD-{args.n_daughters}_L{args.wavelet_len}_nMF{args.n_mother_freqs}"
train_data, train_loader, valid_data, valid_loader, test_data, test_loader = get_data(
    args.data, './data', run_label=label
)
model = learnable_wavelets(
    args.n_bases,
    args.n_daughters,
    args.wavelet_len,
    mother_num_freqs=args.n_mother_freqs,
    stride=args.stride,
    padding='same',
    device='cpu'
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
model.train()
colors = ['k', 'b', 'r', 'g', 'y', 'teal', 'pink']
for epx in range(20):
    for ibtc, batch in enumerate(train_loader):
        #print("batch", batch.shape, torch.mean(batch), torch.std(batch))
        #batch = rearrange(batch, "b t l -> (b t) l")
        batch = batch.to(torch.float32)
        optimizer.zero_grad()
        coeffs = model(batch)
        results = model.invert_projection(coeffs)
        loss = torch.mean((results - batch.unsqueeze(1))**2)
        loss.backward(retain_graph=True)
        #print("gradient", model.mother_ft_real)
        #print("coeffs", coeffs)
        #print("LOSSSS", loss)
        optimizer.step()

        #print(batch.shape, coeffs.shape, results.shape)
        if ibtc == 0:
            fig, axs = model.plot_power_spectrums()
            fig.savefig(os.path.join(f"power_spectrum_{epx}.png"))
            plt.figure()
            print("SHAPES", batch.shape, results.shape)
            plt.plot(batch[0])
            plt.plot(results[0,0].detach())
            plt.savefig(f"test_{epx}.png")
            plt.close()
            print(loss.item()/torch.mean(batch**2).item())
            print(coeffs.shape)

            cplot = coeffs
            cplot = cplot.detach()
            fig, ax = plt.subplots()
            for i in range(model.num_daughters + 1):
                ax.plot(cplot[0,0,i,:], color=colors[i], label=f"coeff {i}")
            fig.legend()
            fig.savefig(f"coefficients_{epx}.png")
            print("COEFFS", coeffs.shape)


