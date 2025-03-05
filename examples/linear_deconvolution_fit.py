import torch
import numpy as np
from einops import rearrange
from learnable_wavelets import learnable_wavelets
from matplotlib import pyplot as plt

def linear_dconvolution_fit():
    
    N_time = 81
    wave = learnable_wavelets(1, 4, N_time, mother_num_freqs=16)

    # Setting mother
    waveform = torch.sin(
        torch.pi/(wave.mother_num_freqs-1)*torch.arange(
            wave.mother_num_freqs)
    )**2
    mother = waveform.unsqueeze(0)
    wave.set_mother_spectrum(mother, mother)

    wavelets = wave.get_wavelets()

    data = torch.sum(wavelets[0], dim=0).unsqueeze(0)
    plt.figure()
    plt.plot(data[0].detach())
    plt.savefig("test_data.png")
    plt.close()
    data = torch.nn.functional.pad(data, (N_time, N_time), 'constant', 0)
    print(data.shape)
    coeffs, deconv_coeffs = wave(data)
    print("C SHAPE", coeffs.shape)
    fig, axs = plt.subplots(wave.num_daughters+1, 1)
    for iax, ax in enumerate(axs):
        ax.plot(coeffs[0,0,iax,:].detach())
    fig.savefig("linfit_coeffs.png")
    fig, axs = plt.subplots(wave.num_daughters+1, 1)
    for iax, ax in enumerate(axs):
        ax.plot(deconv_coeffs[0,0,iax,:].detach())
    fig.savefig("linfit_deconv_coeffs.png")

    N = wave.num_daughters+1
    sum_corrs = torch.sum(wave.correlations[0], dim=1)
    """
    sum_corrs = wave.correlations[0,N-1,N-1]
    for i in range(N-1):
        for j in range(i, N-1):
            sum_corrs = sum_corrs + wave.correlations[0,i,j]
    """
    #fig, ax = plt.subplots()
    #ax.plot(sum_corrs.detach())
    fig, axs = plt.subplots(wave.num_daughters+1, 1)
    for iax, ax in enumerate(axs):
        ax.plot(sum_corrs[iax,:].detach())
    fig.savefig("linfit_added_corrs.png")
    adsf



    # Linear Kernel: [num_bases, len*(num_daughters+1), len*(num_daughters+1)]
    linear_kernel, linear_norm = wave._build_linear_kernel(N_time)
    Y = torch.zeros((wave.num_daughters+1, wave.num_bases, wave.num_daughters+1, N_time))
    for d in range(wave.num_daughters+1):
        Y[d,:,d,N_time//2] = 1

    print(linear_kernel.shape, Y.shape, rearrange(Y, 'N nb c l -> N nb (l c)').shape) 
    # Correlations: [num_bases, len*(num_daughters+1), len*(num_daughters+1)]
    correlations = torch.einsum(
        'NAa,SNa->SNA',
        linear_kernel,
        rearrange(Y, 'N Nb c l -> N Nb (l c)')
    )
    correlations = rearrange(correlations, 'N nb (L B) -> N nb B L', B=wave.num_daughters+1)
    idx = torch.arange(wave.num_daughters+1)
    correlations_truth = wave.correlations[:,idx,idx,:]
    fig = plt.figure()
    plt.plot(wave.correlations[0,0,0].detach())
    plt.savefig("expected.png")
    del fig
    for i in np.arange(N_time//4)*4:
        fig = plt.figure()
        plt.plot(linear_kernel[0,:N_time,i*4].detach())
        plt.savefig(f"linK_{i}.png")
        del fig



    print("CORR SHAPE", correlations.shape, correlations_truth.shape)
    assert np.allclose(correlations_truth.detach(), correlations.detach())

linear_dconvolution_fit()
                  