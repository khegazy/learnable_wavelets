import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from einops import rearrange


class learnable_wavelets(nn.Module):

    def __init__(self,
                 num_bases: int,
                 num_daughters: int,
                 length: int,
                 padding: str = 'valid',
                 stride: int = 1,
                 mother_num_freqs: int = None,
                 init_sine2=False,
                 device=None,
                 dtype=torch.float32) -> None:
        """
        Randomly initialize the mother wavelet spectra with random and
        calculate the initial daughter wavelets. The daughter wavelets
        are calculated after every backpropogation to reflect the 
        changes of the updated mother.
        
        Input:
            num_bases: Number of wavelet basis sets to create
            num_daughters: Number of daughters for each basis set
            length: Seqence length of the data that is projected upon
            padding: 'valid' for no padding or 'same' to add padding
            mother_num_freqs: Number of mother Fourier components
            device: The device this class will be evaluated on
            dtype: Type of the tensor elements
        """
        super().__init__()
        self.factory_kwargs = {'device' : device, 'dtype' : dtype}
        self.num_bases = num_bases
        self.num_daughters = int(num_daughters)
        self.length = length
        self.N_freqs = self.length//2 + 1
        self.freqs = torch.arange(self.N_freqs)*2*torch.pi/self.length
        self.stride = stride
        self.padding = padding
        self.left_padding = (self.length - self.stride)//2
        self.time = torch.arange(self.length) - self.length//2 + 1
        self.recalculate_wavelets = True
        self.wavelets = torch.empty((self.num_bases, self.num_daughters+1, self.length), requires_grad=True)
        self.correlations = None
        self.linear_kernel_length = 0

        if mother_num_freqs:
            self.mother_num_freqs = mother_num_freqs
        else:
            self.mother_num_freqs = length//4
            print("Find proper calculation ", self.mother_num_freqs)
        if self.stride > 1 and self.padding.lower() == 'same':
            print("WARNING: Padding cannot be 'same' with stride > 1, will add padding")
            self.padding = 'valid'
        self.mother_num_freqs -= self.mother_num_freqs%2
        if self.mother_num_freqs > self.length//2 + 1:
            raise ValueError("""mother_num_freqs cannot be larger 
                than length//2 + 1 (the maximum number of orthogonal 
                Fourier components)""")
        print(self.mother_num_freqs, 2**self.num_daughters)
        if self.mother_num_freqs < 2**(self.num_daughters):
            raise ValueError("""mother_num_freqs must be >= 
                2**num_daughters, otherwise the mother wavelet cannot 
                be divided num_daughter times""")
        if self.mother_num_freqs%(2**(self.num_daughters)) != 0:
            raise ValueError("""mother_num_freqs must be divisible 
                by 2**num_daughters""")               
        
        # Create mother wavelet
        self.mother_ft_real = nn.Parameter(
            torch.empty(
                (self.num_bases, self.mother_num_freqs),
                requires_grad=True,
                **self.factory_kwargs))
        self.mother_ft_imag = nn.Parameter(torch.empty(
            (self.num_bases, self.mother_num_freqs),
            requires_grad=True,
            **self.factory_kwargs))
        if init_sine2:
            self._init_mother_sin2()
        else:
            torch.nn.init.kaiming_uniform_(
                self.mother_ft_real,
                a=np.sqrt(5))
            torch.nn.init.kaiming_uniform_(
                self.mother_ft_imag,
                a=np.sqrt(5))
        print("Check if parameter is declared before or after init")

        # Calculate daughter wavelets
        idx_offset = (self.length//2 + 1) - self.mother_num_freqs
        self.freq_idxs = [
            torch.arange(self.mother_num_freqs//(2**(i+1)))
            + idx_offset//(2**(i+1)) + 1
                for i in range(self.num_daughters)
        ]
        self.freq_idxs =\
            [torch.arange(self.mother_num_freqs) + idx_offset]\
            + self.freq_idxs
        self.wavelets_ft_real = None
        self.wavelets_ft_imag = None
        self.build_daughters_ft()

        # Recalculate daughter wavelets after backprop
        #self.back_hook_handle = self.register_full_backward_hook(
        #    self.build_daughters_ft)

    def _init_mother_sin2(self):
        waveform = torch.sin(
            torch.pi/(self.mother_num_freqs-1)*torch.arange(
            self.mother_num_freqs)
        ).unsqueeze(0)

        self.set_mother_spectrum(waveform, waveform, calculate_daughters=False, calculate_wavelets=False)
        
    def build_daughters_ft(self) -> None:
        """
        Calculate Fourier spectra of daughter wavelets given the 
        learned mother Fourier spectrum. For daughter i, the mother 
        frequency component is changed by 2**(-i).
        """
        # Reset wavelets
        #self.wavelets = None
        self.recalculate_wavelets = True
        self.correlations = None
        # Set mother wavelet and daughter fourier spectras
        self.wavelets_ft_real = torch.zeros(
            (self.num_bases, self.num_daughters+1, self.N_freqs),
            **self.factory_kwargs)
        self.wavelets_ft_imag = torch.zeros(
            (self.num_bases, self.num_daughters+1, self.N_freqs),
            **self.factory_kwargs)
        norm = self.get_wavelet_norm_ft_(
                self.mother_ft_real,
                self.mother_ft_imag)
        self.wavelets_ft_real[:,0,self.freq_idxs[0]] = self.mother_ft_real/norm 
        self.wavelets_ft_imag[:,0,self.freq_idxs[0]] = self.mother_ft_imag/norm
        for idx_d,freq_daughter_idxs in enumerate(self.freq_idxs[1:]):
            self.wavelets_ft_real[:,idx_d+1,freq_daughter_idxs] =\
                torch.sum(
                    torch.reshape(
                        self.mother_ft_real,
                        (-1,freq_daughter_idxs.shape[0],2**(idx_d+1))),
                    dim=-1)
            self.wavelets_ft_imag[:,idx_d+1,freq_daughter_idxs] =\
                torch.sum(
                    torch.reshape(
                        self.mother_ft_imag,
                        (-1,freq_daughter_idxs.shape[0],2**(idx_d+1))),
                dim=-1)
            norm = self.get_wavelet_norm_ft_(
                self.wavelets_ft_real[:,idx_d+1,freq_daughter_idxs],
                self.wavelets_ft_imag[:,idx_d+1,freq_daughter_idxs])
            self.wavelets_ft_real[:,idx_d+1,freq_daughter_idxs] /= norm
            self.wavelets_ft_imag[:,idx_d+1,freq_daughter_idxs] /= norm
        
        # Wavelet amissability requires DC component is 0
        self.wavelets_ft_real[:,0,0] = 0
        self.wavelets_ft_imag[:,0,0] = 0

    
    def get_wavelet_norm_ft_(self, 
                             real_part: torch.FloatTensor, 
                             imag_part: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculates a normalization factor so wavelets are square 
        normalized.
        
        Input:
            reals_part: Real part of the wavelet spectrum
            imag_part: Imaginary part of the wavelet spectrum
        Return:
            norm torch.FloatTensor : normalization factor
        """
        return torch.sqrt(
            (2*self.length)*torch.sum(
                real_part**2 + imag_part**2,
                dim=-1,
                keepdims=True))
    
    
    def set_mother_spectrum(self, 
                            mother_real: torch.FloatTensor,
                            mother_imag: torch.FloatTensor,
                            calculate_daughters: bool = True,
                            calculate_wavelets: bool = True) -> None:
        """
        Set the mother Fourier spectrum and recalculate
        
        Inputs:
            mother_real: Real part of mother Fourier spectrum
            mother_imag: Imaginary part of mother Fourier spectrum
                calculate_daughters: Recalculate daughters after changing
                mother
        """
        print("SIZES", mother_real.shape, self.mother_ft_real.shape)
        state_dict = self.state_dict()
        state_dict['mother_ft_real'] = mother_real
        state_dict['mother_ft_imag'] = mother_imag
        self.load_state_dict(state_dict)
        #self.mother_ft_real = mother_real
        #self.mother_ft_imag = mother_imag
        if calculate_daughters or calculate_wavelets:
            self.build_daughters_ft()
        if calculate_wavelets:
            _ = self.get_wavelets()
        
        
    def get_fourier_spectrum(self,
                             full_spectra: bool = False
                             ):# -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Returns the Fourier spectra for the mother and daughter
        wavelets.
        
        Input:
            full_spectra: If True return positive and negative 
                freqency components otherwise only the positive
        Returns:
            fourier_spect: The complex Fourier spectra
            freqs: The corresponding freqencies
        """
        fourier_spect = torch.complex(
            self.wavelets_ft_real,
            self.wavelets_ft_imag)
        freqs = self.freqs
        if full_spectra:
            fourier_spect = torch.concatenate(
                [torch.conj(fourier_spect.flip(dims=[-1])), 
                    fourier_spect[:,:,1:]],
                dim=-1
            )
            freqs = torch.concatenate(
                [freqs.flip(dims=[-1]), freqs[1:]])
        
        return fourier_spect, freqs

    
    def plot_wavelet_spectrums_(self,
                                ax: plt.axes,
                                fourier_spect: torch.FloatTensor,
                                freqs: torch.FloatTensor,
                                params: dict) -> None:
        """
        Plot the wavelet Fourier spectrums for a single basis set
        
        Input:
            ax: a single Matplotlib plot axis to plot wavelet
                spectrums on
            fourier_spect: The wavelet Fourier spectrum for a single 
                basis
            freqs: The frequencies for the provided fourier_spect
            params: A dictionary of plotting parameters
        """
        colors = [
            'k', 'b', 'r', 'g', 'orange', 'purple',
            'teal', 'navy', 'y', 'pink'
        ]
        num_colors = len(colors)
        for idx_d in range(self.num_daughters+1):
            color = colors[idx_d%num_colors]
            ax.plot(
                freqs, 
                np.real(fourier_spect[idx_d]),
                color=color,
                linestyle='--')
            ax.plot(
                freqs,
                np.imag(fourier_spect[idx_d]),
                color=color,
                linestyle=':')
            ax.plot(
                freqs,
                np.abs(fourier_spect[idx_d]),
                color=color,
                linestyle='-')
            ax.set_xlim(freqs[0], freqs[-1])
        ax.set_xlabel(params['xlabel'], fontsize=params['label_size'])
        
    
    def plot_power_spectrums(self, fig_size: tuple = (13,5)):# -> tuple[plt.figure, plt.axes]:
        """
        Plot the power spectrums for all wavelet bases
        
        Input:
            fig_size: Figure size
        Returns:
            fig: Matplotlib figure that is plotted
            axs: List of Matplotlib axes that are on the figure
        """
        fourier_spect, freqs = self.get_fourier_spectrum()
        fourier_spect = fourier_spect.detach().numpy()
        fig_params = {
            'xlabel' : r'Frequency [rad]',
            'label_size' : 13
        }
        
        if self.num_bases == 1:
            fig, axs = plt.subplots(figsize=fig_size)
            self.plot_wavelet_spectrums_(
                axs,
                fourier_spect[0],
                freqs,
                fig_params)
        else:
            fig, axs = plt.subplots(
                int(np.ceil(float(self.num_bases)/2)),
                2,
                figsize=fig_size)
            if self.num_bases == 2:
                axs = [axs]
            for idx_b in range(self.num_bases):
                self.plot_wavelet_spectrums_(
                    axs[idx_b//2][idx_b%2],
                    fourier_spect[idx_b],
                    freqs,
                    fig_params)
        
        return fig, axs
    
    
    def get_wavelets(self) -> torch.FloatTensor:
        """
        Calculate the wavelets in the time domain
        
        Returns:
            Wavelets: Time domain wavelet signals
        """
        if self.recalculate_wavelets:
            freq_time = self.freqs.unsqueeze(1)*self.time.unsqueeze(0)
            cosines = torch.cos(freq_time)
            cosines = 2*self.wavelets_ft_real.unsqueeze(-1)*cosines
            sines = torch.sin(freq_time)
            sines = 2*self.wavelets_ft_imag.unsqueeze(-1)*sines
            self.wavelets = torch.sum(cosines+sines, dim=-2)
            self.recalculate_wavelets = False
            self.calculate_correlations()
            #print("WAVELET CALC MAX MIN", torch.amax(self.wavelets_ft_real), torch.amin(self.wavelets_ft_real))
            #print("WAVELET CALC MAX MIN", torch.amax(self.wavelets_ft_imag), torch.amin(self.wavelets_ft_imag))
        
        return self.wavelets
    
    
    def plot_wavelets_(self, axs: list, wavelets: torch.FloatTensor) -> None:
        """
        Plot wavelets in the time domain for a single basis set
        
        Input:
            axs: List of matplotlib.pyplot.axes for each wavelet in
                a single basis set.
            wavelets: The wavelets for a single wavelet basis 
        """
        ymin = np.amin(wavelets)
        ymax = np.amax(wavelets)
        for idx_d in range(self.num_daughters+1):
            axs[idx_d].plot(self.time, wavelets[idx_d], 'k')
            axs[idx_d].set_xlim(self.time[0], self.time[-1])
            axs[idx_d].set_ylim(ymin, ymax)
            if idx_d > 0:
                axs[idx_d].get_yaxis().set_visible(False)
    
    
    def plot_wavelets(self, 
                      fig_size: tuple = (10,5), 
                      label_size: int = 13):# -> tuple[plt.figure, plt.axes]:
        """
        Plot all the wavelet bases in the time domain
        
        Input:
            fig_size: Figure size
            label_size: Font size of labels
        Returns:
            fig: Matplotlib figure that is plotted
            axs: List of Matplotlib axes that are on the figure
        """
        self.get_wavelets()
        
        fig, axs = plt.subplots(
            self.num_bases, self.num_daughters+1,
            figsize=fig_size, sharex=True)
        fig_params = {
            'ylabel' : 'Wavelets',
            'xlabel' : 'Time [bin]',
            'label_size' : 13
        }
        
        if self.num_bases == 1:
            self.plot_wavelets_(axs, self.wavelets[0].detach().numpy())
            self.beautify_wavelet_plots_(
                axs,
                fig_params,
                is_bottom=True)
        else:
            for idx_b in range(self.num_bases):
                self.plot_wavelets_(
                    axs[idx_b],
                    self.wavelets[idx_b].numpy())
                self.beautify_wavelet_plots_(
                    axs[idx_b],
                    fig_params,
                    is_bottom=(idx_b==self.num_bases-1))
        
        plt.tight_layout()
        return fig, axs
        
    
    def beautify_wavelet_plots_(self,
                                axs: list,
                                params: dict,
                                is_bottom: bool = False) -> None:
        """
        Sets the labels of the wavelet time domain plots (axes) in axs
        
        Input:
            axs: List of Matplotlib axes that are plotted upon
            params: Dictionary of label titles and other plot
                parameters
            is_bottom: Is this list of axes at the bottom
        """
        if 'label_size' not in params:
            params['label_size'] = 13
        if 'ylabel' in params:
            axs[0].set_ylabel(
                params['ylabel'],
                fontsize=params['label_size'])
        if is_bottom:
            for idx_d in range(self.num_daughters+1):
                if 'xlabel' in params:
                    axs[idx_d].set_xlabel(
                        params['xlabel'],
                        fontsize=params['label_size'])       

    def get_wavelet_correlations(self):
        if self.recalculate_wavelets:
            self.get_wavelets()
        return self.correlations

    def calculate_correlations(self):
        wavelets = self.get_wavelets()
        B, W, L = wavelets.shape
        self.correlations = torch.zeros((B, W, W, 2*L+1), dtype=torch.float64)
        print("WAVE SHAPE", wavelets.shape)
        padded_wavelets = F.pad(
            wavelets, (self.length, self.length), 'constant', 0
        ).to(torch.float64)
        print("PADDED", wavelets.shape, padded_wavelets.shape)
        for ibs in range(B):
            for iwv0 in range(W):
                for iwv1 in range(iwv0, W):
                    self.correlations[ibs,iwv0,iwv1] = F.conv1d(
                        padded_wavelets[None,None,ibs,iwv0],
                        wavelets[None,None,ibs,iwv1].to(torch.float64),#view(-1, 1, self.length),
                        bias=None,
                        stride=self.stride,
                        padding='valid'
                    )
                    self.correlations[ibs,iwv1,iwv0] = self.correlations[ibs,iwv0,iwv1]
                    
                    #.reshape(
                    
                    #        (input.shape[0],
                    #            self.num_bases,
                    #            self.num_daughters+1,
                    #            -1))
                    #print("CORR SHAPE", corr.shape)
                    #self.correlations[ibs,iwv0,iwv1] = corr[0,0]
                    fig, ax = plt.subplots()
                    ax.plot(self.correlations[ibs,iwv0,iwv1].detach())
                    fig.savefig(f"corr_{iwv0}_{iwv1}.png")
        
        return self.correlations


    def get_linear_kernel(self, length):
        if self.recalculate_wavelets:
            self.get_wavelets()
        elif self.linear_kernel_length == length:
            return self.linear_kernel, self.linear_norm
        
        return self._build_linear_kernel(length)
     
    def _build_linear_kernel(self, length):
        #TODO: Consider sidebands from regions outside of the window appearing. Solution, do wavelet projection starting with no overlap
        wavelets = self.get_wavelets()
        strided_length = length//self.stride
        #padding_len = int((corr_length//self.stride)//2)
        self.linear_kernel = torch.zeros(
            (self.num_bases, self.num_daughters+1, strided_length, strided_length),
            dtype=torch.float64
        )
        L = wavelets.shape[-1]
        print("SHAPES", wavelets.shape, self.linear_kernel.shape)
        for l in range(strided_length):
            l_idx = max(0, l-L//2)
            r_idx = L//2 + l + (L % 2)
            r_idx = min(strided_length, r_idx)
            lc_idx = max(0, L//2-l)
            rc_idx = lc_idx+(r_idx-l_idx)
            print("idxs", l_idx, r_idx, lc_idx, rc_idx)
            print(self.linear_kernel[:,:,l,l_idx:r_idx].shape, wavelets[:,:,lc_idx:rc_idx].unsqueeze(2).shape, wavelets.shape)
            self.linear_kernel[:,:,l,l_idx:r_idx] = wavelets[:,:,lc_idx:rc_idx]
        self.linear_kernel = torch.permute(self.linear_kernel, (0, 3, 1, 2))
        """
        fig = plt.figure()
        plt.plot(self.correlations[0,:,-1,0].detach())
        plt.savefig("expected.png")
        del fig
        for i in np.arange(strided_length//10)*10:
            fig = plt.figure()
            #plt.plot(self.linear_kernel[0,:self.length,i*4].detach())
            plt.plot(self.linear_kernel[0,:,-1,i].detach())
            plt.savefig(f"linK_{i}.png")
            del fig
        """

        self.linear_kernel = rearrange(self.linear_kernel, "N L b l -> N L (l b)")

        return self.linear_kernel

    def linear_fit(self, input):
        print("INPUT SHAPE", input.shape)
        wavelets = self.get_wavelets()
        linear_kernel = self._build_linear_kernel(input.shape[-1]).to(torch.float64)
        U, S, Vh = torch.linalg.svd(linear_kernel, full_matrices=False)
        print("SINGULAR VALS", S[0])
        print("SVD SHAPES", U.shape, S.shape, Vh.shape, input.shape, linear_kernel.shape)
        S = S + 1e-5
        S_inv = 1./S.unsqueeze(1)
        S_inv[torch.abs(S_inv)>1e8] = 0
        fit = torch.einsum(
            "NAa,Nba->NbA",
            Vh.transpose(-1,-2),
            S_inv*torch.einsum(
                "NAa,NA->Na", U, input.to(torch.float64)
            )
        )
        return rearrange(fit, 'N b (l c) -> N b c l', l=input.shape[-1])
        return fit




    def _build_linear_kernel_deconv(self, length):
        #TODO: Consider sidebands from regions outside of the window appearing. Solution, do wavelet projection starting with no overlap
        corr_length = self.correlations.shape[-1]
        strided_length = length//self.stride
        #padding_len = int((corr_length//self.stride)//2)
        self.linear_kernel = torch.zeros(
            (self.num_bases, strided_length, self.num_daughters+1, self.num_daughters+1, strided_length),
            dtype=torch.float64
        )
        L = self.correlations.shape[-1]
        print("SHAPES", self.correlations.shape, self.linear_kernel.shape)
        for l in range(strided_length):
            l_idx = max(0, l-L//2)
            r_idx = L//2 + l + (L % 2)
            r_idx = min(strided_length, r_idx)
            lc_idx = max(0, L//2-l)
            rc_idx = lc_idx+(r_idx-l_idx)
            #print("idxs", l_idx, r_idx, lc_idx, rc_idx)
            self.linear_kernel[:,l,:,:,l_idx:r_idx] = self.correlations[:,:,:,lc_idx:rc_idx]
        self.linear_kernel = torch.permute(self.linear_kernel, (0, 4, 2, 1, 3))
        print("KERNEL SHAPE", self.linear_kernel.shape, self.length)
        fig = plt.figure()
        plt.plot(self.correlations[0,0,0].detach())
        plt.savefig("expected.png")
        del fig
        """
        for i in np.arange(self.length//4)*4:
            fig = plt.figure()
            #plt.plot(self.linear_kernel[0,:self.length,i*4].detach())
            plt.plot(self.linear_kernel[0,:,0,i,0].detach())
            plt.savefig(f"linK_{i}.png")
            del fig
        """

        """
        self.linear_kernel = rearrange(self.linear_kernel, 'Nb L B l b -> Nb L B (l b)') #torch.flatten(self.linear_kernel, start_dim=-2, end_dim=-1)
        for ib in [0, 1]:
            fig = plt.figure()
            plt.plot(self.correlations[0,0,ib].detach())
            plt.savefig(f"expected_{ib}.png")
            del fig
            #for i in range(30):
            for i in np.arange(self.length//4)*4:
                fig = plt.figure()
                #plt.plot(self.linear_kernel[0,:,0,i].detach())
                plt.plot(self.linear_kernel[0,:,0,i*(self.num_daughters+1)+ib].detach())
                #plt.plot(self.linear_kernel[0,ib*self.length:(ib+1)*self.length,ib*self.length+i*4].detach())
                plt.savefig(f"lhinK_{ib}_{i}.png")
                del fig

        sdfgsdf
        """
        
        self.linear_kernel = rearrange(self.linear_kernel, 'Nb L B l b -> Nb (L B) (l b)') #torch.flatten(self.linear_kernel, start_dim=-2, end_dim=-1)
        
        
        """
        for ib in [0, 1]:
            fig = plt.figure()
            plt.plot(self.correlations[0,0,ib].detach())
            plt.savefig(f"expected_{ib}.png")
            del fig
            #for i in range(30):
            for i in np.arange(self.length//4)*4:
                fig = plt.figure()
                #plt.plot(self.linear_kernel[0,:,0,i*(self.num_daughters+1)+ib].detach())
                len_idx = torch.arange(strided_length)*(self.num_daughters+1)
                plt.plot(self.linear_kernel[0,len_idx,i*(self.num_daughters+1)+ib].detach())
                plt.savefig(f"linK_{ib}_{i}.png")
                del fig

        sdfgsdf
        """

        #self.linear_kernel = torch.flatten(self.linear_kernel, start_dim=1, end_dim=2)
        eps = 0
        corr = torch.matmul(self.linear_kernel.transpose(-1, -2), self.linear_kernel) + torch.eye(self.linear_kernel.shape[-1])*eps
        print("EIGSV", torch.sort(torch.abs(torch.linalg.eigvals(corr)))[0][0])
        print("DETERM", torch.det(corr)*1e10)
        print("LARGEST CORR VALS", torch.sort(torch.abs(torch.flatten(corr)))[:10])
        self.linear_norm = torch.linalg.inv(corr)
        print("LARGEST INV VALS", torch.sort(torch.abs(torch.flatten(self.linear_norm)))[:10])
        return self.linear_kernel, self.linear_norm
        padded_corrs = F.pad(
            self.correlations, (padding_len, padding_len), 'constant', 0
        )
        print("PADDING SHAPE", self.length, self.correlations.shape, padded_corrs.shape)
        print("TODO: FIGURE OUT PADDING FOR STRIDES")
        self.linear_kernel = []
        for ibs in range(self.num_bases):
            self.linear_kernel.append(
                torch.stack(
                    [
                        padded_corrs[ibs,:,:,l:l+strided_length]\
                            for l in range(strided_length)
                    ]
                )
            )
        print("asd", len(self.linear_kernel), self.linear_kernel[0].shape)
        self.linear_kernel = torch.tensor(self.linear_kernel)
        print("LINEAR KERNEL SHAPE", self.linear_kernel.shape)
        return self.linear_kernel


    def linear_fit_deconvolution(self, coefficients):
        if self.linear_norm is None:
            raise RuntimeError("Using the wrong linear kernel")
        print("COEFFS", coefficients.shape)
        N, nb, C, L = coefficients.shape
        coeffs = coefficients.transpose(-2, -1)
        coeffs = rearrange(
            coefficients, 'N nb l b -> N nb (l b)'
        ).to(torch.float64)
        linear_kernel, linear_norm = self.get_linear_kernel(L)
        U, S, Vh = torch.linalg.svd(linear_kernel, full_matrices=False)
        print("SINGULAR VALS", S[0])
        print("SVD SHAPES", U.shape, S.shape, Vh.shape, coeffs.shape)
        S = S + 1e-5
        S_inv = 1./S.unsqueeze(1)
        S_inv[torch.abs(S_inv)>1e8] = 0
        fit = torch.einsum(
            "NAa,Nba->NbA",
            Vh,
            S_inv*torch.einsum(
                "NAa,NbA->Nba", U, coeffs
            )
        )
        #inverse = torch.matmul(Vh.transpose(-1,-2), S_inv*U.transpose(-1,-2))
        #fit = torch.einsum('baA,NbA->Nba', inverse, coeffs.to(torch.float64))
        #dsf
        return rearrange(fit, 'N nb (l c) -> N nb c l', l=L)
        print("INVERSE", inverse.shape)
        asdf
        print("transpose", coeffs.shape)
        #coeffs = torch.flatten(coefficients.transpose(-1,-2), start_dim=-2, end_dim=-1)
        print("asdfa", coeffs.shape, linear_kernel.shape, linear_norm.shape)
        numer = torch.einsum('bAa,NbA->Nba', linear_kernel, coeffs)
        #numer = torch.sum(linear_kernel.transpose(-1, -2)*coeffs.unsqueeze(-2), dim=-1)
        print("NUMER", numer.shape, linear_norm.shape)
        fit = torch.einsum('Nba,boa->Nbo', numer, linear_norm)
        #fit = torch.sum(linear_norm*numer.unsqueeze(-2), dim=-1)
        return rearrange(fit, 'b nb (l c) -> b nb c l', l=L)


    def _linear_fit_deconvolution(self, coefficients):
        print("COEFFS", coefficients.shape)
        B, _, C, L = coefficients.shape
        linear_kernel, linear_norm = self.get_linear_kernel(L)
        coeffs = coefficients.transpose(-2, -1)
        print("transpose", coeffs.shape)
        coeffs = rearrange(coefficients, 'N nb l b -> N nb (l b)')
        #coeffs = torch.flatten(coefficients.transpose(-1,-2), start_dim=-2, end_dim=-1)
        print("asdfa", coeffs.shape, linear_kernel.shape, linear_norm.shape)
        numer = torch.einsum('bAa,NbA->Nba', linear_kernel, coeffs)
        #numer = torch.sum(linear_kernel.transpose(-1, -2)*coeffs.unsqueeze(-2), dim=-1)
        print("NUMER", numer.shape, linear_norm.shape)
        fit = torch.einsum('Nba,boa->Nbo', numer, linear_norm)
        #fit = torch.sum(linear_norm*numer.unsqueeze(-2), dim=-1)
        return rearrange(fit, 'b nb (l c) -> b nb c l', l=L)



    def plot_correlations(self):
        wavelets = self.get_wavelets().detach().numpy()
        correlations = self.get_wavelet_correlations().detach().numpy()
        B, W, _ = wavelets.shape
        for ibs in range(B):
            for iwv in range(W):
                fig, axs = plt.subplots(W+1, 1)
                for iax, ax in enumerate(axs):
                    if iax == 0:
                        ax.plot(wavelets[ibs,iwv], '-k')
                        ax.xaxis.set_visible(False)
                        continue
                    ax.plot(correlations[ibs,iwv,iax-1])
                    if iax-1 == iwv:
                        ax.set_ylim(-1.05, 1.05)
                    else:
                        ax.set_ylim(-0.1, 0.1)
                    if iax != W:
                        ax.xaxis.set_visible(False)
                    else:
                        ax.set_xlabel("Time [steps]")
                plt.tight_layout()
                fig.savefig(f"correlations_B{ibs}_D{iwv}.png")

    



    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        """
        Convolve the wavelet bases on the last dimension of the input
        tensor
        
        Inputs:
            input: Tensor to be projected onto the wavelet bases
        Returns:
            coefficients: The projection coefficients after convolving 
                the wavelet bases with the input signal
        """
        #print("start", self.wavelets.shape)
        print("DATA SHAPE", input.shape)
        if self.training:
            self.build_daughters_ft()
        #print("after build ft", self.wavelets.shape)
        # Add padding when stride > 1
        if self.stride > 1:
            self.prev_input_len = input.shape[1]
            B = input.shape[0]
            dims = input.shape[2:]
            input = torch.concatenate(
                [
                    torch.zeros((B, self.left_padding, *dims)),
                    input,
                    torch.zeros((B, self.length-self.left_padding, *dims))
                ],
                dim=1
            )

        coeffs = nn.functional.conv1d(
            input.unsqueeze(-2),
            self.get_wavelets().view(-1, 1, self.length),
            bias=None,
            stride=self.stride,
            padding=self.padding).reshape(
                (input.shape[0],
                    self.num_bases,
                    self.num_daughters+1,
                    -1))
        #print("FORWARD NANS", coeffs.shape, torch.sum(torch.isnan(coeffs)))
        #print("FORWARD NANS", coeffs.shape, torch.sum(torch.isnan(coeffs)))
        #print("\t", torch.sum(torch.sum(torch.isnan(coeffs[:,0]), dim=-1), dim=-1))
        eps=1e-13
        coeffs[torch.abs(coeffs) < eps] = 0
        #mask = coeffs > 0
        #coeffs[mask] = torch.max(coeffs[mask]-eps, torch.zeros_like(coeffs[mask]))
        #coeffs[~mask] = torch.min(coeffs[~mask]+eps, torch.zeros_like(coeffs[~mask]))
        #print("wavess MAX MIN", torch.amax(self.wavelets).item(), torch.amin(self.wavelets).item())
        #print("coeffs MAX MIN", torch.amax(coeffs).item(), torch.amin(coeffs).item())
        
        fit_coeffs = self.linear_fit(input)
        #deconvolved_coeffs = self.linear_fit_deconvolution(coeffs)

        return coeffs, fit_coeffs
        return coeffs, deconvolved_coeffs

    def invert_projection(self, coefficients):
        #print("WAVELETS", self.get_wavelets().shape)
        wavelets = self.get_wavelets().unsqueeze(0).unsqueeze(-2)
        #print("INVERT SHAPES", coefficients.shape, self.get_wavelets().shape)

        cf = coefficients.unsqueeze(-1)
        weighted_waves = cf*wavelets
        #print("SHAPES 1", weighted_waves.shape, self.left_padding, self.stride)
        weighted_waves = weighted_waves[:,:,:,:,self.left_padding:self.left_padding+self.stride]
        #print("SHAPES 2", weighted_waves.shape)
        weighted_waves = torch.flatten(weighted_waves, start_dim=3)
        if self.stride > 1:
            weighted_waves = weighted_waves[:,:,:,:self.prev_input_len]
        #print("SHAPES 3", weighted_waves.shape)
        return torch.sum(weighted_waves, dim=2)
        print("\t", cf.shape, wavelets.shape, (cf*wavelets).shape)
        adsf
        return torch.sum(
            torch.sum(
                torch.sum(
                   cf*wavelets, dim=1
                ),
                dim=1
            ),
            dim=-1
        )