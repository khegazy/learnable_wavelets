import torch
import numpy as np
from matplotlib import pyplot as plt


class learnable_wavelets(torch.nn.Module):

    def __init__(self,
                 num_bases: int,
                 num_daughters: int,
                 length: int,
                 padding: str = 'valid',
                 stride: int = 1,
                 mother_num_freqs: int = None, 
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
        factory_kwargs = {'device' : device, 'dtype' : dtype}
        self.num_bases = num_bases
        self.num_daughters = int(num_daughters)
        self.length = length
        self.N_freqs = self.length//2 + 1
        self.freqs = torch.arange(self.N_freqs)*2*torch.pi/self.length
        self.wavelets = None
        self.padding = padding
        self.stride = stride
        self.time = torch.arange(self.length) - self.length//2 + 1
        
        if mother_num_freqs:
            self.mother_num_freqs = mother_num_freqs
        else:
            self.mother_num_freqs = length//4
            print("Find proper calculation ", self.mother_num_freqs)
        self.mother_num_freqs -= self.mother_num_freqs%2
        if self.mother_num_freqs > self.length//2 + 1:
            raise ValueError("""mother_num_freqs cannot be larger 
                than length//2 + 1 (the maximum number of orthogonal 
                Fourier components)""")
        if self.mother_num_freqs < 2**(self.num_daughters):
            raise ValueError("""mother_num_freqs must be >= 
                2**num_daughters, otherwise the mother wavelet cannot 
                be divided num_daughter times""")
        if self.mother_num_freqs%(2**(self.num_daughters)) != 0:
            raise ValueError("""mother_num_freqs must be divisible 
                by 2**num_daughters""")               
        
        # Create mother wavelet
        self.mother_ft_real = torch.empty(
            (self.num_bases, self.mother_num_freqs),
            **factory_kwargs)
        self.mother_ft_imag = torch.empty(
            (self.num_bases, self.mother_num_freqs),
            **factory_kwargs)
        torch.nn.init.kaiming_uniform_(
            self.mother_ft_real,
            a=np.sqrt(5))
        torch.nn.init.kaiming_uniform_(
            self.mother_ft_imag,
            a=np.sqrt(5))

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
        self.wavelets_ft_real = torch.zeros(
            (self.num_bases, self.num_daughters+1, self.N_freqs),
            **factory_kwargs)
        self.wavelets_ft_imag = torch.zeros(
            (self.num_bases, self.num_daughters+1, self.N_freqs),
            **factory_kwargs)
        self.build_daughters_ft()

        # Recalculate daughter wavelets after backprop
        self.back_hook_handle = self.register_full_backward_hook(
            self.build_daughters_ft)

        
    def build_daughters_ft(self) -> None:
        """
        Calculate Fourier spectra of daughter wavelets given the 
        learned mother Fourier spectrum. For daughter i, the mother 
        frequency component is changed by 2**(-i).
        """
        # Reset wavelets
        self.wavelets = None
        # Set mother wavelet and daughter fourier spectras
        norm = self.get_wavelet_norm_ft_(
                self.mother_ft_real,
                self.mother_ft_imag)
        self.wavelets_ft_real[:,0,self.freq_idxs[0]] =\
            self.mother_ft_real/norm
        self.wavelets_ft_imag[:,0,self.freq_idxs[0]] =\
            self.mother_ft_imag/norm
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
        
        # AWavelet amissability requires DC component is 0
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
                real_part**2+imag_part**2,
                dim=-1,
                keepdims=True))
    
    
    def set_mother_spectrum(self, 
                            mother_real: torch.FloatTensor,
                            mother_imag: torch.FloatTensor,
                            calculate_daughters: bool = True) -> None:
        """
        Set the mother Fourier spectrum and recalculate
        
        Inputs:
            mother_real: Real part of mother Fourier spectrum
            mother_imag: Imaginary part of mother Fourier spectrum
            calculate_daughters: Recalculate daughters after chaging
                mother
        """
        self.mother_ft_real = mother_real
        self.mother_ft_imag = mother_imag
        if calculate_daughters:
            self.build_daughters_ft()
        
        
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
        fourier_spect = fourier_spect.numpy()
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
        if self.wavelets is None:
            cosines = torch.cos(
                self.freqs.unsqueeze(1)*self.time.unsqueeze(0))
            cosines = 2*self.wavelets_ft_real.unsqueeze(-1)*cosines
            sines = torch.sin(
                self.freqs.unsqueeze(1)*self.time.unsqueeze(0))
            sines = 2*self.wavelets_ft_imag.unsqueeze(-1)*sines
            self.wavelets = torch.sum(cosines+sines, dim=-2)
        
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
            self.plot_wavelets_(axs, self.wavelets[0].numpy())
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
        return torch.nn.functional.conv1d(
            input.unsqueeze(-2),
            self.get_wavelets().view(-1, 1, self.length),
            None,
            padding=self.padding).reshape(
                (input.shape[0],
                    self.num_bases,
                    self.num_daughters+1,
                    -1))
