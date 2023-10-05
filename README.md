# learnable_wavelets

This module provides the learnable_wavelet class which builds a wavelet basis set from a learnable mother wavelet.
In doing so, the only learnable parameters are the Fourier components of the mother wavelet.
All the daughter wavelets are derived from the mother and do not introduce any further learnable parameters.
The $i^\text{th}$ daughter is defined as
\begin{equation}
D_i(t) = M(\frac{t}{2^i})
\end{equation}
where $M(t)$ is the learned mother wavelet.
With this structure, one is able to expand the expressiveness of their basis set without adding more learnable parameters.
This is because one is provided with N bases but only one (the mother) must be learned.


Learnable wavelets applied to a temporal signal is analogous to a learned filter/kernel in a convolutional neural network (CNN).
In a CNN a small (localized) filter is learned and then convolved with the image where regions of higher similarity produce a larger overlap value.
Similarly, wavelets are localized in both time and in their Fourier components.
These learned filters are then convolved in time with the temporal signal.
Wavelets can therefore find localized changes in a temporal signal, similar to how a CNN filter that is shaped like a corner can highlight corners in an image.
Consequently, wavelets can find the initial onset (t=0) of a temporal signal, which often cannot be done in a Fourier transform as the Fourier basis is not localized in time.

