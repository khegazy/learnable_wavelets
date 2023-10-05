from .signal_projection import run_test as signal_projection
from .plotting_wavelets_spectrums import run_test as wavelet_spectrums

test_dict = {
    "signal_projection" : signal_projection,
    "plotting_wavelets_spectrums" : wavelet_spectrums,
}