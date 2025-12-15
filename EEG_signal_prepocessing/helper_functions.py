# Utility: Reusable plotting functions for waveform, spectrum, spectrogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as _spectrogram
from scipy.signal import butter, filtfilt
import pywt

def plot_waveform(signal, sr, ax=None, title='Waveform'):
    """Plot time-domain waveform.
    Arguments:
      signal: 1D numpy array
      sr: sample rate (Hz)
      ax: optional matplotlib Axes to plot into
    """
    time = np.arange(len(signal)) / sr
    if ax is None:
        plt.figure(figsize=(10, 3))
        plt.plot(time, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        ax.plot(time, signal)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True)


def plot_spectrum(signal, sr, n_fft=4096, ax=None, title='Magnitude Spectrum', xlim=None):
    """Plot magnitude spectrum using FFT.
    Arguments:
      signal: 1D numpy array
      sr: sample rate (Hz)
      n_fft: FFT length (zero-pad or truncate)
      ax: optional matplotlib Axes to plot into
      title: title of the plot
      xlim: tuple (min, max) for x-axis limits
    """

    # Ensure float
    sig = np.asarray(signal, dtype=float)
    # Compute rFFT
    spec = np.abs(np.fft.rfft(sig, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    if ax is None:
        plt.figure(figsize=(10, 3))
        plt.plot(freqs, spec)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(title)
        if xlim is not None:
            plt.xlim(xlim)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        ax.plot(freqs, spec)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.grid(True)


def plot_spectrogram(signal, sr, nperseg=1024, noverlap=None, cmap='magma', title='Spectrogram'):
    """Plot spectrogram (dB) using scipy.signal.spectrogram.
    Arguments:
      signal: 1D numpy array
      sr: sample rate (Hz)
    """
    noverlap = noverlap if noverlap is not None else nperseg // 2
    f, t, Sxx = _spectrogram(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap=cmap)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.colorbar(label='dB')
    plt.ylim(0, sr / 2)
    plt.tight_layout()
    plt.show()

def create_bandpass_filter(sample_rate, lowcut, highcut, order=5):
    """Create a bandpass Butterworth filter."""
    nyquist = sample_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def plot_wavelet_spectrogram(ax, signal, sr, title="Wavelet Spectrogram", wavelet='db4', level=None):
    """
    Plot a wavelet spectrogram of a 1D signal on the provided axis.

    Parameters:
    - ax: matplotlib axis to plot on
    - signal: 1D numpy array, the signal
    - sr: sampling rate
    - title: title of the plot
    - wavelet: wavelet type (default 'db4')
    - level: decomposition level (None = maximum)
    """
    # Perform DWT
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    # coeffs[1:] are detail coefficients, coeffs[0] is approx
    detail_coeffs = coeffs[1:]
    
    # Upsample each detail coefficient to original length for plotting
    detail_signals = []
    for i, c in enumerate(detail_coeffs, start=1):
        upsampled = pywt.upcoef('d', c, wavelet, level=i, take=len(signal))
        detail_signals.append(upsampled)
    
    # Stack into 2D array: rows = levels, columns = time
    spectrogram = np.vstack(detail_signals)
    
    # Plot
    im = ax.imshow(np.abs(spectrogram), aspect='auto', origin='lower', 
                   extent=[0, len(signal)/sr, 1, len(detail_signals)], cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Wavelet Level")
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.05)
