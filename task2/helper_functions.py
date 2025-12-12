# Utility: Reusable plotting functions for waveform, spectrum, spectrogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as _spectrogram


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