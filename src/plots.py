from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal as sig
from scipy import fftpack
import numpy as np


def get_spectrogram(wav_file):
    """
    Calculate and plot spectrogram for a wav audio file.

    :param wav_file: (str) - path to the audio clip
    :return: pxx - (np.ndarray) - spectrum, columns are the periodograms of successive segments.
    """
    sample_rate, data = wavfile.read(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim

    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx


def plot_sample_domains(signal, sample_rate):    
    # Get sample duration.
    duration = len(signal) // sample_rate
    
    # Generate timeline.
    time = np.arange(0, duration, 1 / sample_rate)
    
    # Generate frequency axis.
    freqs = np.arange(0, sample_rate, 1 / duration)
    
    # Generate amplitude spectrum.
    spectrum = np.abs(2 * fftpack.fft(signal) / len(signal))
    
    # Plot time domain.
    plt.subplot(1, 2, 1)
    plt.plot(time, signal, c="g")
    plt.title(f"Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot frequency domain.
    plt.subplot(1, 2, 2)
    plt.plot(freqs, spectrum)
    plt.title(f"Frequency Domain (Sample Rate: {sample_rate} Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, sample_rate // 2)
    
    plt.style.use("seaborn")
    plt.rcParams["figure.figsize"] = (20,5)
    plt.tight_layout()
    plt.show()
    
    
def get_shrinked_spectrogram(audio_path):
    plt.style.use("default")
    return get_spectrogram(audio_path)