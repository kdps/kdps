from __future__ import print_function
import scipy.io.wavfile as wavf
import numpy as np

if __name__ == "__main__":

    samples = [np.random.randn(44100),np.random.randn(44100),np.random.randn(44100)]
    fs = 44100
    out_f = 'out.wav'

    wavf.write(out_f, fs, samples)
