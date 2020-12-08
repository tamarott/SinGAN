#!/usr/bin/python
import numpy as np
import matplotlib.image as mpimg
import wave
from array import array


def make_wav(image_filename):
    """ Make a WAV file having a spectrogram resembling an image """
    # Load image
    image = mpimg.imread(image_filename)
    image = np.sum(image, axis = 2).T[:, ::-1]
    image = image**3 # ???
    w, h = image.shape

    # Fourier transform, normalize, remove DC bias
    data = np.fft.irfft(image, h*2, axis=1).reshape((w*h*2))
    data -= np.average(data)
    data *= (2**15-1.)/np.amax(data)
    data = array("h", np.int_(data)).tostring()

    # Write to disk
    output_file = wave.open(image_filename+".wav", "w")
    output_file.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
    output_file.writeframes(data)
    output_file.close()
    print ("Wrote %s.wav" % image_filename)


if __name__ == "__main__":
#    import sys
#    make_wav(sys.argv[1])
    make_wav("/Users/Jane/Library/Mobile Documents/com~apple~CloudDocs/ITU/3.semester/Research Project/elgen-master/input/spectrogram.png")