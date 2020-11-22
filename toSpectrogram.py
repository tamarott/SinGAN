import os
import matplotlib.pyplot as plt

# For loading and visualizing audio files
import librosa
import librosa.display

# To play audio
import IPython.display as ipd


# Finding the desired audio file
# audio_fpath = "../Input/Sounds/"
audio_fpath = "/Users/cecilieneckelmann/Documents/SinGAN-1/Input/Sounds/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))


# Returns a time series, sr is sample rate (how many samples per sec). 
# Explanation: Typically an audio signal, denoted by y, and represented as a one-dimensional numpy.ndarray of floating-point values. y[t] corresponds to the amplitude of the waveform at sample t. 
x, sr = librosa.load(audio_fpath+audio_clips[0], sr=44100)

print(type(x), type(sr))
print(x.shape, sr)

# Creating a figure-object for waveform
plt.figure(figsize=(14, 5))


librosa.display.waveplot(x, sr=sr)

# stft(x): Short-Time Fourier Transform
# This function returns a complex-valued matrix D such that 

# - np.abs(D[f, t]) is the magnitude of frequency bin f at frame t, and
# - np.angle(D[f, t]) is the phase of frequency bin f at frame t.

# created from waveform
# liniar scale - is it an array of complex no? 
# stft contains both phase and magnitude 
# what is x in librosa library. Extract every magnitude and phase of each complex no. 
X = librosa.stft(x)
# decible scale 
Xdb = librosa.amplitude_to_db(abs(X))

# find a way to convert db scale to picture. Red magnitude on DB-scale, Blue is the phase 


# creates new figure for spectrogram from the stft matrix
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()


# recovered_audio_orig = invert_pretty_spectrogram(wav_spectrogram, fft_size = fft_size,
#                                             step_size = step_size, log = True, n_iter = 10)
# IPython.display.Audio(data=recovered_audio_orig, rate=rate)