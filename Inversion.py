import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

if __name__ == '__main__':
    file = "Input/Sounds/Waves.wav"
    x, sr = librosa.load(file, sr=None)
    
    #worst quality
    #D = np.abs(librosa.stft(x))**2
    #melspectrogram = librosa.feature.melspectrogram(y=x, sr=sr, S=D)
    #print('melspectrogram.shape', melspectrogram.shape)
    #print(melspectrogram)
    #audio_signal = librosa.feature.inverse.mel_to_audio(melspectrogram)
    #sf.write('test.wav', audio_signal, sr)

    #2nd way (better with Applause than Waves?)
    abs_spectrogram = np.abs(librosa.stft(x)) 
    plt.figure(figsize=(14, 5), frameon=False)
    plt.axis('off')
    librosa.display.specshow(abs_spectrogram, sr=sr)
    plt.tight_layout()
    plt.savefig('test.png', bbox_inches='tight', pad_inches=0)
    #audio_signal = librosa.griffinlim(abs_spectrogram) #estimates the phase w. the Griffin-Lim Algorithm (GLA)
    #print(audio_signal, audio_signal.shape)
    #sf.write('test.wav', audio_signal, sr)

    #3rd way (same output as 2nd?)
    #spectrum = librosa.stft(x)
    #reconstructed_audio = librosa.istft(spectrum) #Inverse STFT
    #sf.write('Applause4.wav', reconstructed_audio, sr)
    #print(sum(x[:len(reconstructed_audio)] - reconstructed_audio))  # very close to 0