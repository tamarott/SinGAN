import os
import matplotlib.pyplot as plt
import SinGAN.functions as functions

# For loading and visualizing audio files
import librosa
import librosa.display

# To play audio
import IPython.display as ipd
from config import get_arguments
import sys

def main():
    fileno = int(sys.argv[1])
    functions.create_Spect(fileno)
     

if __name__ == "__main__":
    main()

