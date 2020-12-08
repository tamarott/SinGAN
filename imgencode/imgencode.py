#!/usr/bin/python
from PIL import Image
import math, wave, array, sys, getopt

def start(inputfile, outputfile, duration):
    im = Image.open(inputfile)
    width, height = im.size
    rgb_im = im.convert('RGB')
    
    durationSeconds = float(duration) 
    tmpData = []
    maxFreq = 0
    data = array.array('h')
    sampleRate = 44100
    channels = 1
    dataSize = 2 
    
    numSamples = int(sampleRate * durationSeconds)
    samplesPerPixel = math.floor(numSamples / width)
    
    C = 20000 / height
    
    for x in range(numSamples):
        rez = 0
        
        pixel_x = int(x / samplesPerPixel)
        if pixel_x >= width:
            pixel_x = width -1
            
        for y in range(height):
            r, g, b = rgb_im.getpixel((pixel_x, y))
            s = r + g + b
            
            volume = s * 100 / 765
            
            if volume == 0:
                continue
            
            freq = int(C * (height - y + 1))
            
            rez += getData(volume, freq, sampleRate, x)

        tmpData.append(rez)
        if abs(rez) > maxFreq:
            maxFreq = abs(rez)
    
    for i in range(len(tmpData)):
        data.append(int(32767 * tmpData[i] / maxFreq))
    
    f = wave.open(outputfile, 'w')
    f.setparams((channels, dataSize, sampleRate, numSamples, "NONE", "Uncompressed"))
    f.writeframes(data.tobytes())
    f.close()
            
def getData(volume, freq, sampleRate, index):
    return int(volume * math.sin(freq * math.pi * 2 * index /sampleRate))

if __name__ == '__main__':
    inputfile = ''
    outputfile = ''
    duration = ''
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:t:")
    except getopt.GetoptError:
        print('imgencode.py -i <input_picture> -o <output.wav> -t <duration_seconds>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print('imgencode.py -i <input_picture> -o <output.wav> -t <duration_seconds>')
            sys.exit()
        elif opt == "-i":
            inputfile = arg
        elif opt == "-o":
            outputfile = arg
        elif opt == "-t":
            duration = arg

    start(inputfile, outputfile, duration)
