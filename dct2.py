# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:11:58 2024

@author: noahv
"""

from pydub import AudioSegment
import numpy as np
from scipy.fftpack import dct, idct
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


#WORKS FOR 16-BIT PCM WAV

#input audio file
audio_file = 'audio/wow.wav'

#amount of compression, so right now it compresses the file by 95% 
compression_ratio=.95


output_file='compressed/compressed_output.wav'

#use I = 1 for mono audio and I = 2 for stereo
I = 2

audio = AudioSegment.from_file('audio/wow.wav', format='wav')

def dctcomp(audio_path, comp_ratio):
    #Import the audio file
    audio = AudioSegment.from_file(audio_path, format='wav')
    
    #get the waveform form the data module
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    #take the normalized discrete fourier transform
    audio_dct = dct(samples, type=2, norm = 'ortho')
    
    #decide how much to be cut off / compressed, 
    # and generate a new array containing a lot of zeros.
    cutoff = int((1-comp_ratio) * len(audio_dct))
    compressed_dct = np.zeros_like(audio_dct)
    compressed_dct[:cutoff] = audio_dct[:cutoff]
    
    #we will store the compressed data in a text file
    compressed_file = (compressed_dct[:cutoff])
    
    
    return samples, audio_dct, compressed_dct, compressed_file


samples, audio_dct, compressed_dct, compressed_file = dctcomp(audio_file, compression_ratio)


#we can store the compressed file in a txt file 
#together with the length of the original file
compressed_txt = "compressed/comptxt.txt"
np.savetxt(compressed_txt, np.append(compressed_file, len(samples)), delimiter=",", fmt='%s')

#now that we have our compressed file, 
#we can try and decompress it into an audio file
def dctdecomp(audio_path):
    #load the compressed file
    comp_file = np.loadtxt(compressed_txt, delimiter=",")
    
    #the last element of this array is the original size of the audio file
    # we use it to reshape our array into the wanted form
    decomp_dct = np.zeros(int(comp_file[-1]))
    decomp_dct[:len(comp_file)-1] = comp_file[0:-1]
    decomp_dct = decomp_dct.astype(np.float32)
    
    #we do the inverse cosine transform
    decompressed_audio = idct(decomp_dct, type=2, norm = 'ortho')
    decompressed_audio = decompressed_audio.astype(np.int16) 
    
    return decompressed_audio


#decompress your audio
decompressed_audio = dctdecomp(compressed_txt)

#save the file to pc
write(output_file, I*44100, decompressed_audio)



#%%
#Plots
x = np.arange(0, len(samples))
fig, ax = plt.subplots(2, 2)

ax[0,0].plot(x, samples)
ax[0,0].set_title("original waveform")
ax[0,0].set_xlabel('Index')
ax[0,0].set_ylabel('Amplitude')

ax[0,1].plot(x, decompressed_audio)
ax[0,1].set_title("waveform after decompression " + r"$C = $" + str(compression_ratio))
ax[0,1].set_xlabel('index')
ax[0,1].set_ylabel('Amplitude')

ax[1,0].plot(x, audio_dct)
ax[1,0].set_title("DCT of waveform")
ax[1,0].set_xlabel('Index')
ax[1,0].set_ylabel('Energy')

ax[1,1].plot(np.arange(0, len(compressed_file)), compressed_file)
ax[1,1].set_title("Compressed DCT")
ax[1,1].set_xlim(0, len(samples))
ax[1,1].set_xlabel('Index')
ax[1,1].set_ylabel('Energy')


plt.tight_layout()
plt.savefig('AudioCompression.png')
plt.show()




