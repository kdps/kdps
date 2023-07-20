import math
import wave
import struct

nchannels = 1
sampwidth = 2
framerate = 44100
nframes = 44100
comptype = "NONE"
compname = "not compressed"
amplitude = 4000
frequency = 15000

wav_file = wave.open('15khz_sine.wav', 'w')
wav_file.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
for i in xrange(nframes):
    sample = math.sin(2*math.pi*frequency*(float(i)/framerate))*amplitude/2
    wav_file.writeframes(struct.pack('h', sample))
wav_file.close()
