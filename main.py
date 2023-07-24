import argparse
import array
import math
import wave
import struct
import math
import matplotlib.pyplot as plt
import random
import numpy
import pywt
import heartpy as hp
import csv
import pandas
import os

from os import path
from pydub import AudioSegment
from typing import List
from pydantic import BaseModel
from scipy import signal
from fastapi import FastAPI, UploadFile

app = FastAPI()

def bpm_detector(data, fs):
    cA = []
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2 ** (levels - 1)
    min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

    for loop in range(0, levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = numpy.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")

        # 2) Filter
        cD = signal.lfilter([0.01], [1 - 0.99], cD)

        # 4) Subtract out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[:: (2 ** (levels - loop - 1))])
        cD = cD - numpy.mean(cD)

        # 6) Recombine the signal before ACF
        #    Essentially, each level the detail coefs (i.e. the HPF values) are concatenated to the beginning of the array
        cD_sum = cD[0 : math.floor(cD_minlen)] + cD_sum

    if [b for b in cA if b != 0.0] == []:
        return no_audio_data()

    # Adding in the approximate data as well...
    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - numpy.mean(cA)
    cD_sum = cA[0 : math.floor(cD_minlen)] + cD_sum

    # ACF
    correl = numpy.correlate(cD_sum, cD_sum, "full")

    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
    if len(peak_ndx) > 1:
        return no_audio_data()

    peak_ndx_adjusted = peak_ndx[0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
    print(bpm)
    return bpm, correl

def peak_detect(data):
    max_val = numpy.amax(abs(data))
    peak_ndx = numpy.where(data == max_val)
    if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
        peak_ndx = numpy.where(data == -max_val)
    return peak_ndx

def no_audio_data():
    print("No audio data for sample, skipping...")
    return None, None

def read_wav(filename):
    # open file, get metadata for audio
    try:
        wf = wave.open(filename, "rb")
    except IOError as e:
        print(e)
        return

    # typ = choose_type( wf.getsampwidth() ) # TODO: implement choose_type
    nsamps = wf.getnframes()
    assert nsamps > 0

    fs = wf.getframerate()
    assert fs > 0

    # Read entire file and make into an array
    samps = list(array.array("i", wf.readframes(nsamps)))

    try:
        assert nsamps == len(samps)
    except AssertionError:
        print(nsamps, "not equal to", len(samps))

    return samps, fs

@app.get("/")
def main():
  return 'HelloWorld';

class audioParameter(BaseModel):
  minFrequences: int
  maxFrequences: int

@app.post("/generate_audio")
def generateAudio(frequence: List[audioParameter]):

  list_names = []
  
  sampleRate = 44100.0 # hertz
  duration = 10.0 # seconds
  frequency = 440.0 # hertz
  obj = wave.open('sound.wav','w')
  obj.setnchannels(1) # mono
  obj.setsampwidth(2)
  obj.setframerate(sampleRate)

  for hz in frequence:
    for i in range(99999):
       value = random.randint(hz.minFrequences, hz.maxFrequences)
       data = struct.pack('<h', value)
       obj.writeframesraw( data )
  
  obj.close()
    
  return list_names












from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas

sys.path.append('../')
import heartpy as hb

__author__ = "Paul van Gent"
__version__ = "Version 0.9"

def read_wav(file):
    '''Function reads file and outputs sample rate and data array'''
    samplerate, wv = wavfile.read(file)
    wv_data = np.array(wv, dtype=np.float32)
    return samplerate, wv_data

def mark_audio_peaks(peaklist, samplerate):
    '''
    Marks peaks in audio signal, implements crude type detection

    Keyword arguments:
    peaklist -- 1-dimensional list or array containing detected peak locations
    samplerate -- sample rate of the original signal
    '''

    hb.calc_rr(samplerate)
    first_heartsounds = []
    second_heartsounds = []
    rejected = []

    cnt = 0
    for rr in hb.working_data['RR_list']:
        if rr <= 0.6*np.mean(hb.working_data['RR_list']):
            rejected.append(peaklist[cnt])
        elif rr >= 0.9*np.mean(hb.working_data['RR_list']):
            first_heartsounds.append(peaklist[cnt])
        else:
            second_heartsounds.append(peaklist[cnt])
        cnt += 1

    hb.working_data['peaklist'] = first_heartsounds
    hb.working_data['second_heartsounds'] = second_heartsounds
    hb.working_data['removed_beats'] = rejected

def plotter(show=True, title='Heart Rate Audio Signal Peak Detection'):
    '''Alternative plotter function to plot signals'''

    plt.title(title)
    plt.plot(abs(hb.working_data['hr']))
    plt.plot(hb.working_data['rolmean'])
    plt.scatter(hb.working_data['peaklist'], [hb.working_data['hr'][x] for x in hb.working_data['peaklist']], color='green', label='first heartsounds')
    plt.scatter(hb.working_data['second_heartsounds'], [hb.working_data['hr'][x] for x in hb.working_data['second_heartsounds']], color='black', label='second heartsounds')
    plt.scatter(hb.working_data['removed_beats'], [hb.working_data['hr'][x] for x in hb.working_data['removed_beats']], color='red', label='rejected peaks')
    plt.legend()
    plt.show()

def process(filename):
    '''Processes the wav file passed to it

    Keyword arguments:
    filename -- absolute or relative path to WAV file
    '''

    print('resampling signal to 1000Hz')
    samplerate, wv_data = read_wav(filename)
    wv_data = abs(wv_data)
    wavlength = len(wv_data) / samplerate

    wv_data = resample(wv_data, int(1000*wavlength))
    new_samplerate = len(wv_data) / wavlength

    print('detecting peaks')

    df = pandas.DataFrame(wv_data)
    df.to_csv("sound.csv", sep=',',index=False)

    data = hb.get_data('sound.csv')

    working_data, measures = hb.process(data, new_samplerate)  # 100 HZ

    return measures['bpm']





@app.post("/bpm_api")
async def bpm_api(file: UploadFile):
    UPLOAD_DIR = "./"  # 이미지를 저장할 서버 경로
    
    content = await file.read()
    filename = f"stream.mp3"  # uuid로 유니크한 파일명으로 변경
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)

    file_name = "stream.mp3";
    
    # files
    src = f"{file_name}"
    dst = f"{file_name[0:len(file_name)-4]}.wav"


    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)

    sound = sound[-5000:]
    
    sound.export(dst, format="wav")


    # --------------------------------------------------------------------------------
    samplerate, wv_data = read_wav(filename)
    wv_data = abs(wv_data)
    wavlength = len(wv_data) / samplerate

    wv_data = resample(wv_data, int(1000*wavlength))
    new_samplerate = len(wv_data) / wavlength

    print('detecting peaks')

    df = pandas.DataFrame(wv_data)
    df.to_csv("sound.csv", sep=',',index=False)

    data = hb.get_data('sound.csv')

    working_data, measures = hb.process(data, new_samplerate)  # 100 HZ

    return measures['bpm']
