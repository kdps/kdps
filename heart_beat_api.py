import heartpy as hp
import wave
import array
import csv
import pandas
from os import path
from pydub import AudioSegment
import os

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

# --------------------------------------------------------------------------------
def bpm_api(file_name):
    # files
    src = f"{file_name}"
    dst = f"{file_name[0:len(file_name)-4]}.wav"


    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")


    # --------------------------------------------------------------------------------

    samps, fs = read_wav(dst)


    # print(len(samps))
    # print(fs/1000)
    HZ = fs/1000

    df = pandas.DataFrame(samps)
    df.to_csv("sound.csv", sep=',',index=False)

    # --------------------------------------------------------------------------------


    data = hp.get_data('sound.csv')

    working_data, measures = hp.process(data, HZ)  # 100 HZ
    # hp.plotter(working_data, measures)
    # hp.plotter(working_data, measures, show = True, title = 'Heart Rate Signal Peak Detection')

    os.remove(src)
    os.remove(dst)
    os.remove('sound.csv')

    return measures['bpm']
    # print(measures['bpm']) #returns BPM value


    # plot_object = hp.plotter(working_data, measures, show=False)
    # plot_object.savefig('plot_1.jpg') #saves the plot as JPEG image.
