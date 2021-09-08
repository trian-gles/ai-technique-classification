import librosa
import librosa.display
import os
import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, choice
import sounddevice as sd
import soundfile as sf
import string


def random_string() -> str:
    name = ''
    for _ in range(8):
        name += choice(string.ascii_uppercase)
    return name


def plot_wf(y: np.array, sr: int) -> None:
    onsets_time = librosa.onset.onset_detect(y, sr, backtrack=True, units='time')

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[0])
    otherimg = librosa.display.waveshow(y, sr=sr, ax=ax[1])

    ax[1].vlines(onsets_time, max(y), min(y), color='r', alpha=0.9, linestyle='--')
    plt.show()


def plot_segments(samps: [], sr: int) -> None:
    for i, segment in enumerate(samps):
        r = i // 3
        c = i % 3

        librosa.display.waveshow(playback_samps[i], sr=sr, ax=ax[r][c])
    plt.show()


playback_samps = []
def on_press(event):
    sample_i = int(event.key) - 1
    sd.play(playback_samps[sample_i], 22050)

dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, "samples/unsorted")
save_path = os.path.join(dir_path, "samples/manual")
filenames = map(lambda f: os.path.join(path, f), os.listdir(path))
for filename in filenames:
    tech_name = input("Enter technique name")
    y, sr = librosa.load(filename)
    onsets = librosa.onset.onset_detect(y, sr, backtrack=True, units='samples')


    segments = []
    last_onset = 0
    for on in onsets:
        new_seg = y[last_onset:on]
        peak = np.max(np.abs(new_seg))
        if peak > 0.07:
            segments.append(new_seg)
        last_onset = on

    playback_samps = segments.copy()
    shuffle(playback_samps)
    fig, ax = plt.subplots(3, 3)
    fig.canvas.mpl_connect('key_press_event', on_press)


    tech_dir = os.path.join(save_path, tech_name)
    os.mkdir(os.path.join(save_path, tech_name))
    for s in segments:
        name = random_string() + ".wav"
        print("Writing file " + name)
        fullpath = os.path.join(tech_dir, name)
        sf.write(fullpath, s, sr)


