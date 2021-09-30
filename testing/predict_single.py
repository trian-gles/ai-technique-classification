import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import seaborn as sns
import simpleaudio
import librosa
from utilities import parse_result, int_to_string_results, plot_prediction

import time

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # use GPU instead of AVX
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, "samples/manual")

#techniques = (tf.io.gfile.listdir(path))
techniques = ["IGNORE", "Slide", "Chord", "Harm", "Pont", "Tasto", "Smack"]

test_basefiles = os.listdir("../test_sampls")
test_files = [os.path.join("../test_sampls", bf) for bf in test_basefiles]

#print(techniques)

def get_waveform(path: str):
    bin = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(bin)  # somewhere here it breaks.......
    tf.cast(audio, tf.float32)
    return tf.squeeze(audio, axis=-1)


def get_waveform_and_label(path: str):
    bin = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(bin) # somewhere here it breaks.......
    tf.cast(audio, tf.float32)
    return tf.squeeze(audio, axis=-1), get_label(path)

def get_label(file_path: str):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_spectrogram(waveform: tf.Tensor):
    if tf.shape(waveform) > 16000:
        waveform = waveform[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32) #fix this so the padding isn't huge

    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)

    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == techniques)
    return spectrogram, label_id

def preprocess_dataset(files: list):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds

#audio, _ =  librosa.load(test_file, 22050)
#tf.cast(audio, tf.float32)


#audio = tf.convert_to_tensor(audio)
#spec = get_spectrogram(audio)
model = tf.keras.models.load_model("savedModel")


for test_file in test_files:
    audio, _ =  librosa.load(test_file, 22050)
    tf.cast(audio, tf.float32)

    audio = tf.convert_to_tensor(audio)
    spec = get_spectrogram(audio)
    #spec = get_spectrogram(get_waveform(test_file))
    sample_ds = tf.data.Dataset.from_tensors(spec)
    for spectrogram in sample_ds.batch(1):
        start_tim = time.time()
        prediction = model(spectrogram)
        print(f"Time to predict = {time.time() - start_tim}")
        plot_prediction(techniques, prediction, tf)



