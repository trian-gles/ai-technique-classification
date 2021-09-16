import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import seaborn as sns
import simpleaudio
import librosa

import time

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # use GPU instead of AVX
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, "samples/manual")

techniques = (tf.io.gfile.listdir(path))

test_file = "samples/manual/chord/chord10.wav"
print(techniques)


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

def parse_result(prediction, tf):
    sftmax = tf.nn.softmax(prediction[0])
    sorted = np.sort(sftmax)[::-1]
    index_of = lambda x: np.where(sftmax == x)[0][0]
    prediction_ranks = list(map(index_of, sorted))
    return prediction_ranks

audio, _ =  librosa.load(test_file, 22050)
tf.cast(audio, tf.float32)


audio = tf.convert_to_tensor(audio)
spec = get_spectrogram(audio)


sample_ds = tf.data.Dataset.from_tensors(spec)
for spectrogram in sample_ds.batch(1):
    model = tf.keras.models.load_model("savedModel")
    start_tim = time.time()
    prediction = model(spectrogram)
    int_results = parse_result(prediction, tf)
    str_results = list(map(lambda i: techniques[i], int_results))
    print(str_results)
    print(f"Time to predict = {time.time() - start_tim}")

    plt.bar(techniques, tf.nn.softmax(prediction[0]))
    plt.title(f'Predictions for tasto')
    plt.show()