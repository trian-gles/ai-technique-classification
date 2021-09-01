import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_io as tfio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # use GPU instead of AVX

path = "samples/manual"



techniques = np.array(tf.io.gfile.listdir(path))

filenames = tf.io.gfile.glob(path + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)

AUTOTUNE = tf.data.AUTOTUNE

def get_waveform_and_label(path: str):
    bin = tf.io.read_file(path)
    audio = tfio.audio.decode_wav(bin, dtype=tf.int16)
    return tf.squeeze(audio, axis=-1), get_label(path)

def get_label(file_path: str):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def get_spectrogram(waveform: tf.Tensor):
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

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

files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

def plot_waveforms(dataset: tf.data.Dataset):
    rows = 3
    cols = 3
    n = rows*cols
    sampleset = iter(dataset)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i in range(n):
        successful = False
        while not successful:
            try:
                audio, label= next(sampleset)
            except:
                continue
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            ax.plot(audio.numpy())
            ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
            label = label.numpy().decode('utf-8')
            ax.set_title(label)
            successful = True
    plt.show()


def plot_spectrogram(spectrogram: tf.Tensor, ax: plt.Axes):
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    x = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    y = range(height)
    ax.pcolormesh(x, y, log_spec)


def plot_spectrograms(dataset: tf.data.Dataset):
    rows = 3
    cols = 3
    n = rows * cols
    sampleset = iter(dataset)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i in range(n):
        successful = False
        while not successful:
            try:
                spectrogram, label_id = next(sampleset)
            except:
                continue
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
            ax.set_title(techniques[label_id.numpy()])
            successful = True
    plt.show()


plot_spectrograms(spectrogram_ds)
