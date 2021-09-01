import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_io as tfio

import time

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # use GPU instead of AVX
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, "samples/manual")



techniques = np.array(tf.io.gfile.listdir(path))

filenames = tf.io.gfile.glob(path + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print(filenames[3])

AUTOTUNE = tf.data.AUTOTUNE
#     TypeError: Cannot convert a list containing a tensor of dtype <dtype: 'int32'> to <dtype: 'float32'> (Tensor is: <tf.Tensor 'DecodeWav:1' shape=() dtype=int32>)
def get_waveform_and_label(path: str):
    bin = tf.io.read_file(path)
    audio = tf.audio.decode_wav(bin) # somewhere here it breaks.......
    tf.cast(audio, tf.float32)
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

def test_map_fn(idk):
    time.sleep(0.5)
    return idk

train_files = filenames[:120]
val_files = filenames[120:150]
test_files = filenames[150:]
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
test_ds = files_ds.map(test_map_fn, num_parallel_calls=AUTOTUNE)
for ts in test_ds:
    print(ts)

waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
for ts in waveform_ds:
    print(ts)
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


def plot_spectrogram(spectrogram: np.ndarray, ax: plt.Axes):
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


def preprocess_dataset(files: list):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

sampleset = iter(spectrogram_ds)
for i in range(1):
    successful = False
    while not successful:
        try:
            spectrogram, _ = next(sampleset)
        except:
            continue
        successful = True
        input_shape = spectrogram.shape

print('Input shape:', input_shape)
num_labels = len(techniques)

norm_layer = preprocessing.Normalization()
successful = False
while not successful:
    try:
        just_features = spectrogram_ds.map(lambda x, _: x)
    except:
        continue
    successful = True

while not successful:
    try:
        norm_layer.adapt(just_features)
    except:
        continue
    successful = True


model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)


print(spectrogram_ds.element_spec)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)