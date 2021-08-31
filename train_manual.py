import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_io as tfio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # use GPU instead of AVX

file_paths = {}
audio_tensors = {}

path = "samples/manual"



techniques = np.array(tf.io.gfile.listdir(path))
print("Techs: ", techniques)

filenames = tf.io.gfile.glob(path + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)

print("Total examples:", num_samples)
print('Example file tensor:', filenames[0])

AUTOTUNE = tf.data.AUTOTUNE

def get_waveform_and_label(path: str):
    bin = tf.io.read_file(path)
    audio = tfio.audio.decode_wav(bin, dtype=tf.int16)
    return tf.squeeze(audio, axis=-1), get_label(path)

def get_label(file_path: str):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

files_ds = tf.data.Dataset.from_tensor_slices(filenames)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

for i in range(n):
    audio, label = waveform_ds[i]
    print(f"plotting {label} at i {i}")
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(audio.numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    label = label.numpy().decode('utf-8')
    ax.set_title(label)

plt.show()
