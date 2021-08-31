import tensorflow as tf
import tensorflow_io as tfio
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # use GPU instead of AVX

file_paths = {}
audio_tensors = {}

for dir in os.listdir('samples/manual'):
    fullpath = os.path.join('samples/manual', dir)
    new_list = [os.path.join(fullpath, filepath) for filepath in os.listdir(fullpath)]
    file_paths[dir] = new_list

for tech_name, path in file_paths.items():
    print(f"loading audio at path {path}")
    audio = tfio.audio.AudioIOTensor(path)
    audio_tensor = audio.to_tensor()
    audio_tensor = tf.squeeze(audio_tensor, axis=[-1])
    tensor = tf.cast(audio_tensor, tf.float32) / 32768.0

def split_tensor(input: tf.Tensor, axis: int, epsilon: float):
    shape = tf.shape(input, out_type=tf.int64)
    length = shape[axis]
    nonzero = tf.math.greater(input, epsilon)
