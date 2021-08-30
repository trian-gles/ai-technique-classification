import tensorflow as tf
import tensorflow_io as tfio
import os

file_paths = {}
audio_tensors = {}

for dir in os.listdir('samples/manual'):
    fullpath = os.path.join('samples/manual', dir)
    new_list = [filepath for filepath in os.listdir(fullpath)]
    file_paths[dir] = new_list

print(file_paths)

for tech_name, path in file_paths:
    audio = tfio.audio.AudioIOTensor(path)
    audio_tensors[tech_name] = tfio.audio.split(audio, axis=0, epsilon=0.1) #consider variable threshold...
