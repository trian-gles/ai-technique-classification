import os

tech_name = input("Enter the name of the technique you are training: ")
dir_path = os.path.join("training_wavs", tech_name)
try:
    os.mkdir(dir_path)
except OSError:
    print("Technique has already been added.  Appending new training files.")

from pyo import *
s = Server().boot()

filename = os.path.join(dir_path, 'initial_sample')
s.recordOptions(-1, dir_path)

s.recstart()
s.gui(locals())

