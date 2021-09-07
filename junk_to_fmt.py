import tensorflow as tf
import soundfile as sf
import librosa
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # use GPU instead of AVX
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, "samples/manual")
filenames = tf.io.gfile.glob(path + '/*/*')

for fn in filenames:
    bin, sr = librosa.load(fn)
    sf.write(fn, bin, sr, format='wav')
