import tensorflow as tf
import soundfile as sf
import librosa
import os

## Converts fmt encoded wav files to bin and limits them to the first 16000 samples

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # use GPU instead of AVX
dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir_path, "samples/manual")
filenames = tf.io.gfile.glob(path + '/*/*')

for fn in filenames:
    bin, sr = librosa.load(fn)

    if len(bin) > 16000:
        bin = bin[0:16000]
    sf.write(fn, bin, sr, format='wav')
