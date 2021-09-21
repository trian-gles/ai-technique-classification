import numpy as np
import librosa
from typing import List
import matplotlib.pyplot as plt

TECHNIQUES = ["IGNORE", "Slide", "Chord", "Harm", "Pont", "Tasto", "Smack"]


def find_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """Takes a numpy array and returns an array of onsets, currenly using librosa"""
    return librosa.onset.onset_detect(y, sr=sr, backtrack=True, units="samples")


def get_waveform_from_ndarray(audio: np.ndarray, tf):
    audio = tf.convert_to_tensor(audio)

    tf.cast(audio, tf.float32)
    return audio


def get_waveform_from_bin(wfbin, tf):
    """Returns a tf tensor float32 waveform from a binary file"""
    audio, _ = tf.audio.decode_wav(wfbin)  # somewhere here it breaks.......
    tf.cast(audio, tf.float32)
    return tf.squeeze(audio, axis=-1)


def get_waveform_from_path(path: str, tf):
    """Returns a tf tensor float32 waveform from a path"""
    wfbin = tf.io.read_file(path)
    return get_waveform_from_bin(wfbin, tf)


def get_spectrogram(waveform, tf):
    """Takes a tf.float32 waveform and returns a spectrogram.  Max size = 16000 samples"""
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


def numpy_to_tfdata(note: np.ndarray, tf):
    """Turn a numpy buffer note into a tensorflow dataset of the spectrogram"""
    waveform = get_waveform_from_ndarray(note, tf)
    spec = get_spectrogram(waveform, tf)
    ds = tf.data.Dataset.from_tensors([spec])
    return ds


def int_to_string_results(int_results: List[int], techniques: List[str]) -> List[str]:
    return list(map(lambda i: techniques[i], int_results))


def prediction_to_int_ranks(prediction, tf):
    sftmax = tf.nn.softmax(prediction[0])
    sorted = np.sort(sftmax)[::-1]
    index_of = lambda x: np.where(sftmax == x)[0][0]
    prediction_ranks = list(map(index_of, sorted))
    return prediction_ranks


def plot_prediction(techniques, prediction, tf):
    """view a matplotlib graph of the prediction"""
    plt.bar(techniques, tf.nn.softmax(prediction[0]))
    plt.title(f'Predictions for new note:')
    plt.show()


def note_above_threshold(note: np.ndarray):
    """Checks if the peak of a note is above a set threshold"""
    return np.max(np.abs(note)) > 0.09

