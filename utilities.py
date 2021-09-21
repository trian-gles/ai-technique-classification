import numpy as np
import librosa

techniques = ["IGNORE", "Slide", "Chord", "Harm", "Pont", "Tasto", "Smack"]


def find_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """Takes a numpy array and returns an array of onsets, currenly using librosa"""
    return librosa.onset.onset_detect(y, sr=sr, backtrack=True, units="samples")

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