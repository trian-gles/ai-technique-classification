import numpy as np
import librosa


def find_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    o_env = librosa.onset.onset_strength(y, sr=sr)
    return librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, backtrack=True, units="samples", delta=0.2)