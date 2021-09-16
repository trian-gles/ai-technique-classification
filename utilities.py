import numpy as np
import librosa


def find_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.onset.onset_detect(y, sr=sr, backtrack=True, units="samples")