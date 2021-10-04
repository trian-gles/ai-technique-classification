from typing import Union
import numpy as np
from utilities.analysis import TECHNIQUES, int_to_string_results, high_partials
from webrtcmix import generate_rtcscore, web_request


class Note:
    """Stores the result dict in a class for easier access"""
    def __init__(self, note_dict: dict):
        str_results = int_to_string_results(note_dict["prediction"], TECHNIQUES)
        print(f"New note: {str_results[0:3]}")
        self.prediction: str = str_results[0]
        self.waveform: np.ndarray = note_dict["waveform"]
        self.amp: int = note_dict["amp"]

    def get_high_partials(self):
        return high_partials(self.waveform, 3)


class Silence(Note):
    def __init__(self, note_dict: dict):
        super().__init__(note_dict)


class NotSilence(Note):
    def __init__(self, note_dict: dict):
        super().__init__(note_dict)
        self.spectrogram = note_dict["spectrogram"]


def dict_to_note(note_dict: dict) -> Note:
    if int_to_string_results(note_dict["prediction"], TECHNIQUES)[0] == "SILENCE":
        return Silence(note_dict)
    else:
        return NotSilence(note_dict)


class Stack:
    def __init__(self, depth: int):
        self.list = []
        self.depth = depth

    def add(self, val):
        self.list.insert(0, val)
        if len(self.list) > self.depth:
            self.list = self.list[:-2]

    def peek(self):
        """Look at the top value"""
        return self.list[0]


class NoteStack(Stack):
    def filter_to_list(self, entry: str):
        return [note[entry] for note in self.list]

    def add(self, new_note: Note):
        """Handle combining of silences"""
        super(NoteStack, self).add(new_note)


class Brain:
    def __init__(self):
        self.heat = 0
        self.prior_notes = NoteStack(7)
        self.response = None
        self.total_notes = 0

    def new_note(self, note_dict: dict):
        new_note = dict_to_note(note_dict)

        if new_note.prediction == "SILENCE" and self.total_notes > 0:
            prior_note: Note = self.prior_notes.peek()
            if prior_note.prediction == "Pont":
                high_ps = prior_note.get_high_partials()
                print(high_ps)
                sco = generate_rtcscore.guitar_partials_score(*high_ps)
                wav = web_request.webrtc_request(sco)
                web_request.play_np(wav)

        self.prior_notes.add(new_note)
        self.total_notes += 1

    def get_wave_response(self) -> Union[np.ndarray, None]:
        return self.response

