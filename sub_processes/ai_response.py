from typing import Union
import numpy as np
from utilities.analysis import TECHNIQUES, int_to_string_results, get_partials
from webrtcmix import generate_rtcscore, web_request
import librosa
from typing import List
from multiprocessing import Queue, Value, Process
import queue


class Note:
    """Stores the result dict in a class for easier access"""
    def __init__(self, note_dict: dict):
        str_results = int_to_string_results(note_dict["prediction"], TECHNIQUES)
        print(f"New note: {str_results[0:3]}")

        self.prediction: str = str_results[0]
        self.waveform: np.ndarray = note_dict["waveform"]
        self.amp: int = note_dict["amp"]


class Silence(Note):
    def __init__(self, note_dict: dict):
        super().__init__(note_dict)


class NotSilence(Note):
    def __init__(self, note_dict: dict):
        super().__init__(note_dict)
        self.spectrogram = note_dict["spectrogram"]


    def get_pitch_or_lowest(self):
        if self.prediction == "Chord":
            return self.get_lowest_partial()
        else:
            return self.get_fundamental()


    def get_fundamental(self):
        freqs, _, _ = librosa.pyin(self.waveform, 80, 1279) # 80 is the lowest note on the guitar
        freqs = np.array(list(filter(lambda freq: not np.isnan(freq), freqs))) # get rid of nan
        return np.average(freqs)

    def get_lowest_partial(self) -> float:
        """for chords"""
        all_partials = get_partials(self.waveform, 22050)
        return min(all_partials)


    def get_high_partials(self) -> List[float]:
        fund = self.get_fundamental()
        all_partials = get_partials(self.waveform, 22050)
        high_partials = list(filter(lambda freq: freq > fund * 2, all_partials))
        while len(high_partials) < 3: # make sure there are at least 3 partials
            first_part = high_partials[0]
            high_partials.append(first_part)
        return high_partials[:3]


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
    """Controls the behaviour of the AI"""
    def __init__(self, wav_responses: Queue, identified_notes: Queue, ready: Value, finished: Value):
        self.heat = 0
        self.prior_notes = NoteStack(7)

        self.wav_responses = wav_responses
        self.identified_notes = identified_notes
        self.finished = finished

        self.total_notes = 0

    def main(self):
        while True:
            note_dict = None
            try:
                note_dict = self.identified_notes.get_nowait()
            except queue.Empty:
                continue
            self.new_note(note_dict)


    def new_note(self, note_dict: dict):
        new_note = dict_to_note(note_dict)

        if new_note.prediction == "SILENCE" and self.total_notes > 0:
            prior_note: Note = self.prior_notes.peek()
            if prior_note.prediction == "Pont":
                high_ps = prior_note.get_high_partials()
                print(high_ps)
                sco = generate_rtcscore.guitar_partials_score(*high_ps)
                wav = web_request.webrtc_request(sco)
                self.send_wav_response(wav)

        if new_note.prediction == "Chord":
            if new_note.amp > 0.2:
                fund = new_note.get_fundamental()
                if fund < 130:
                    sco = generate_rtcscore.funny_scale_score(fund)
                    wav = web_request.webrtc_request(sco)
                    self.send_wav_response(wav)


        self.prior_notes.add(new_note)
        self.total_notes += 1

    def send_wav_response(self, wav: np.ndarray):
        self.wav_responses.put(wav)

