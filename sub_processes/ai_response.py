from typing import Union
import numpy as np
from utilities.analysis import TECHNIQUES, int_to_string_results, get_partials
from webrtcmix import generate_rtcscore, web_request
from audiolazy.lazy_midi import freq2midi, midi2str
import librosa
from typing import List
from multiprocessing import Queue, Value, Process
import queue
import threading
from itertools import cycle


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

    def get_amp(self) -> float:
        return np.max(self.waveform)


    def get_high_partials(self) -> List[float]:
        fund = self.get_fundamental()
        all_partials = get_partials(self.waveform, 22050)
        high_partials = list(filter(lambda freq: freq > fund * 2, all_partials))
        while len(high_partials) < 3: # make sure there are at least 3 partials
            first_part = high_partials[0]
            high_partials.append(first_part)
        return high_partials[:3]


def dict_to_note(note_dict: dict) -> Union[Silence, NotSilence]:
    if int_to_string_results(note_dict["prediction"], TECHNIQUES)[0] == "SILENCE":
        return Silence(note_dict)
    else:
        return NotSilence(note_dict)


class Stack:
    def __init__(self, depth: int):
        self._list = []
        self._depth = depth

    def add(self, val):
        self._list.insert(0, val)
        if len(self._list) > self._depth:
            self._list = self._list[:-2]

    def peek(self, index: int):
        """Look at a value"""
        return self._list[index]


class NoteStack(Stack):
    def add(self, new_note: Note):
        """Handle combining of silences"""
        super(NoteStack, self).add(new_note)

    def get_predictions(self) -> List[str]:
        return [n.prediction for n in self._list]

    def majority_silence(self):
        predictions = self.get_predictions()
        if predictions.count("SILENCE") >= len(predictions) // 2:
            return True

    def check_contains(self, prediction: str):
        preds = self.get_predictions()
        if prediction in preds[1:]:
            return True

class Brain:
    """Controls the behaviour of the AI"""
    def __init__(self, wav_responses: Queue, identified_notes: Queue, other_actions: Queue, ready: Value, finished: Value):
        self.part = 2

        self.heat = False
        self.prior_notes = NoteStack(7)

        self.wav_responses = wav_responses
        self.identified_notes = identified_notes
        self.other_actions = other_actions
        self.finished = finished

        self.total_notes = 0

        harm_choices = ([6, 9], [3, 8])
        self.current_harm_intervals = cycle(harm_choices)

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

        self.prior_notes.add(new_note)
        self.total_notes += 1

        if self.total_notes < 5: # can't peek at prior notes at the beginning
            return

        if self.part == 1:
            self.handle_part_1(new_note)
        else:
            self.handle_part_2(new_note)

    def handle_part_1(self, new_note: Note):
        if new_note.prediction == "SILENCE":
            prior_note: Note = self.prior_notes.peek(1)
            if prior_note.prediction == "Pont":
                prior_note: NotSilence = prior_note
                high_ps = prior_note.get_high_partials()
                sco = generate_rtcscore.guitar_partials_score(*high_ps)
                self.get_send_rtc_response(sco)

        elif new_note.prediction == "Tasto":
            new_note: NotSilence = new_note
            freq = new_note.get_pitch_or_lowest()
            print(f"Tasto frequency = {freq}")
            if freq < 200:
                self.other_actions.put(
                    {
                        "METHOD": "BASS_NOTE",
                        "NOTE": freq
                    })
        elif new_note.prediction == "Harm":
            self.other_actions.put(
                {
                    "METHOD": "SR_FREAK"
                })

    def handle_part_2(self, new_note: Note):
        if new_note.prediction == "Smack":
            self.other_actions.put(
                {
                    "METHOD": "ADVANCE_GENERATIVE"
                })
        elif new_note.prediction == "Chord":
            self.other_actions.put(
                {
                    "METHOD": "CHANGE_SOUND"
                })

    def get_send_rtc_response(self, sco: str):
        new_thread = threading.Thread(target=self.async_rtc_call, args=(sco,))
        new_thread.start()

    def async_rtc_call(self, sco):
        wav = web_request.webrtc_request(sco)
        self.wav_responses.put(wav)



