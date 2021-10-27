from typing import Union
import numpy as np
from utilities.analysis import TECHNIQUES, int_to_string_results, get_partials
from webrtcmix import generate_rtcscore, web_request, binary_scores
from audiolazy.lazy_midi import freq2midi, midi2str
import librosa
from typing import List
from multiprocessing import Queue, Value, Process
import queue
import threading


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


def dict_to_note(note_dict: dict) -> Note:
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
    def __init__(self, wav_responses: Queue, identified_notes: Queue, ready: Value, finished: Value):
        self.heat = False
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

        self.prior_notes.add(new_note)
        self.total_notes += 1

        if self.total_notes < 5: # can't peek at prior notes at the beginning
            return
        self.check_heat()

        if new_note.prediction == "SILENCE":
            prior_note: Note = self.prior_notes.peek(1)
            if prior_note.prediction == "Pont":
                high_ps = prior_note.get_high_partials()
                sco = generate_rtcscore.guitar_partials_score(*high_ps)
                self.get_send_rtc_response(sco)

        if new_note.prediction == "Harm":
            if self.prior_notes.check_contains("Harm") and not self.heat:
                cont = binary_scores.TreeContainer()
                cont.rand_graph_intervals([6, 9], num_nodes=6, pitches=(("E6"),))
                sco = cont.get_rtc_score()
                self.get_send_rtc_response(sco)

        if new_note.prediction == "Chord":
            new_note: NotSilence = new_note
            if new_note.amp > 0.2:
                fund = new_note.get_fundamental()
                amp = new_note.get_amp()
                print(f"AMP OF CHORD : {amp}")
                if fund < 130 and amp > 0.5:
                    fund_pitch = midi2str(round(freq2midi(fund)))
                    cont = binary_scores.TreeContainer()
                    cont.rand_graph_intervals([1, 5], num_nodes=6, pitches=(fund_pitch,))
                    sco = cont.get_rtc_score()
                    self.get_send_rtc_response(sco)

    def get_send_rtc_response(self, sco: str):
        new_thread = threading.Thread(target=self.async_rtc_call, args=(sco,))
        new_thread.start()

    def async_rtc_call(self, sco):
        wav = web_request.webrtc_request(sco)
        self.wav_responses.put(wav)

    def check_heat(self):
        if self.prior_notes.majority_silence():
            self.set_heat(False)
        else:
            self.set_heat(True)

    def set_heat(self, high: bool):
        if high != self.heat:
            if high:
                print(
                    """
                    #############
                    # HIGH HEAT #
                    #############
                    """)

            else:
                print("""
                    #############
                    # LOW HEAT #
                    #############
                    """)

        self.heat = high



