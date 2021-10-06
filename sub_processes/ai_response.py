from typing import Union
import numpy as np
from utilities.analysis import TECHNIQUES, int_to_string_results, get_partials
from webrtcmix import generate_rtcscore, web_request
import librosa
from typing import List
from pyo import *
from multiprocessing import Queue, Value
from utilities.pyo_util import fill_tab_np


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
                self.response = wav

        if new_note.prediction == "Chord":
            if new_note.amp > 0.2:
                fund = new_note.get_fundamental()
                if fund < 130:
                    sco = generate_rtcscore.funny_scale_score(fund)
                    self.response = web_request.webrtc_request(sco)


        self.prior_notes.add(new_note)
        self.total_notes += 1

    def get_wave_response(self) -> Union[np.ndarray, None]:
        resp = self.response
        self.response = None
        return resp


class PlaybackTable(DataTable):
    """Tables for storing and playing wavs from RTCMIX"""
    def __init__(self, sr: int):
        super(PlaybackTable, self).__init__(size=sr * 80, chnls=2)
        self.reader = TableRead(table=self, freq=self.getRate())

    def play_wav(self, wav: np.ndarray):
        fill_tab_np(wav, self)
        self.reader.reset()
        self.reader.play().out()

    def check_playing(self) -> bool:
        return self.reader.isPlaying()


class TableManager:
    def __init__(self, voices, sr):
        self.voices = voices
        self.tabs = [PlaybackTable(sr) for _ in range(voices)]
        self.cursor = 0

    def allocate_wav(self, wav: np.ndarray):
        init_index = self.cursor
        while True:
            if not self.tabs[self.cursor].check_playing():
                self.tabs[self.cursor].play_wav(wav)
                break

            self.cursor = (self.cursor + 1) % self.voices
            if init_index == self.cursor:
                print("Could find empty table")
                self.tabs[self.cursor].play_wav(wav)
                break


def ai_process(identified_notes: Queue, ready_count: Value, finished: Value, ready: Value):
    sr = 44100
    s = Server(sr=44100).boot()
    brain = Brain()
    table_man = TableManager(3, sr)
    print("AI ready to go")

    while ready.value == 0:  # wait for all processes to be ready
        pass
    s.start()
    while not finished.value == 1:
        if not identified_notes.empty():
            note_dict = identified_notes.get()
            brain.new_note(note_dict)

            wave_response = brain.get_wave_response()
            if wave_response is not None:
                table_man.allocate_wav(wave_response)

