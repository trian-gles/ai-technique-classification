from pyo import *
from multiprocessing import Queue, Value, Process
import time
import numpy as np


class PlaybackTable(DataTable):
    """Tables for storing and playing wavs from RTCMIX"""
    def __init__(self, sr: int):
        super(PlaybackTable, self).__init__(size=sr * 80, chnls=2)
        self.reader = TableRead(table=self, freq=self.getRate())

    def play_wav(self, arr):
        samplist = [list(arr[:, 0]), list(arr[:, 1])]
        self.replace(samplist)
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

class AudioServer:
    def __init__(self, buffer_excerpts: Queue, ready: Value, finished: Value):
        self.s = Server(buffersize=2048)
        self.s.deactivateMidi()
        self.s.boot()

        self.buffer_excerpts = buffer_excerpts

        self.table_man = TableManager(3, self.s.getSamplingRate())

        self.t = DataTable(size=self.s.getBufferSize())
        self.inp = Input()
        self.rec = TableRec(self.inp, table=self.t).play()
        self.osc = Osc(table=self.t, freq=self.t.getRate(), mul=0.5).out()  # simple playback
        self.s.setCallback(self.callback)

    def callback(self):
        tablist = self.t.getTable()
        self.buffer_excerpts.put(tablist)
        self.rec.play()

    def start(self):
        self.s.start()

    def playback_wav(self, wav: np.ndarray):
        self.table_man.allocate_wav(wav)


