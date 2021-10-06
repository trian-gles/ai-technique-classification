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
    def __init__(self, buffer_excerpts: Queue, wav_responses: Queue, ready: Value, finished: Value):
        self.s = Server(buffersize=2048)
        self.s.deactivateMidi()
        self.s.boot()

        self.buffer_excerpts = buffer_excerpts
        self.wav_responses = wav_responses

        self.table_man = TableManager(3, int(self.s.getSamplingRate()))

        self.t = DataTable(size=self.s.getBufferSize())
        self.inp = Input()
        self.rec = TableRec(self.inp, table=self.t).play()

        def callback():
            tablist = self.t.getTable()
            self.buffer_excerpts.put(tablist)
            self.rec.play()

        self.s.setCallback(callback)





    def main(self):
        self.osc = Osc(table=self.t, freq=self.t.getRate(), mul=0.5).out()  # simple playback
        self.s.start()
        while True:
            pass

    def playback_wav(self, wav: np.ndarray):
        self.table_man.allocate_wav(wav)


def test():
    ###### Set up shared information ######
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    unidentified_notes = Queue()  # stores waveforms of prepared notes that must be identified
    identified_notes = Queue()  # stores arrays of ints indicating the most to least likely technique
    wav_responses = Queue()  # wav files to be played back by the audio server
    ready_count = Value('i', 0)  # track how many processes are ready
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start

    auser = AudioServer(buffer_excerpts, wav_responses, ready, finished)
    auser.main()

if __name__ == "__main__":
    test()

