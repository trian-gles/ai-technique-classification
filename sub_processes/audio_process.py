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


def audio_server(buffer_excerpts: Queue, wav_responses: Queue, ready: Value, finished: Value):
    """Still needs to handle new audio"""
    s = Server(buffersize=2048)
    s.deactivateMidi()
    s.boot()
    table_man = TableManager(3, int(s.getSamplingRate()))
    t = DataTable(size=s.getBufferSize())
    inp = Input()
    rec = TableRec(inp, table=t).play()
    count = 0

    def callback():
        tablist = t.getTable()
        buffer_excerpts.put(tablist)
        rec.play()

    s.setCallback(callback)
    osc = Osc(table=t, freq=t.getRate(), mul=0.5).out()  # simple playback

    ready.value = 1
    s.start()
    while finished.value == 0:
        if not wav_responses.empty():
            new_wav = wav_responses.get()
            table_man.allocate_wav(new_wav)


def test():
    import librosa
    import webrtcmix.web_request
    ###### Set up shared information ######
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    unidentified_notes = Queue()  # stores waveforms of prepared notes that must be identified
    identified_notes = Queue()  # stores arrays of ints indicating the most to least likely technique
    wav_responses = Queue()  # wav files to be played back by the audio server
    ready_count = Value('i', 0)  # track how many processes are ready
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start

    wav = webrtcmix.web_request.webrtc_request(webrtcmix.web_request.score_str1)

    wav_responses.put(wav)


    audio_server(buffer_excerpts, wav_responses, ready, finished)

    print("Finished")


if __name__ == "__main__":
    test()

