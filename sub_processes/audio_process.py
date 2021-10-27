from pyo import *
from multiprocessing import Queue, Value, Process
import time
import numpy as np


class PlaybackTable(DataTable):
    """Tables for storing and playing wavs from RTCMIX"""
    def __init__(self, sr: int):
        super(PlaybackTable, self).__init__(size=sr * 80, chnls=2)
        self.reader = TableRead(table=self, freq=self.getRate())
        self.length = 0
        self.start_time = 0

    def play_wav(self, arr):
        self.length = arr.shape[0] / 44100
        samplist = [list(arr[:, 0]), list(arr[:, 1])]

        self.replace(samplist)
        self.reader.reset()


        self.start_time = time.time()
        self.reader.play().out()

    def check_playing(self) -> bool:
        if not self.reader.isPlaying():
            return False

        if (time.time() - self.start_time) > self.length:
            return False
        else:
            return True


class TableManager:
    def __init__(self, voices, sr):
        self.voices = voices
        self.tabs = [PlaybackTable(sr) for _ in range(voices)]
        self.cursor = 0

    def allocate_wav(self, wav: np.ndarray):
        # TODO - this should fade the next up table if all tables are full
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


def test_playback():
    import librosa
    import webrtcmix.web_request
    ###### Set up shared information ######
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    wav_responses = Queue()  # wav files to be played back by the audio server
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start
    wav = webrtcmix.web_request.webrtc_request(webrtcmix.web_request.score_str1)
    wav_responses.put(wav)
    audio_server(buffer_excerpts, wav_responses, ready, finished)

    print("Finished")


def test_buffer_quality():
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    wav_responses = Queue()  # wav files to be played back by the audio server
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start

    def buffer_gather_process(buffer_excerpts: Queue, wav_responses: Queue):
        start_time = time.time()
        while (time.time() - start_time) < 10:
            pass
        print("finished")
        playback_buffers = buffer_excerpts.get()
        while not buffer_excerpts.empty():
            playback_buffers = np.concatenate((playback_buffers, buffer_excerpts.get()))
        wav_responses.put(playback_buffers)


    test_buffer = Process(target=buffer_gather_process, args=(buffer_excerpts, wav_responses))
    test_buffer.start()
    audio_server(buffer_excerpts, wav_responses, ready, finished)

def buffer_gather_test_process(buffer_excerpts: Queue, wav_responses: Queue):
    empty_buf = True
    playback_buffer = None

    while True:
        if not buffer_excerpts.empty():
            if empty_buf:
                playback_buffer = buffer_excerpts.get()
                empty_buf = False
            else:
                playback_buffer = np.concatenate((playback_buffer, buffer_excerpts.get()))
                if len(playback_buffer) > 441000:
                    break



    print("Finished")
    import soundfile
    print(playback_buffer)
    soundfile.write("test_buffer.wav", playback_buffer, 44100)






    print("Ten seconds is up")


if __name__ == "__main__":
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    wav_responses = Queue()  # wav files to be played back by the audio server
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start





    test_buffer = Process(target=buffer_gather_test_process, args=(buffer_excerpts, wav_responses))
    test_buffer.start()
    audio_server(buffer_excerpts, wav_responses, ready, finished)

