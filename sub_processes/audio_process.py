from pyo import *
from multiprocessing import Queue, Value, Process
import time
import numpy as np
from pyo_presets.huge_bass import StereoBass
from pyo_presets.chopped_generative2 import ChoppedGen
import keyboard
from threading import Thread


class PlaybackTable(DataTable):
    """Tables for storing and playing wavs from RTCMIX"""
    def __init__(self, sr: int):
        super(PlaybackTable, self).__init__(size=sr * 80, chnls=2)
        self.reader = TableRead(table=self, freq=self.getRate())
        self.length = 0
        self.start_time = 0

    def play_wav(self, arr):
        self.length = arr.shape[0] / 44100

        right = list(arr[:, 0])
        left = list(arr[:, 1])

        self.replace([right, left])
        self.reader.reset()


        self.reader.play().out()

    def check_playing(self) -> bool:
        if not self.reader.isPlaying():
            self.reset()
            return False

        if (time.time() - self.start_time) > self.length:
            self.reset()
            return False
        else:
            return True

    def fade_out_stop(self):
        """Fade out this table to make room for new data incoming"""

    def stop(self):
        self.reader.stop()


class TableManager:
    def __init__(self, voices, sr):
        self.voices = voices
        self.tabs = [PlaybackTable(sr) for _ in range(voices)]
        self.cursor = 0

    def allocate_wav(self, wav: np.ndarray):
        # TODO - this should fade the next up table if all tables are full
        if self.all_tabs_playing():
            print("Can't allocate new wav, all tables are full")
            return
        init_index = self.cursor
        while True:
            if not self.tabs[self.cursor].check_playing():
                thread = Thread(target=self.tabs[self.cursor].play_wav, args=(wav,))
                thread.start()
                break

            self.cursor = (self.cursor + 1) % self.voices
            if init_index == self.cursor:
                print("Could find empty table")
                self.tabs[self.cursor].play_wav(wav)
                break

    def all_tabs_playing(self):
        return all([t.check_playing() for t in self.tabs])

    def stop_all(self):
        for tab in self.tabs:
            tab.stop()


send_out = True

def toggle_send_out():
    global send_out
    send_out = not send_out
    print(f"Send out set to {send_out}")

def audio_server(buffer_excerpts: Queue, wav_responses: Queue, other_actions: Queue, ready: Value, finished: Value):
    """Still needs to handle new audio"""

    keyboard.add_hotkey("space", toggle_send_out)
    s = Server(buffersize=512)
    s.deactivateMidi()
    s.boot()

    bass = StereoBass()
    freeverb = Freeverb(bass.get_voices(), mul=0.25).out()

    gen_vox = ChoppedGen()
    gen_vox.get_pyoObj().out()

    table_man = TableManager(3, int(s.getSamplingRate()))
    t = DataTable(size=s.getBufferSize())
    inp = Input()
    rec = TableRec(inp, table=t).play()
    count = 0

    def callback():
        if send_out:
            tablist = t.getTable()
            buffer_excerpts.put(tablist)
        rec.play()

    s.setCallback(callback)
    #osc = Osc(table=t, freq=t.getRate(), mul=0.5).out()  # simple playback

    ready.value = 1
    s.start()
    while finished.value == 0:
        if not wav_responses.empty():
            new_wav = wav_responses.get()
            table_man.allocate_wav(new_wav)

        if not other_actions.empty():
            action_dict: dict = other_actions.get()
            if action_dict["METHOD"] == "BASS_NOTE":
                bass.set_notes(float(action_dict["NOTE"]) / 2)
            elif action_dict["METHOD"] == "SR_FREAK":
                bass.sr_freaks()
            elif action_dict["METHOD"] == "BASS_OFF":
                bass.off()
                table_man.stop_all()
            elif action_dict["METHOD"] == "BASS_ON":
                bass.on()
            elif action_dict["METHOD"] == "ADVANCE_GENERATIVE":
                gen_vox.advance_phase()
            elif action_dict["METHOD"] == "CHANGE_SOUND":
                gen_vox.change_sound()
            elif action_dict["METHOD"] == "PAUSE":
                gen_vox.pause(action_dict["COUNT"])
            elif action_dict["METHOD"] == "DRUMS":
                gen_vox.drums(24)
            elif action_dict["METHOD"] == "FINISH":
                gen_vox.finish()



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

