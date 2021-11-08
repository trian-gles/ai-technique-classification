from pyo import *
import random

class ChoppedVox(PyoObject):
    def __init__(self, path: str, init_freq: float, dur: float, mul:float=1, add:float=0):
        super(ChoppedVox, self).__init__(mul, add)

        self._dur = dur

        self._init_freq = init_freq
        self._table = SndTable(path)
        self._env = Adsr(dur = self._dur)
        self._mul = Sig(value=mul)
        self._looper = Looper(self._table, mul=self._env * self._mul, mode=3, xfade=0)

        self._base_objs = self._looper.getBaseObjects()


    def change_freq(self, new_freq: float):
        self._looper.pitch = new_freq / self._init_freq

    def play(self, dur=0, delay=0):
        self._looper.dur = self._dur
        self._looper.loopnow()
        self._env.play()

    def set_start(self, new_start: float):
        self._looper.start = new_start

    def ctrl(self, map_list=None, title=None, wxnoserver=False):
        self._env.ctrl()



if __name__ == "__main__":
    # TODO - needs white noise explosion at start
    s = Server().boot()
    cv = ChoppedVox("COY_Halcyon_vocals_70bpm_Bm.wav", midiToHz(64), dur=1)
    cv.ctrl()
    f = Freeverb(cv).out()
    s.gui(locals())

