from pyo import *
import random

class ChoppedVox(PyoObject):
    """TODO - make ramps always the same length"""
    def __init__(self, path: str, init_pitch: int, mul:float=1, add:float=0):
        super(ChoppedVox, self).__init__(mul, add)
        self._init_freq = midiToHz(init_pitch) # the pitch describing the sample
        self._t = SndTable(path)

        self._orig_freq = self._t.getRate() # the original frequency of the table
        self._no_len_freq = self._orig_freq # the frequency if the whole table were used
        self._freq = self._orig_freq # the actual frequency used by the cursor

        self._float_len = 1
        self._len = 1
        self._start_point = 0
        self._ramp = Fader(fadein=.5 / self._freq, dur=0, fadeout=0,  mul = self._len, add=self._start_point - 1)
        self._env = Fader(dur = .5 / self._freq)

        self._index = Phasor(freq=self._freq, mul = self._len * 2, add=self._start_point - 1)

        self._lookup = Lookup(table=self._t, index=self._ramp, mul=self._env)
        self._base_objs = self._lookup.getBaseObjects()

    def setFreq(self, desired_freq: float):
        self._freq = (desired_freq * self._orig_freq) / self._init_freq
        self._index.freq = self._freq / self._float_len
        self._no_len_freq = self._freq
        self._ramp.fadein = self._len / self._no_len_freq

    def setLen(self, len: float):
        print(f"Setting len to {len}")
        self._len = len
        self._index.mul = self._len
        self._index.freq = self._no_len_freq * .5/ self._len

        self._ramp.mul = self._len
        self._ramp.fadein = self._len * .5 /self._no_len_freq


    def setStart(self, start_point: float):
        self._start_point = start_point
        self._index.add = self._start_point - 1
        self._ramp.add = self._start_point - 1

    def stop(self, wait=0):
        self._ramp.stop()
        self._env.stop()

    def play(self, dur=0, delay=0):
        self._ramp.play()
        self._env.play()



if __name__ == "__main__":
    # TODO - needs white noise explosion at start
    s = Server().boot()

    #for Halcyon - len = 0.03
    #start = 0.03

    cv = ChoppedVox("burp.wav", 64, mul=0.1)
    cv2 = ChoppedVox("COY_Halcyon_vocals_70bpm_Bm.wav", 64, mul=0.3)
    cv.setLen(1)
    cv2.setLen(.1)
    cv.out()
    cv2.out()

    notes = Notein(poly=7, scale=1, mul=.5)
    notes.keyboard()

    def setFreqRand():
        possible_notes = list(filter(lambda n: n > 20, notes.get(all=True)))
        print(possible_notes)
        chose_pitch = random.choice(possible_notes)
        cv.setFreq(chose_pitch)
        cv2.setFreq(chose_pitch)
        if random.getrandbits(1) == 1:
            cv2.setStart(random.uniform(0, 0.9))
        cv.stop()
        cv2.stop()
        global c
        c = CallAfter(ca, time= .01)

    def ca():
        cv.play()
        cv2.play()

    pattern_trig = Beat(time=.125, taps=16, w1=[90,80], w2=50, w3=35, poly=1).play()
    pattern_trig.ctrl()
    tf = TrigFunc(pattern_trig, setFreqRand)

    s.gui(locals())