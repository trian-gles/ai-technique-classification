from pyo import *
from random import uniform

def rand():
    return uniform(0, 1)

class Drone:
    def __init__(self, vol: float):
        self.vol = vol
        self.master = SigTo(0, time=1)
        self.lfo_muls = SigTo([.1, .1])
        self.lfo = Sine([.25, .4], 0, self.lfo_muls, self.lfo_muls)
        self.a = SineLoop(freq=[midiToHz(40), midiToHz(47)], feedback=self.lfo, mul=self.master).out()



    def randomize_muls(self):
        self.lfo_muls.value = [uniform(0.01, 0.1), uniform(0.01, 0.1)]

    def randomize_freqs(self):
        self.lfo.freq = [uniform(0.2, 1), uniform(0.2, 1)]

    def randomize_all(self):
        self.randomize_freqs()
        self.randomize_muls()

    def new_pitch(self, pitch: int):
        self.a.freq = [midiToHz(pitch), midiToHz(pitch + 7)]
        self.randomize_all()
        self.master.value = self.vol

    def off(self):
        self.master.value = 0


if __name__ == "__main__":

    s = Server().boot()
    s.start()
    d = Drone(0.2)
    s.gui(locals())