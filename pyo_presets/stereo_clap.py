from pyo import *

class StereoClap:
    def __init__(self):
        # TODO - make this stereo
        self.noise = Noise()
        self.noise2 = Noise()
        self.cutoff = Sig(2800)
        self.width = Sig(300)

        self.trig = Trig()
        self.onoff = Sig(0)
        self.env = ExpTable([(0, 0.), (20, 1.), (250, 0.2), (20000, 0.)], 2)
        self.tburst = TrigBurst(self.trig, time=.05, count=2)
        self.tenv = TrigEnv(self.tburst, self.env, dur=4)
        self.hip = Biquad([self.noise, self.noise2], self.cutoff, q=3, type=1)
        self.lop = ButLP(self.hip, self.cutoff + self.width, mul=self.tenv * self.onoff)

    def set_freq(self, new_freq: float):
        self.cutoff.value = new_freq

    def ctrl(self):
        self.cutoff.ctrl([SLMap(20, 20000, 'log', 'value', 2800)], "cutoff")
        self.width.ctrl([SLMap(20, 20000, 'log', 'value', 300)], "width")

    def get_pyoobj(self):
        return self.lop.mix(2)

    def play(self):
        self.trig.play()
        self.onoff.value = 1



if __name__ == "__main__":
    s = Server().boot()


    clap = StereoClap()

    pat = Pattern(clap.play, 1)
    pat.ctrl()
    pat.play()

    mix = clap.get_pyoobj().mix(2)
    mix.out()

    s.start()

    s.gui(locals())