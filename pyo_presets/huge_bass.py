from pyo import *
import random

class HugeBass:
    def __init__(self, voices=4, harms=24):
        self.voices = voices

        saw_harms = [1 / i for i in range(1, harms + 1)]
        self.square_table = HarmTable(saw_harms)
        self.freq = SigTo(value=0, time=.1)

        self.master_mastervol = SigTo(value=1, time=2, init=1)
        self.reg_mul = SigTo(value= 1 / voices, time=2, init= 1 / voices)
        self.oct_mul = SigTo(value=0, time=2, init=0)

        self.players = [Osc(mul=self.reg_mul * self.master_mastervol, freq=self.freq * random.uniform(0.97, 1.03),
                            table=self.square_table, phase=random.uniform(0, 1)).play() for _ in range(voices)]

        self.octave_players = [Osc(mul=self.oct_mul, freq=self.freq * random.uniform(0.90, 1.1) * 8,
                            table=self.square_table, phase=random.uniform(0, 1)).play() for _ in range(voices)]

        self.sr_subtraction = Choice(choice=[0.04, 0.05, 0.1, 0.2, 0.09], freq=3, mul=1).play()
        self.degrade = Degrade(sum(self.players) + sum(self.octave_players), bitdepth=3, srscale=1)

        self.mode = "on"


        self.biquad_cutoff_env = Linseg([(0, 0), (1, 0), (2, 40), (4, 0)], loop = False)

        self.biquad_cutoff_mul = Sig(10 + self.biquad_cutoff_env)
        self.biquad = Biquad(self.degrade, freq=self.freq * self.biquad_cutoff_mul, q=0.56)


    def get_pyoobj(self):
        return self.biquad

    def randomize(self):
        for s in self.players:
            s.freq = 220 * random.uniform(0.97, 1.03)

    def sr_freakout(self):
        """TODO - can you make this gradual"""
        if self.mode == "off":
            return
        self.mode = "sr"
        self.c = CallAfter(self.pitch_return, 4.5)
        self.degrade.srscale = self.sr_subtraction
        self.sr_subtraction.freq = random.uniform(1, 3)
        self.reg_mul.value = 0
        self.oct_mul.value = 1 / self.voices
        self.biquad_cutoff_env.play()

    def pitch_return(self):
        self.d = CallAfter(self.sr_return, 1)
        self.reg_mul.value = 1 / self.voices
        self.oct_mul.value = 0

    def sr_return(self):
        self.degrade.srscale = 1
        self.mode = "on"


    def set_note(self, pitch: float):
        self.freq.time = 1
        self.freq.value = pitch
        self.biquad_cutoff_env.play()
        self.mode = "on"

    def off(self):
        self.master_mastervol.value = 0
        self.reg_mul.value = 0
        self.mode = "off"

    def on(self):
        self.reg_mul.value = 1 / self.voices
        self.master_mastervol.value = 1
        self.mode = "on"

    def ctrl(self):
        self.biquad.ctrl()
        self.degrade.ctrl()


class StereoBass:
    def __init__(self):
        self.voices = [HugeBass(), HugeBass()]

    def get_voices(self):
        return [v.get_pyoobj() for v in self.voices]

    def set_notes(self, note):
        if self.voices[0].mode == 'on':
            for voice in self.voices:
                voice.set_note(note)

    def off(self):
        for voice in self.voices:
            voice.off()

    def on(self):
        for voice in self.voices:
            voice.on()

    def sr_freaks(self):
        if self.voices[0].mode == 'on':
            for voice in self.voices:
                voice.sr_freakout()



if __name__ == "__main__":
    s = Server().boot()
    bass = StereoBass()

    notes = Notein(poly=1, scale=1, mul=.5)
    notes.keyboard()
    note_change_trig = TrigFunc(notes['trigon'], bass.set_notes, notes['pitch'])

    freeverb = Freeverb(bass.get_voices()).out()

    s.gui(locals())