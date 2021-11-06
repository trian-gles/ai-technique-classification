from pyo import *
from pyo_presets.chopped_samper import ChoppedVox
from pyo_presets.clap import Clap
import random

class ChoppedGen:
    def __init__(self):
        self.cv = ChoppedVox("pyo_presets/COY_Halcyon_vocals_70bpm_Bm.wav", 61)
        self.cv2 = ChoppedVox("pyo_presets/COY_Halcyon_vocals_70bpm_Bm.wav", 61)

        self.cvalt = ChoppedVox("pyo_presets/Vox_DirtySample_E-D.wav", 64)
        self.cv2alt = ChoppedVox("pyo_presets/Vox_DirtySample_E-D.wav", 64)

        self.clap = Clap()

        self.noise_env = Adsr()
        self.noise = Noise()

        self.cv.setLen(0.4)
        self.cv2.setLen(0.4)

        self.max_playback = 0
        self.index = 0

        self.pattern = Beat(time=.125, taps=16, w1=[90,80], w2=50, w3=35, poly=1)
        self.tf = TrigFunc(self.pattern, self.playthrough_seq)


        scale = [0, 2, 4, 7, 9]
        scale += [n + 12 for n in scale]

        self.note_sequence = random.choices(scale, k=24)

        self.changed_sound = False

    def get_pyoObj(self):
        return self.cv + self.cv2 + self.clap.get_pyoobj() + self.cvalt + self.cv2alt

    def change_sound(self):
        self.changed_sound = not self.changed_sound

    def advance_phase(self):
        self.max_playback += random.randrange(1, 3)
        self.index = 0
        self.pattern.play()

    def play_cv(self):
        self.clap.play()
        if self.changed_sound:
            self.cvalt.play()
            self.cv2alt.play()
        else:
            self.cv.play()
            self.cv2.play()


    def playthrough_seq(self):
        print(f"Index: {self.index} Max Index: {self.max_playback}")
        sound1 = self.cv
        sound2 = self.cv2

        if self.changed_sound:
            sound1 = self.cvalt
            sound2 = self.cv2alt

        if self.index < self.max_playback:
            sound1.setFreq(midiToHz(self.note_sequence[self.index] + 81))
            sound2.setFreq(midiToHz(self.note_sequence[self.index] + 69))
            self.clap.set_freq(midiToHz(self.note_sequence[self.index] + 93))

            if random.getrandbits(1) == 1:
                new_start = random.uniform(0, 0.9)
                sound1.setStart(new_start)
                sound2.setStart(new_start)
            sound1.stop()
            sound2.stop()
            self.c = CallAfter(self.play_cv, time=.01)
            self.index += 1
            if (self.max_playback > self.pattern.taps) and (self.index == self.pattern.taps):
                self.index = 0
        else:
            self.pattern.stop()

if __name__ == "__main__":
    s= Server().boot()
    cg = ChoppedGen()
    cg.get_pyoObj().out()


    s.gui(locals())