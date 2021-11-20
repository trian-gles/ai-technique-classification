from pyo import *
from pyo_presets.chopped_sampler2 import ChoppedVox
from pyo_presets.stereo_clap import StereoClap
import random

class ChoppedGen:
    # TODO - get a reverse snare sound!!!
    def __init__(self):
        file1 = "pyo_presets/COY_Halcyon_vocals_70bpm_Bm.wav"

        file2 = "pyo_presets/Vox_DirtySample_E-D.wav"
        file3 = "pyo_presets/guitar.wav"
        self.cv = ChoppedVox(file1, midiToHz(61), dur=2, mul=1)
        self.cv2 = ChoppedVox(file1, midiToHz(61), dur=2, mul=1)

        self.cvalt = ChoppedVox(file2, midiToHz(64), dur=2, mul=.8)
        self.cv2alt = ChoppedVox(file2, midiToHz(64), dur=2, mul=.8)

        self.cvfin = ChoppedVox(file3, midiToHz(61), dur=2, mul=1)
        self.cv2fin = ChoppedVox(file3, midiToHz(61), dur=2, mul=1)


        self.temp_drum_count = 0
        self.finished = False

        self.snaretab = SndTable("pyo_presets/snare.wav")
        self.kicktab = SndTable("pyo_presets/kick.wav")
        self.thumptab = SndTable("pyo_presets/thump_1.wav")
        self.rev_snare_tab = SndTable("pyo_presets/rev_snare.wav")

        self.rev_snare = TableRead(table=self.rev_snare_tab, mul=0.4)
        self.snare = TableRead(table=self.snaretab, mul=0.3)
        self.kick = TableRead(table=self.kicktab, mul=0.3, freq=self.kicktab.getRate())
        self.thump = TableRead(table=self.thumptab, mul=0.3, freq=self.thumptab.getRate())

        self.clap = StereoClap()

        self.noise_env = Adsr()
        self.noise = Noise() + Noise()

        self.max_playback = 0
        self.index = 0

        self.pattern = Beat(time=.125, taps=16, w1=90, w2=50, w3=35, poly=2)
        self.tf = TrigFunc(self.pattern, self.playthrough_seq)

        self.pauses_remaining = 0


        scale = [0, 2, 4, 7, 9]
        scale += [n + 12 for n in scale]

        self.note_sequence = random.choices(scale, k=24)

        self.changed_sound = False

        self.initial_sequence = []

    def get_pyoObj(self):
        return self.cv + self.cv2 + self.clap.get_pyoobj() + self.cvalt + \
               self.cv2alt + self.kick + self.snare + self.thump + self.rev_snare + self.cvfin + self.cv2fin

    def change_sound(self):
        self.changed_sound = not self.changed_sound
        if self.changed_sound == False:
            self.pattern.new()

    def advance_phase(self):
        if self.pattern.isPlaying():
            return
        self.max_playback += random.randrange(1, 3)
        if self.max_playback > 12:
            self.max_playback = self.pattern.taps + 1
        self.index = 0
        self.pattern.play()

    def play_cv(self):
        self.clap.play()
        if self.finished:
            self.cvfin.play()
            self.cv2fin.play()
        elif self.changed_sound:
            self.cvalt.play()
            self.cv2alt.play()
        else:
            self.cv.play()
            self.cv2.play()

        if self.temp_drum_count > 0:
            s = random.randrange(3)
            if s == 0:
                self.kick.play()
            elif s == 1:
                self.snare.play()
                if random.getrandbits(1):
                    self.c = CallAfter(self.snare.play, time = .125)
            self.temp_drum_count -= 1


    def stop(self):
        self.pattern.stop()

    def pause(self, count: int):
        if self.finished:
            return
        self.pauses_remaining = count

    def drums(self, count: int = 32):
        if self.finished:
            return
        if self.max_playback == self.pattern.taps + 1:
            self.temp_drum_count = count

    def finish(self):
        self.finished = True
        self.clap.noise.mul, self.clap.noise2.mul = 0, 0

    def playthrough_seq(self):
        if self.pauses_remaining > 0:
            self.pauses_remaining -= 1
            self.thump.play()

            if self.pauses_remaining == 4 and self.temp_drum_count > 2:
                self.rev_snare.play(1)
            return

        sound1 = self.cv
        sound2 = self.cv2

        if self.changed_sound:
            sound1 = self.cvalt
            sound2 = self.cv2alt
        elif self.finished:
            sound1 = self.cvfin
            sound2 = self.cv2fin

        if self.index < self.max_playback:
            sound1.change_freq(midiToHz(self.note_sequence[self.index] + 81))
            sound2.change_freq(midiToHz(self.note_sequence[self.index] + 69))
            self.clap.set_freq(midiToHz(self.note_sequence[self.index] + 93))

            if len(self.initial_sequence) > (self.index + 1):
                new_start = self.initial_sequence[self.index]
                sound1.set_start(new_start)
                sound2.set_start(new_start)
            else:
                if random.getrandbits(1) == 1:
                    new_start = random.uniform(0, 3)
                    sound1.set_start(new_start)
                    sound2.set_start(new_start)
                self.initial_sequence.append(sound1._looper.start)
            self.play_cv()
            self.index += 1

            if (self.max_playback > self.pattern.taps):
                if random.getrandbits(1) == 1:
                    new_start = random.uniform(0, 3)
                    sound1.set_start(new_start)
                    sound2.set_start(new_start)
                if (self.index == self.pattern.taps):
                    self.index = 0
        else:
            self.pattern.stop()

if __name__ == "__main__":
    s= Server(buffersize=512).boot()
    cg = ChoppedGen()
    f = Freeverb(cg.get_pyoObj()).out()
    f.ctrl()


    s.gui(locals())