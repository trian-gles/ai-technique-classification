from pyo_presets.chopped_generative2 import ChoppedGen
from pyo_presets.huge_bass import StereoBass

from pyo import *

s= Server(buffersize=512).boot()
cg = ChoppedGen()


bass = StereoBass()

notes = Notein(poly=1, scale=1, mul=.5)
notes.keyboard()


def change_pitch():
    pit = int(notes["pitch"].get(all=True)[0])
    cg.query_trans(round(hzToMidi(pit)))
    bass.set_notes(pit)


note_change_trig = TrigFunc(notes['trigon'], change_pitch)
f = Freeverb(cg.get_pyoObj() + bass.get_voices()).out()



s.gui(locals())