rtsetparams(44100, 2)
load("MCLAR")
load("FREEVERB")

bus_config("MCLAR", "aux 0 out")
bus_config("FREEVERB", "aux 0 in", "out 0-1")


srand(27)

st = 0
pitch = irand(59, 71)
breath = 1

env = maketable("line", 1000, 0, 1, 1, 0.6, 2, 0.6, 3, 0)

for(i = 0; i < irand(10, 15); i = i + 1)
{
	MCLAR(st, 0.20, 20000 * env, cpsmidi(pitch), 0.2, 0.7, 0.5, 0.5, breath)
	pitch += -(2 + rand() / 5) + ((i + 1) % 2) * (rand() + 1) * 9
	breath = breath - 0.03
	st += 0.12
}

outskip = 0
inskip = 0
dur = st + 2
amp = .2
roomsize = maketable("line", 1000, 0, 0.7, 1, 1)
predelay = .03
ringdur = 3
damp = 70
dry = 40
wet = 40
width = 100
   
 
FREEVERB(outskip, inskip, dur, amp, roomsize, predelay, ringdur, damp, dry, wet, width)
