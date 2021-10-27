from random import randrange
import webrtcmix.web_request as wr

def guitar_partials_score(f1: float, f2: float, f3: float):
    print(f"Making rtc score with {f1, f2, f3}")
    return f"""
load("STRUM2")
load("FREEVERB")
total_dur = 10
srand({randrange(0, 20000)})
slow_down = .01

bus_config("STRUM2", "aux 0-1 out")
bus_config("FREEVERB", "aux 0-1 in", "out 0-1")

outskip = 0
inskip = 0

amp = .5
roomsize = 0.6
predelay = .03
ringdur = 3
damp = 70
dry = 40
wet = 30
width = 100


  
f1 = {f1}
f2 = {f2}
f3 = {f3}
freqs = {{ f1, f2, f3 }}

last_note_time = 0

float rand_pat(float freq, float total_dur, float minnum, float maxnum)
{{
	n = irand(minnum, maxnum)
	inc = total_dur / n

	start_time = inc
    
    
    
	for (j = 0; j < n; j += 1)
    {{
    	STRUM2(start_time, 1.4, 20000 / (j / 4 + 1), freq, 10, 1.4, (rand() + 1) / 2);
        inc += slow_down
      	start_time += inc
      	
      	if (start_time > last_note_time)
        {{
            last_note_time = start_time    
        }}
    }}

	return 0
}}


for (i = 0; i < len(freqs); i += 1)
{{
	rand_pat(freqs[i], total_dur, 12, 29)
}}

for (i = 0; i < len(freqs) - 1; i += 1)
{{
	inter_pitch = irand(freqs[i], freqs[i + 1])
    rand_pat(inter_pitch, total_dur, 3, 9)
}}

FREEVERB(outskip, inskip, last_note_time + 2, amp, roomsize, predelay, ringdur, damp, dry, wet, width)

"""

def funny_scale_score(f1: float):
    return f"""
load("STRUM2")
total_dur = 2


for (i = 0; i < 25; i = i + 1)
{{
    STRUM2(i / 10, total_dur, 20000, {f1} + i * 10, 5, total_dur / 2, 0.5);
}}
"""

if __name__ == "__main__":
    nparr = wr.webrtc_request(guitar_partials_score(230, 470, 3460))
    wr.play_np(nparr)