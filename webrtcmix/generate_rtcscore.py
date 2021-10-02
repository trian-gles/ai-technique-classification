from random import randrange
import webrtcmix.web_request as wr

def guitar_partials_score(f1: float, f2: float, f3: float):
    return f"""
load("STRUM2")
total_dur = 10
srand({randrange(0, 20000)})
slow_down = .01
  
f1 = {f1}
f2 = {f2}
f3 = {f3}
freqs = {{ f1, f2, f3 }}

float rand_pat(float freq, float total_dur, float minnum, float maxnum)
{{
	n = irand(minnum, maxnum)
	inc = total_dur / n

	start_time = inc

	for (j = 0; j < n; j += 1)
    {{
    	STRUM2(start_time, 1.4, 20000 / (j / 4 + 1), freq, 10, 1.4, 0.5);
        inc += slow_down
      	start_time += inc
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
"""

if __name__ == "__main__":
    nparr = wr.webrtc_request(guitar_partials_score(230, 470, 3460))
    wr.play_np(nparr)