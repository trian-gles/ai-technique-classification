from pyo import *
from utilities.pyo_util import fill_tab_np
from webrtcmix.web_request import webrtc_request, score_str
sr = 44100
s = Server(sr=sr).boot()

playback = DataTable(size = sr * 80)
wav = webrtc_request(score_str)
print("filling table")
fill_tab_np(wav, playback)
print("table filled")
a = Osc(table=playback, freq=playback.getRate(), mul=0.4).out()
s.start()
while True:
    pass