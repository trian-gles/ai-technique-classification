import requests
import io
import sounddevice
import numpy as np

url = 'https://timeout2-ovo53lgliq-uc.a.run.app'
# method : 'POST'
# body : let formData = new FormData();
#   formData.append('file', new Blob([editor.getValue('\n')], {type : 'text/plain'}), 'file.sco');

score_str = """
load("GRANSYNTH")
   
dur = 30
   
amp = maketable("line", 1000, 0,0, 1,1, 2,0.5, 3,1, 4,0)
wave = maketable("wave", 2000, 1, .5, .3, .2, .1)
granenv = maketable("window", 2000, "hanning")
hoptime = maketable("line", "nonorm", 1000, 0,0.01, 1,0.002, 2,0.05)
hopjitter = 0.0001
mindur = .04
maxdur = .06
minamp = maxamp = 1
pitch = maketable("line", "nonorm", 1000, 0,6, 1,9)
transpcoll = maketable("literal", "nonorm", 0, 0, .02, .03, .05, .07, .10)
pitchjitter = 1
   
st = 0
GRANSYNTH(st, dur, amp*7000, wave, granenv, hoptime, hopjitter,
   mindur, maxdur, minamp, maxamp, pitch + webpitch*0.01, transpcoll, pitchjitter, 14, 0, 0)
   
st = st+0.14
pitch = pitch+0.002
GRANSYNTH(st, dur, amp*7000, wave, granenv, hoptime, hopjitter,
   mindur, maxdur, minamp, maxamp, pitch + webpitch*0.01, transpcoll, pitchjitter, 21, 1, 1)
"""

def webrtc_request(score_str: str) -> np.ndarray:
    files = {"file": ('text/plain', score_str.encode('utf-8'), "file.sco"), 'pitch': (None, 48)}
    request = requests.post(url=url, files=files)
    nparr = np.frombuffer(request.content, dtype=int)
    return nparr

def play_np(nparr: np.ndarray):
    sounddevice.play(nparr)
    sounddevice.wait()

if __name__ == "__main__":
    play_np(webrtc_request(score_str))