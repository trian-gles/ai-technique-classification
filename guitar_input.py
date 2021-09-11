from pyo import *
import librosa
import numpy as np

# this all needs to be threaded


s = Server().boot()

t = NewTable(length=2)

inp = Input()

rec = TableRec(inp, table=t).play()

first_buf_update = False
buffer = np.array([], dtype=np.float32)


def parse_note(arr: np.ndarray):
    print(arr)


def check_buffer_attacks():
    global buffer
    global first_buf_update
    onsets: np.ndarray = librosa.onset.onset_detect(buffer, sr=44100, backtrack=True)
    if len(onsets > 1):
        last_onset = onsets[len(onsets) - 1]


        parseable_buffer = buffer[:last_onset] # we want to save the last onset for when more data is added to the buffer

        if not first_buf_update:
            parse_note(parseable_buffer[:onsets[0]])

        first_buf_update = False

        for i, point in enumerate(onsets):
            if not  (point == last_onset):
                parse_note(parseable_buffer[point:onsets[i + 1]])

        buffer = buffer[:last_onset] # the buffer now starts at the last attack



def update_buffer():
    global buffer
    y = np.asarray(t.getBuffer())
    rec.play()
    buffer = np.concatenate((buffer, y))
    check_buffer_attacks()


tf = TrigFunc(rec["trig"], update_buffer)

s.gui(locals())




