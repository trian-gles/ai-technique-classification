from pyo import *
import librosa
import numpy as np
import multiprocessing as mp
import ctypes


def main():
    buffer_size = 2

    s = Server().boot()
    sr = int(s.getSamplingRate())
    t = NewTable(length=buffer_size)

    inp = Input()

    rec = TableRec(inp, table=t).play()
    buffer_size = 2
    buf = mp.Array(ctypes.c_float, buffer_size * sr * 2)




    tf = TrigFunc(rec["trig"], init_analysis_sub, arg=(buf, t, rec))

    s.gui(locals())


def init_analysis_sub(tup): # because trigfunc only can send one arg, I use a tuple
    np_arr = np.frombuffer(tup[0].get_obj())
    np_arr[:] = np.array(tup[1].getBuffer())[:] # copy the buffer into the new np array.  at the moment, this will drop notes at the beginning and end of the buffer
    p = mp.Process(target=sub_process, args = (tup[0],))
    p.start()
    tup[2].play()


def sub_process(buf):
    with buf:
        np_arr: np.ndarray = np.frombuffer(buf.get_obj())
        copied_arr = np_arr.copy()
        del np_arr
    onsets: np.ndarray = librosa.onset.onset_detect(copied_arr, sr=44100, backtrack=True, normalize=False) # I need to set up an envelope equation in advance
    print(f"new onsets in sub process: {onsets}")


if __name__ == "__main__":
    main()




