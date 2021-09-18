from pyo import *
import utilities
import numpy as np
import multiprocessing as mp
import ctypes
import time


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
    cur_time = time.time()
    np_arr = np.frombuffer(tup[0].get_obj())
    np_arr[:] = np.array(tup[1].getBuffer())[:] # copy the buffer into the new np array.  at the moment, this will drop notes at the beginning and end of the buffer
    p = mp.Process(target=sub_process, args = (tup[0],))
    p.start()
    tup[2].play()
    print(f"Time required for analysis = {time.time() - cur_time}")


def sub_process(buf):
    with buf:
        np_arr: np.ndarray = np.frombuffer(buf.get_obj())
        copied_arr = np_arr.copy()
        del np_arr
    onsets: np.ndarray = utilities.find_onsets(copied_arr, 22050)
    print(f"new onsets in sub process: {onsets}")


if __name__ == "__main__":
    main()