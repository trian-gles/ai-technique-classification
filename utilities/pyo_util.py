import numpy as np

def fill_tab_np(arr: np.ndarray, table):
    """TODO: make this stereo"""
    table_arr = np.asarray(table.getBuffer())
    print(table_arr.dtype)
    print(arr.dtype)

    length = len(arr)
    table_arr[:length] = arr[:, 0]