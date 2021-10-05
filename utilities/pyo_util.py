import numpy as np
from pyo import *

def fill_tab_np(arr: np.ndarray, table: DataTable):
    """TODO: make this resize properly"""
    #table_arr = np.asarray(table.getBuffer())
    #print(table_arr.shape)
    #print(arr.shape)
    #table.setSize(len(arr))
    samplist = [list(arr[:, 0]), list(arr[:, 1])]
    table.replace(samplist)
    #length = len(arr)
    #table_arr[:length] = arr[:, 0]