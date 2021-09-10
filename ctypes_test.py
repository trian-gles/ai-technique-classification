import ctypes

testlib = ctypes.CDLL('libs/testlib.so')
testlib.myprint()