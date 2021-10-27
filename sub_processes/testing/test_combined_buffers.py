import numpy as np
from multiprocessing import Process, Queue, Value
from sub_processes.audio_process import audio_server
import os
import queue

def test_recombined_buffers(buffer_excerpts: Queue, ready: Value, finished: Value):
    import soundfile
    num_bufs = 0
    while ready.value == 0:  # wait for all processes to be ready
        pass
    while True:
        if finished.value == 1:  # the main process says it's time to quit
            break
        try:
            buf: np.ndarray = buffer_excerpts.get_nowait()
        except queue.Empty:
            continue
        if num_bufs == 0:
            full_buf = buf
        else:
            full_buf = np.concatenate((full_buf, buf))
        num_bufs += 1
        if num_bufs == 100:
            soundfile.write(f"test_buffers/buf.wav", full_buf, 44100)



def main():
    number_of_processes = 1


    ###### Set up shared information ######
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    unidentified_notes = Queue()  # stores waveforms of prepared notes that must be identified
    identified_notes = Queue()  # stores arrays of ints indicating the most to least likely technique
    wav_responses = Queue()  # wav files to be played back by the audio server
    ready_count = Value('i', 0)  # track how many processes are ready
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start


    buffer_save = Process(target=test_recombined_buffers, args=(buffer_excerpts, ready, finished))
    buffer_save.start()

    print("All processes ready, initiating audio")
    ready.value = 1
    audio_server(buffer_excerpts, wav_responses, ready, finished)


if __name__ == "__main__":
    main()