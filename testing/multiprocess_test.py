
import os
import numpy as np
import time
from multiprocessing import Process, Queue, current_process, Value
import queue
import librosa
from sub_processes.identify_note import identification_process
from sub_processes.buffer_split import SplitNoteParser
from utilities.analysis import find_onsets, TECHNIQUES, plot_prediction, numpy_to_tfdata, prediction_to_int_ranks


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # use GPU instead of AVX

def main():
    techniques = TECHNIQUES
    print(techniques)
    number_of_processes = 3
    buffer_length = 2  # value in seconds
    sr = 22050

    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    unidentified_notes = Queue()  # stores waveforms of prepared notes that must be identified
    identified_notes = Queue()  # stores arrays of ints indicating the most to least likely technique
    ready_count = Value('i', 0)  # track how many processes are ready
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start

    processes = []

    # make a set of buffers for testing

    test_basefiles = os.listdir("test_sampls")
    test_files = [os.path.join("test_sampls", bf) for bf in test_basefiles]
    total_notes = len(test_files)
    for f in test_files:
        sample_wav, _ = librosa.load(f, sr)
        buffer_excerpts.put(sample_wav)

    ###### Start all the processes ######
    for w in range(number_of_processes):
        p = Process(target=identification_process,
                    args=(unidentified_notes, identified_notes, ready_count, finished, ready))
        processes.append(p)
        p.start()
    parser = SplitNoteParser(buffer_excerpts, unidentified_notes, finished)

    note_split = Process(target=parser.mainloop)
    note_split.start()

    while ready_count.value != number_of_processes:
        pass
    cur_time = time.time()
    print("All processes ready")
    ready.value = 1



    identified_notes_count = 0
    ready_to_quit = False
    results = []

    while True:
        if identified_notes_count == total_notes:
            ready_to_quit = True
        if not identified_notes.empty():
            results.append(identified_notes.get()["prediction"])
            # str_results = list(map(lambda i: techniques[i], int_results))
            # str_results = str_results[0:-1]
            # str_results.reverse()
            print(identified_notes.get()["prediction"])
            identified_notes_count += 1
        if ready_to_quit:
            finished.value = 1
            break
    for pred in results:
        print(pred)

    print(f"Total time : {time.time() - cur_time}, notes identified = {identified_notes_count}")

if __name__ == "__main__":
    main()