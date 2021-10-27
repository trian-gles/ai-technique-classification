import numpy as np
from multiprocessing import Process, Queue, Value
from sub_processes.buffer_split import SplitNoteParser
from sub_processes.identify_note import identification_process
from sub_processes.audio_process import audio_server
import os
import queue
from utilities.analysis import TECHNIQUES, int_to_string_results


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # use GPU instead of AVX


def list_predictions(identified_notes: Queue):
    while True:
        note_dict = None
        try:
            note_dict = identified_notes.get_nowait()
        except queue.Empty:
            continue
        str_results = int_to_string_results(note_dict["prediction"], TECHNIQUES)
        print(f"New note: {str_results[0:3]}")


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


    ###### Create objects for individual processes ######
    parser = SplitNoteParser(buffer_excerpts, unidentified_notes, ready, finished)

    ###### Start all the processes ######
    print("Loading tensorflow models...")
    processes = []
    for w in range(number_of_processes):
        p = Process(target=identification_process,
                    args=(unidentified_notes, identified_notes, ready_count, finished, ready))
        processes.append(p)
        p.start()

    print("Starting note split...")
    note_split = Process(target=parser.mainloop)
    note_split.start()

    print("Starting AI...")
    ai = Process(target=list_predictions, args=(identified_notes,))
    ai.start()




    while ready_count.value != number_of_processes:
        pass
    print("All processes ready, initiating audio")
    ready.value = 1
    audio_server(buffer_excerpts, wav_responses, ready, finished)


if __name__ == "__main__":
    main()
