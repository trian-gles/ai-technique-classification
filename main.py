import numpy as np
from multiprocessing import Process, Queue, Value
from sub_processes.ai_response import Brain
from sub_processes.buffer_split import SplitNoteParser
from sub_processes.identify_note import identification_process
from sub_processes.audio_process import AudioServer
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # use GPU instead of AVX


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
    brain = Brain(wav_responses, identified_notes, ready, finished)

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
    ai = Process(target=brain.main)
    ai.start()

    print("Starting main audio loop...")
    audio = AudioServer(buffer_excerpts, wav_responses, ready, finished)

    while ready_count.value != number_of_processes:
        pass
    print("All processes ready, initiating piece")
    ready.value = 1
    audio.main()


if __name__ == "__main__":
    main()
