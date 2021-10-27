import os
from sub_processes.audio_process import audio_server
from sub_processes.buffer_split import SplitNoteParser
from multiprocessing import Process, Queue, Value
import numpy as np
import queue
from utilities.analysis import note_above_threshold
import soundfile




def save_process(unidentified_notes: Queue):
    num_notes = 0
    tech_name = "TEST"
    dir_path = os.path.join("samples/manual", tech_name)
    try:
        os.mkdir(dir_path)
    except OSError:
        print("Technique has already been added.  Appending new training files.")
    while True:
        try:
            note: np.ndarray = unidentified_notes.get_nowait()
        except queue.Empty:
            continue
        if note_above_threshold(note):
            soundfile.write(f"samples/manual/{tech_name}/{tech_name}_{num_notes}.wav", note, 44100)
            num_notes += 1


def main():
    number_of_processes = 1


    ###### Set up shared information ######
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    unidentified_notes = Queue()  # stores waveforms of prepared notes that must be identified

    wav_responses = Queue()  # wav files to be played back by the audio server
    ready_count = Value('i', 0)  # track how many processes are ready
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start


    ###### Create objects for individual processes ######
    parser = SplitNoteParser(buffer_excerpts, unidentified_notes, ready, finished)

    ###### Start all the processes ######
    print("Loading tensorflow models...")

    print("Starting note split...")
    note_split = Process(target=parser.mainloop)
    note_split.start()

    saving = Process(target=save_process, args=(unidentified_notes,))
    saving.start()

    print("All processes ready, initiating audio")
    ready.value = 1
    audio_server(buffer_excerpts, wav_responses, ready, finished)


if __name__ == "__main__":
    main()