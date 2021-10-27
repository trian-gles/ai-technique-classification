import numpy as np
from multiprocessing import Process, Queue, Value
from sub_processes.buffer_split import SplitNoteParser
from sub_processes.audio_process import audio_server
import queue


def test_split_buffers(unidentified_notes: Queue, ready: Value, finished: Value):
    import soundfile
    import librosa
    num_notes = 0
    while ready.value == 0:  # wait for all processes to be ready
        pass
    while True:
        if finished.value == 1:  # the main process says it's time to quit
            break
        try:
            note: np.ndarray = unidentified_notes.get_nowait()
        except queue.Empty:
            continue
        soundfile.write(f"test_unidentified_notes/note_{num_notes}.wav", note, 22050)
        num_notes += 1

def listen_split_buffers(unidentified_notes: Queue, ready: Value, finished: Value):
    import soundfile
    import librosa
    num_notes = 0
    while ready.value == 0:  # wait for all processes to be ready
        pass
    while True:
        if finished.value == 1:  # the main process says it's time to quit
            break
        try:
            note: np.ndarray = unidentified_notes.get_nowait()
        except queue.Empty:
            continue
        soundfile.write(f"test_unidentified_notes/note_{num_notes}.wav", note, 22050)
        num_notes += 1



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

    print("Starting note split...")
    note_split = Process(target=parser.mainloop)
    note_split.start()


    buffer_save = Process(target=test_split_buffers, args=(unidentified_notes, ready, finished))
    buffer_save.start()

    print("All processes ready, initiating audio")
    ready.value = 1
    audio_server(buffer_excerpts, wav_responses, ready, finished)


if __name__ == "__main__":
    main()
