
import os
import numpy as np
import time
from multiprocessing import Process, Queue, current_process, Value
import queue
import librosa
from utilities.utilities import find_onsets, TECHNIQUES, plot_prediction, numpy_to_tfdata, prediction_to_int_ranks


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # use GPU instead of AVX


def identification_process(unidentified_notes: Queue, identified_notes: Queue,
                           ready_count: Value, finished: Value, ready: Value):
    """Subprocess which will classify notes in unidentified_notes and place them in identified_notes"""
    #  I should use time to make sure this ALWAYS lasts the same amount of time
    import tensorflow as tfp
    print(f"Starting process {current_process().name}")
    model = tfp.keras.models.load_model("savedModel")
    print(f"Model loaded for {current_process().name}")
    ready_count.value += 1
    while ready.value == 0:  # wait for all processes to be ready
        pass
    while True:
        if finished.value == 1:  # the main process says it's time to quit
            break
        try:
            note = unidentified_notes.get_nowait()
        except queue.Empty:
            pass
        else:
            print(note)
            ds: tfp.data.Dataset = numpy_to_tfdata(note, tfp)
            for spectrogram in ds.batch(1):
                print(current_process().name + f" executing identification")
                spectrogram = tfp.squeeze(spectrogram, axis=1)
                prediction = model(spectrogram)
                parsed_pred = prediction_to_int_ranks(prediction, tfp)
                identified_notes.put(prediction)
    return True


def note_split_process(buffer_excerpts: Queue, unidentified_notes: Queue, finished: Value):
    """Subprocess which extracts notes from buffer_excerpts and places them in unidentified_notes"""
    print("Starting note split process")
    leftover_buf = np.ndarray([0])
    while not finished.value == 1:
        try:
            buf_excerpt: np.ndarray = buffer_excerpts.get_nowait()
        except queue.Empty:
            pass
        else:
            print(buf_excerpt)
            onsets = find_onsets(buf_excerpt, 22050)
            first_onset = onsets[0]
            first_note = np.concatenate([leftover_buf, buf_excerpt[:first_onset]])  #use the leftover from the last buffer
            unidentified_notes.put(first_note)
            for i, onset in enumerate(onsets[1:-2]):  #Don't do this for the first or last onset
                start = onset
                finish = onsets[i + 2]  # +2 because of the funny indexing
                new_note = buf_excerpt[start:finish]
                unidentified_notes.put(new_note)
    return True




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


    ###### Split the sample into 2 second chunks ######
    # print("Splitting huge audio file...")
    # sample_wav, _ = librosa.load(sample_file, sr)
    # total_buffers = (len(sample_wav) // sr) + 1
    # print(total_buffers)
    # for i in range(total_buffers):
    #     buf_excerpt = sample_wav[i * sr * buffer_length: (i + 1) * sr * buffer_length]
    #     buffer_excerpts.put(buf_excerpt)

    test_basefiles = os.listdir("../test_sampls")
    test_files = [os.path.join("../test_sampls", bf) for bf in test_basefiles]
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

    note_split = Process(target=note_split_process, args=(buffer_excerpts, unidentified_notes, finished))
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
            results.append(identified_notes.get())
            # str_results = list(map(lambda i: techniques[i], int_results))
            # str_results = str_results[0:-1]
            # str_results.reverse()
            print(f"New identified note in main process")
            identified_notes_count += 1
        if ready_to_quit:
            finished.value = 1
            break
    import tensorflow as tf
    for pred in results:
        plot_prediction(TECHNIQUES, pred, tf)

    print(f"Total time : {time.time() - cur_time}, notes identified = {identified_notes_count}")

if __name__ == "__main__":
    main()