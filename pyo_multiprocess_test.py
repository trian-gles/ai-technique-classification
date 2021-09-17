import os
import numpy as np
import time
from multiprocessing import Process, Queue, current_process, Value
import queue
import librosa
from utilities import find_onsets
from pyo import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # use GPU instead of AVX


def get_waveform(audio, tf):
    audio = tf.convert_to_tensor(audio)

    tf.cast(audio, tf.float32)
    return audio


def get_spectrogram(waveform, tf):
    if tf.shape(waveform) > 16000:
        waveform = waveform[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)  # fix this so the padding isn't huge

    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    return spectrogram


def prep_data(note: np.ndarray, tf):
    """Turn a numpy buffer note into a tensorflow dataset"""
    waveform = get_waveform(note, tf)
    spec = get_spectrogram(waveform, tf)
    ds = tf.data.Dataset.from_tensors([spec])
    return ds


def parse_result(prediction, tf):
    sftmax = tf.nn.softmax(prediction[0])
    sorted = np.sort(sftmax)[::-1]
    index_of = lambda x: np.where(sftmax == x)[0][0]
    prediction_ranks = list(map(index_of, sorted))
    return prediction_ranks


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

            ds = prep_data(note, tfp)
            for spectrogram in ds.batch(1):
                print(current_process().name + f" executing identification")
                spectrogram = tfp.squeeze(spectrogram, axis=1)
                prediction = model(spectrogram)
                parsed_pred = parse_result(prediction, tfp)
                identified_notes.put(parsed_pred)
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
            print("New buffer identified to split")
            onsets = find_onsets(buf_excerpt, 22050)
            if len(onsets) == 0:
                print("buffer contains no onsets")
                continue
            first_onset = onsets[0]
            first_note = np.concatenate(
                [leftover_buf, buf_excerpt[:first_onset]])  # use the leftover from the last buffer
            unidentified_notes.put(first_note)
            for i, onset in enumerate(onsets[1:-2]):  # Don't do this for the first or last onset
                start = onset
                finish = onsets[i + 2]  # +2 because of the funny indexing
                new_note = buf_excerpt[start:finish]
                unidentified_notes.put(new_note)
    return True


def main():
    techniques = os.listdir("samples/manual")
    print(techniques)
    number_of_processes = 3
    buffer_length = 2  # value in seconds
    sr = 22050


    ###### Set up shared information ######
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    unidentified_notes = Queue()  # stores waveforms of prepared notes that must be identified
    identified_notes = Queue()  # stores arrays of ints indicating the most to least likely technique
    ready_count = Value('i', 0)  # track how many processes are ready
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start


    ###### Set up PYO #######
    s = Server()
    s.setInputDevice(9)
    s.boot()
    t = NewTable(length=buffer_length)
    inp = Input()
    rec = TableRec(inp, table=t).play()

    def send_buf_for_analysis():
        print("Sending out new buffer")
        np_arr = np.frombuffer(t.getBuffer())
        buffer_excerpts.put(np_arr)
        rec.play()

    tf = TrigFunc(rec["trig"], send_buf_for_analysis)


    ###### Start all the processes ######
    print("Loading tensorflow models...")
    processes = []
    for w in range(number_of_processes):
        p = Process(target=identification_process,
                    args=(unidentified_notes, identified_notes, ready_count, finished, ready))
        processes.append(p)
        p.start()

    print("Starting note split...")
    note_split = Process(target=note_split_process, args=(buffer_excerpts, unidentified_notes, finished))
    note_split.start()

    while ready_count.value != number_of_processes:
        pass
    cur_time = time.time()
    print("All processes ready")
    ready.value = 1

    identified_notes_count = 0
    ready_to_quit = False

    ###### Finally run the PYO server ######
    s.start()

    while True:
        if not identified_notes.empty():
            str_results = list(map(lambda i: techniques[i], identified_notes.get()))
            str_results = str_results[0:-1]
            str_results.reverse()
            print(f"New identified note in main process: {str_results}")
            identified_notes_count += 1
        if ready_to_quit:
            finished.value = 1
            break

    print(f"Total time : {time.time() - cur_time}, notes identified = {identified_notes_count}")


if __name__ == "__main__":
    main()
