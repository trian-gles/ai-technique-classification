
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Lock, Process, Queue, current_process
import queue
import librosa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # use GPU instead of AVX


def get_waveform(audio, tf):
    tf.convert_to_tensor(audio)

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


def prep_data(note, tf):
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


def identification_process(unidentified_notes: Queue, identified_notes: Queue):
    import tensorflow as tfp
    techniques = []
    print(f"Starting process {current_process().name}")
    model = tfp.keras.models.load_model("savedModel")
    print(f"Model loaded for {current_process().name}")
    while True:
        try:
            note = unidentified_notes.get_nowait()
        except queue.Empty:
            print(f"Queue empty, {current_process().name} exiting")
            break
        else:

            ds = prep_data(note, tfp)
            for spectrogram in ds.batch(1):
                print(current_process().name + f" executing identification")
                spectrogram = tfp.squeeze(spectrogram, axis=1)
                prediction = model(spectrogram)
                parsed_pred = parse_result(prediction, tfp)
                identified_notes.put(parsed_pred)
    return True


def main():
    number_of_processes = 3
    processes = []

    dir_path = os.path.dirname(os.path.realpath(__file__))

    sample_files = [f"samples/manual/smack/smack1{i}.wav" for i in range(2, 9)]
    unidentified_notes = Queue()
    identified_notes = Queue()

    for f in sample_files:
        print(f)
        unidentified_notes.put(librosa.load(f)[0])

    for w in range(number_of_processes):
        p = Process(target=identification_process, args=(unidentified_notes, identified_notes))
        processes.append(p)
        p.start()
    cur_time = time.time()
    for p in processes:
        p.join()
    print("All processes finished")
    finished_notes = []
    while not identified_notes.empty():
        finished_notes.append(identified_notes.get())
    print(f"Total time : {time.time() - cur_time}")

    for n in finished_notes:
        print(n)

if __name__ == "__main__":
    main()