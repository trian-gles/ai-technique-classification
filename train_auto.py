from pyo import *
import soundfile
import numpy as np
import os
from utilities import find_onsets
import librosa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # use GPU instead of AVX

techniques = os.listdir("samples/manual")

buffer_length = 2

s = Server()
s.setInputDevice(2)
s.boot()
t = NewTable(length=buffer_length)
inp = Input(0)
rec = TableRec(inp, table=t).play()

finished_recording = False
notes = []
saved_buffer = np.array([0])

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



def send_buf_for_analysis():
    global finished_recording
    global saved_buffer
    finished_recording = True
    narr = np.array(t.getTable())
    saved_buffer = narr



trigF = TrigFunc(rec["trig"], send_buf_for_analysis)
s.start()
while not finished_recording:
    pass



s.stop()
import tensorflow as tf
model = tf.keras.models.load_model("savedModel")

notarr, sr = librosa.load("samples/sorted/smacks#04.wav")
notarr = notarr[sr * 6:sr * 8]

print("Recorded")
print(saved_buffer)
print("saved")
print(notarr)

def predict_buffer(buf):
    onsets = find_onsets(buf, 22050)
    for i in range(len(onsets))[1:-2]:
        notes.append(buf[onsets[i]:onsets[i+1]])

    for note in notes:
        ds = prep_data(note, tf)
        for spectrogram in ds.batch(1):
            spectrogram = tf.squeeze(spectrogram, axis=1)
            prediction = model(spectrogram)
            parsed_pred = parse_result(prediction, tf)
            print(parsed_pred)

            str_results = list(map(lambda i: techniques[i], parsed_pred))
            str_results = str_results[0:-1]
            str_results.reverse()
            print(f"New identified note : {str_results}")

predict_buffer(saved_buffer)
predict_buffer(notarr)