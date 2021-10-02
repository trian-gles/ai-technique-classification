from multiprocessing import Queue, Value, current_process
from utilities.utilities import numpy_to_tfdata, prediction_to_int_ranks
import queue
import numpy as np

def identification_process(unidentified_notes: Queue, identified_notes: Queue,
                           ready_count: Value, finished: Value, ready: Value):
    """Subprocess which will classify notes in unidentified_notes and place them in identified_notes"""
    #  I should use time to make sure this ALWAYS lasts the same amount of time
    #  Test to playback the notes sent to this
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
            note: np.ndarray = unidentified_notes.get_nowait()
        except queue.Empty:
            continue
        else:
            #print(current_process().name + f" executing identification")
            #print(note)
            ds = numpy_to_tfdata(note, tfp)
            for spectrogram in ds.batch(1):

                spectrogram = tfp.squeeze(spectrogram, axis=1)
                prediction = model(spectrogram)
                parsed_pred = prediction_to_int_ranks(prediction, tfp)
                identified_notes.put(parsed_pred)
    return True