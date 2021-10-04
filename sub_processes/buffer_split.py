import numpy as np
from multiprocessing import Process, Queue, Value
import queue
from utilities.utilities import find_onsets, note_above_threshold
import soundfile
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # use GPU instead of AVX

####
# NEEDS TO LIMIT BUFFERS TO 16000 SAMPLES, OTHERWISE SEND "SILENCE"
####
class SplitNoteParser:
    def __init__(self, buffer_excerpts: Queue, unidentified_notes: Queue, finished: Value):
        self.leftover_buf = np.ndarray([])
        self.lb_empty = True
        self.finished = finished
        self.new_buf = np.ndarray([])
        self.buffer_excerpts = buffer_excerpts
        self.unidentified_notes = unidentified_notes

    def mainloop(self):
        while not self.finished.value == 1:
            try:
                buf_excerpt: np.ndarray = self.buffer_excerpts.get_nowait()
            except queue.Empty:
                continue
            else:
                self._parse_buffer(buf_excerpt)
        return True



    def _parse_buffer(self, new_buf: np.ndarray):
        note_ons = find_onsets(new_buf, 22050)

        if len(note_ons) == 0: # there are no onsets despite being loud enough
            self._add_to_lb(new_buf)
        else:
            buf_pieces = np.split(new_buf, note_ons)
            self._add_to_lb(buf_pieces[0]) # add the first piece to the leftover

            for note in buf_pieces[1:]:
                if len(note) == 0: # Sometimes librosa spits out simultaneous onsets
                    continue
                if self._check_note_quality(note):
                    self._send_lb()
                self._add_to_lb(note)

    def _check_note_quality(self, note: np.ndarray):
        """Checks if the note is loud enough and the buffer is long enough"""
        thresh = note_above_threshold(note)
        try:
            long_enough = len(self.leftover_buf) > 2048
        except TypeError:
            long_enough = False
        return thresh and long_enough

    def _add_to_lb(self, not_note: np.ndarray):
        try:
            self.leftover_buf = np.concatenate([self.leftover_buf, not_note])
        except ValueError:
            self._set_lb(not_note)
        self._check_length()

    def _set_lb(self, not_note: np.ndarray):
        self.leftover_buf = not_note

    def _empty_lb(self):
        self.leftover_buf = np.ndarray([])

    def _check_length(self):
        """If the current buffer is too long, just send it"""
        if len(self.leftover_buf) > 16000:
            #print("Leftover buf exceeded maximum")
            self._send_lb()

    def _send_lb(self):
        self.unidentified_notes.put(self.leftover_buf)
        self._empty_lb()



def main():
    from pyo import Server, NewTable, Input, TableRec, TrigFunc
    import time
    buffer_length = 2  # value in seconds
    sr = 22050

    global debug_record_file
    debug_record_file = np.array([0])

    ###### Set up shared information ######
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    unidentified_notes = Queue()  # stores waveforms of prepared notes that must be identified
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start

    parser = SplitNoteParser(buffer_excerpts, unidentified_notes, finished)


    ###### Set up PYO #######
    s = Server(sr = sr)
    s.boot()
    t = NewTable(length=buffer_length)
    inp = Input()
    rec = TableRec(inp, table=t).play()
    #osc = Osc(table=t, freq=t.getRate(), mul=0.5).out()  # simple playback

    def send_buf_for_analysis():
        #print("Sending out new buffer")
        global debug_record_file
        print("\n")
        np_arr = np.array(t.getTable())
        buffer_excerpts.put(np_arr)
        debug_record_file = np.concatenate([debug_record_file, np_arr])
        rec.play()

    tf = TrigFunc(rec["trig"], send_buf_for_analysis)


    ###### Start all the processes ######
    print("Starting note split...")
    note_split = Process(target=parser.mainloop)
    note_split.start()

    cur_time = time.time()
    print("All processes ready")
    ready.value = 1

    identified_notes_count = 0
    ready_to_quit = False

    ###### Finally run the PYO server ######
    s.start()

    while True:
        if not unidentified_notes.empty():
            new_note: np.ndarray = unidentified_notes.get()
            print("Saving new file of note")
            soundfile.write(f"test_unclassified_notes/{identified_notes_count}.wav", new_note, sr)
            identified_notes_count += 1
            if identified_notes_count == 10:
                break
        if ready_to_quit:
            finished.value = 1
            break


    soundfile.write("../test_unclassified_notes/full_recording.wav", debug_record_file, sr)


if __name__ == "__main__":
    main()