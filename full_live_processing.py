import numpy as np
from multiprocessing import Process, Queue, Value
from utilities.utilities import TECHNIQUES, int_to_string_results

from sub_processes.buffer_split import SplitNoteParser
from sub_processes.identify_note import identification_process
from pyo import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # use GPU instead of AVX


def main():
    number_of_processes = 1
    buffer_length = 2  # value in seconds
    sr = 22050


    ###### Set up shared information ######
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    unidentified_notes = Queue()  # stores waveforms of prepared notes that must be identified
    identified_notes = Queue()  # stores arrays of ints indicating the most to least likely technique
    ready_count = Value('i', 0)  # track how many processes are ready
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start


    ###### Create objects for individual processes ######
    parser = SplitNoteParser(buffer_excerpts, unidentified_notes, finished)

    ###### Set up PYO #######  THIS SHOULD GET IT'S OWN FILE AND FUNCTION
    s = Server(sr=sr)
    s.boot()
    t = NewTable(length=buffer_length)
    inp = Input()
    rec = TableRec(inp, table=t).play()
    osc = Osc(table=t, freq=t.getRate(), mul=0.5).out()  # simple playback

    def send_buf_for_analysis():
        #print("Sending out new buffer")
        print("\n")
        np_arr = np.array(t.getTable())
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
    note_split = Process(target=parser.mainloop)
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
            result_dict = identified_notes.get()
            str_results = int_to_string_results(result_dict["prediction"], TECHNIQUES)
            print(f"Identified note: {str_results[:3]}")
            identified_notes_count += 1
        if ready_to_quit:
            finished.value = 1
            break

    print(f"Total time : {time.time() - cur_time}, notes identified = {identified_notes_count}")


if __name__ == "__main__":
    main()
