def main():
    print("Loading libraries")
    from multiprocessing import Process, Queue, Value
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # use GPU instead of AVX
    print("Loading brain")
    from sub_processes.ai_response import Brain
    print("Loading note split")
    from sub_processes.buffer_split import SplitNoteParser
    print("Loading identification")
    from sub_processes.identify_note import identification_process
    print("Loading audio")
    from sub_processes.audio_process import audio_server
    print("Libraries imported")

    number_of_processes = 1


    ###### Set up shared information ######
    buffer_excerpts = Queue()  # contains 2 second snippets of buffer that needs to be split into notes
    unidentified_notes = Queue()  # stores waveforms of prepared notes that must be identified
    identified_notes = Queue()  # stores arrays of ints indicating the most to least likely technique
    wav_responses = Queue()  # wav files to be played back by the audio server
    other_actions = Queue() # Brain control of audio engine

    ready_count = Value('i', 0)  # track how many processes are ready
    finished = Value('i', 0)  # track how many processes are finished
    ready = Value('i', 0)  # signal to all subprocesses that we can start


    ###### Create objects for individual processes ######
    parser = SplitNoteParser(buffer_excerpts, unidentified_notes, ready, finished)
    brain = Brain(wav_responses, identified_notes, other_actions, ready, finished)

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

    while ready_count.value != number_of_processes:
        pass
    print("All processes ready, initiating audio")
    ready.value = 1
    audio_server(buffer_excerpts, wav_responses, other_actions, ready, finished)


if __name__ == "__main__":
    main()
