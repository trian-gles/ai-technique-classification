import requests
import io
import sounddevice
import numpy as np
import soundfile as sf
import scipy.io.wavfile

url = 'https://timeout2-ovo53lgliq-uc.a.run.app'
# method : 'POST'
# body : let formData = new FormData();
#   formData.append('file', new Blob([editor.getValue('\n')], {type : 'text/plain'}), 'file.sco');

score_str1 = """
load("STRUM2");
load("FREEVERB")

srand(3)

max_dur = 30;
tempo = 120;

////////////
// REVERB //
////////////

bus_config("STRUM2", "aux 0 out")
bus_config("FREEVERB", "aux 0 in", "out 0-1")

outskip = 0
inskip = 0
dur = 60
amp = .5
roomsize = 0.6
predelay = .03
ringdur = 3
damp = 70
dry = 40
wet = 30
width = 100

FREEVERB(outskip, inskip, dur, amp, roomsize, predelay, ringdur, damp, dry, wet, width)



////////////////
////  NODE  ////
////////////////

node_storage = {}

struct StrumNoteNode
{
    float rest,
    float len,
    float midi_pitch,
    float left_index,
    float right_index
}

float make_note_node(float rest, float len, float midi_pitch, float left_index, float right_index)
{
    struct StrumNoteNode note;
    note.rest = rest;
    note.len = len;
    note.midi_pitch = midi_pitch;
    note.left_index = left_index;
    note.right_index = right_index;
    node_storage[len(node_storage)] = note;
    return 0;
}



////////////////
//// NOTES  ////
////////////////

scheduled_notes = {}
struct StrumNotePlay
{
    float start_time,
    float dur,
    float midi_pitch
}

float schedule_note(float start_time, float dur, float midi_pitch)
{
    struct StrumNotePlay note;
    note.start_time = start_time;
    note.dur = dur;
    note.midi_pitch = midi_pitch;

    scheduled_notes[len(scheduled_notes)] = note;
    return 0;
}

float play_note(struct StrumNotePlay note)
{
    freq = cpsmidi(note.midi_pitch) + rand();
    offset = (rand() + 1) / 4;
    amp = rand() * 8000 + 12000;
    STRUM2(note.start_time + offset, note.dur, amp, freq, 10, note.dur, (rand() + 1) / 2);
    STRUM2(note.start_time + ((rand() + 1) / 4), note.dur, amp, freq * 2, 10, note.dur, (rand() + 1) / 2);
    return 0;
}

////////////////
//// CURSOR ////
////////////////

all_cursors = {};
new_cursors = {};

list limit_cursors(float depth)
{
    active_cursors = {};
    for (i = 0; i < len(all_cursors); i = i + 1)
    {
        curs = all_cursors[i];
        if (curs.index >= 0)
        {
            active_cursors[len(active_cursors)] = curs;
        }
    }
    all_cursors = active_cursors
    
    if (depth == 16)
    {
        limited_cursors = {}
        for (i = 0; i < len(all_cursors); i = i + 1)
        {
            curs = all_cursors[i];
            if (rand() > 0)
            {
                limited_cursors[len(limited_cursors)] = curs
            }
        }
        
        all_cursors = {all_cursors[0]}
    }
    
    return all_cursors;
}

struct CursorStatus
{
    float index,
    float current_time
}

float new_cursor(float index, float current_time)
{
    struct CursorStatus ncurs;
    ncurs.index = index;
    ncurs.current_time = current_time;
    new_cursors[len(new_cursors)] = ncurs;
    return 0;
}

float schedule_and_get_next(struct CursorStatus cursor)
{
    if (cursor.index < 0)
    {
        return 0;
    }
    
    node = node_storage[cursor.index];
    rest_len = node.rest * 60 / tempo;
    note_len = node.len * 60 / tempo;
    current_time = rest_len + cursor.current_time;
    
    schedule_note(current_time, note_len, node.midi_pitch)
    
    
    //// The cursor stays on the left branch
    cursor.index = node.left_index;
    cursor.current_time = current_time;
    
    //// Make a new cursor for the right branch
    new_cursor(node.right_index, current_time)
    

    return 0;
}

//// Build the tree
make_note_node(1.12, 2, 40, 1, -1)
make_note_node(1.02, 2, 32, 2, 4)
make_note_node(1.04, 2, 33, 3, -1)
make_note_node(5, 4, 41, 0, 1)
make_note_node(20, 4, 27, 2, -1)




//// Traverse the tree
quit = 0
struct CursorStatus cursor;
cursor.index = 0;
cursor.current_time = 0;
depth = 0;

all_cursors[0] = cursor;

while (quit == 0)
{
    //// handle all current cursors
    for (i = 0; i < len(all_cursors); i = i + 1)
    {
        schedule_and_get_next(all_cursors[i])
    }
    
    //// add new cursors
    for (i = 0; i < len(new_cursors); i = i + 1)
    {
        all_cursors[len(all_cursors)] = new_cursors[i];
    }
    
    all_cursors = limit_cursors()
    
    if (depth > 18)
    {
        quit = 1;
    }
    new_cursors = {}
    depth = depth + 1;
}


//// Playback scheduled notes
for (i = 0; i < len(scheduled_notes); i = i + 1)
{
    play_note(scheduled_notes[i])
}

"""


score_str2 = """
load("STRUM2");
load("FREEVERB")

srand(3)

max_dur = 30;
tempo = 120;

////////////
// REVERB //
////////////

bus_config("STRUM2", "aux 0-1 out")
bus_config("FREEVERB", "aux 0-1 in", "out 0-1")

outskip = 0
inskip = 0
dur = 40
amp = .5
roomsize = 0.6
predelay = .03
ringdur = 3
damp = 70
dry = 40
wet = 30
width = 100

FREEVERB(outskip, inskip, dur, amp, roomsize, predelay, ringdur, damp, dry, wet, width)



////////////////
////  NODE  ////
////////////////

node_storage = {}

struct StrumNoteNode
{
    float rest,
    float len,
    float midi_pitch,
    float left_index,
    float right_index
}

float make_note_node(float rest, float len, float midi_pitch, float left_index, float right_index)
{
    struct StrumNoteNode note;
    note.rest = rest;
    note.len = len;
    note.midi_pitch = midi_pitch;
    note.left_index = left_index;
    note.right_index = right_index;
    node_storage[len(node_storage)] = note;
    return 0;
}



////////////////
//// NOTES  ////
////////////////

scheduled_notes = {}
struct StrumNotePlay
{
    float start_time,
    float dur,
    float midi_pitch
}

float schedule_note(float start_time, float dur, float midi_pitch)
{
    struct StrumNotePlay note;
    note.start_time = start_time;
    note.dur = dur;
    note.midi_pitch = midi_pitch;

    scheduled_notes[len(scheduled_notes)] = note;
    return 0;
}

float play_note(struct StrumNotePlay note)
{
    freq = cpsmidi(note.midi_pitch) + (rand() * note.start_time / 10);
    offset = (rand() + 1) / 4;
    amp = rand() * 8000 + 12000;
    STRUM2(note.start_time + offset, note.dur, amp, freq, 10, note.dur, (rand() + 1) / 2);
    if (note.midi_pitch < 40)
    {
        STRUM2(note.start_time + offset, note.dur, amp + 2000, freq / 2, 10, note.dur, (rand() + 1) / 2);
    }
    return 0;
}

////////////////
//// CURSOR ////
////////////////

all_cursors = {};
new_cursors = {};

list limit_cursors(float depth)
{
    active_cursors = {};
    for (i = 0; i < len(all_cursors); i = i + 1)
    {
        curs = all_cursors[i];
        if (curs.index >= 0)
        {
            active_cursors[len(active_cursors)] = curs;
        }
    }
    all_cursors = active_cursors
    
    if (depth == 15)
    {
        limited_cursors = {}
        for (i = 0; i < len(all_cursors); i = i + 1)
        {
            curs = all_cursors[i];
            if (rand() > 0)
            {
                limited_cursors[len(limited_cursors)] = curs
            }
        }
        
        all_cursors = limited_cursors
    }
    
    return all_cursors;
}

struct CursorStatus
{
    float index,
    float current_time
}

float new_cursor(float index, float current_time)
{
    struct CursorStatus ncurs;
    ncurs.index = index;
    ncurs.current_time = current_time;
    new_cursors[len(new_cursors)] = ncurs;
    return 0;
}

float schedule_and_get_next(struct CursorStatus cursor)
{
    if (cursor.index < 0)
    {
        return 0;
    }
    
    node = node_storage[cursor.index];
    rest_len = node.rest * 60 / tempo;
    note_len = node.len * 60 / tempo;
    current_time = rest_len + cursor.current_time;
    
    schedule_note(current_time, note_len, node.midi_pitch)
    
    
    //// The cursor stays on the left branch
    cursor.index = node.left_index;
    cursor.current_time = current_time;
    
    //// Make a new cursor for the right branch
    new_cursor(node.right_index, current_time)
    

    return 0;
}

//// Build the tree
make_note_node(1.12, 1, 60, 1, -1)
make_note_node(0.7, 1, 61, 2, 3)
make_note_node(1.51, .5, 68, 0, 4)
make_note_node(1.82, 2.2, 79, 2, 1)
make_note_node(7, 2.2, 99, 5, -1)
make_note_node(1, 4, 63, 6, -1)
make_note_node(1, 4, 68, 7, -1)
make_note_node(5, 4, 56, 8, -1)
make_note_node(5, 4, 49, 6, 9)
make_note_node(6, 4, 37, -1, 0)




//// Traverse the tree
quit = 0
struct CursorStatus cursor;
cursor.index = 0;
cursor.current_time = 0;
depth = 0;

all_cursors[0] = cursor;

while (quit == 0)
{
    //// handle all current cursors
    for (i = 0; i < len(all_cursors); i = i + 1)
    {
        schedule_and_get_next(all_cursors[i])
    }
    
    //// add new cursors
    for (i = 0; i < len(new_cursors); i = i + 1)
    {
        all_cursors[len(all_cursors)] = new_cursors[i];
    }
    
    all_cursors = limit_cursors(depth)
    
    if (depth > 18)
    {
        quit = 1;
    }
    new_cursors = {}
    depth = depth + 1;
}


//// Playback scheduled notes
for (i = 0; i < len(scheduled_notes); i = i + 1)
{
    play_note(scheduled_notes[i])
}

"""

def webrtc_request(score_str: str) -> np.ndarray:
    files = {"file": ('text/plain', score_str.encode('utf-8'), "file.sco"), 'pitch': (None, 48)}
    request = requests.post(url=url, files=files)
    bytesio = io.BytesIO(request.content) #
    nparr, sr = sf.read(bytesio, dtype='float32')
    print(nparr.shape)
    return nparr

def play_np(nparr: np.ndarray):
    print("Playing back wav file with sounddevice")
    sounddevice.play(nparr)
    sounddevice.wait()

if __name__ == "__main__":
    play_np(webrtc_request(score_str2))