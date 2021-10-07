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

score_str = """
load("STRUM2");

max_dur = 30;

tempo = 120;

struct StrumNote
{
    float rest,
    float len,
    float midi_pitch,
    float next_index
}

notes = {}


float make_note(float rest, float len, float midi_pitch, float index, float next_index, list notes)
{
    struct StrumNote note;
    note.rest = rest;
    note.len = len;
    note.midi_pitch = midi_pitch;
    note.next_index = next_index;
    notes[index] = note;
    return 0;
}

float play_note(list notes, float cur_time, float index)
{
    note = notes[index];
    freq = cpsmidi(note.midi_pitch);
    rest_length = note.rest * (60 / tempo);
    length = note.len * (60 / tempo);
    //STRUM2(cur_time + rest_length, length, 20000, freq, 10, length, 0.5);
    
    cur_time = cur_time + rest_length + length;
    
    if ((cur_time < 20) && (note.next_index >= 0))
    {
        play_note(notes, cur_time, note.next_index);
    }
    
    return 0;
}

make_note(0, 2, 61, 0, 1, notes);
make_note(0, 4, 65, 1, 0, notes);

play_note(notes, 0, 0);

"""


score_str2 = """
load("STRUM2");

max_dur = 30;
tempo = 120;



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
    freq = cpsmidi(note.midi_pitch); 
    STRUM2(note.start_time, note.dur, 20000, freq, 10, note.dur, 0.5);
    return 0;
}

////////////////
//// CURSOR ////
////////////////

all_cursors = {}
new_cursors = {}

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
make_note_node(2, 2.2, 99, -1, -1)




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