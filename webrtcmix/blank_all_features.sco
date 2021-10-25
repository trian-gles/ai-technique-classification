load("STRUM2");
load("FREEVERB")

/*
TODO -
make uncoor stop time larger
add transition between decays and harnesses

maybe a gliss in?
make it stop if last note ends before max_dur
make sure right exit is used on first pass
*/

srand(4)

max_dur =
tempo =
max_depth =
max_curs =
init_curs =
limit_depths =
limit_max_cursor_nums =
hardness =
upper_decay =
// relative to the decay time of each node

gliss_mode =  // none, uncoor, coor_stop
coor_stop_time =
// note_uncoor requires longer notes



////////////
// STRUM2 //
////////////

float get_hardness()
{
    return hardness + rand()
}



////////////
// GLISS  //
////////////

float make_note_slide_uncoor(float init_pitch, float start_time, float dur, float amp, float slide_start, float slide_len)
{
    if ((slide_start + slide_len) > dur)
    {
        STRUM2(start_time, dur, amp, init_pitch, get_hardness(), dur / upper_decay, (rand() + 1) / 2)
        return 0
    }

    pitch = maketable("line", "nonorm", "nointerp", 20 * dur, 0, init_pitch, slide_start, init_pitch, slide_start + slide_len, init_pitch * 4, dur, init_pitch * 4)
    STRUM2(start_time, dur, amp, pitch, get_hardness(), dur / upper_decay, (rand() + 1) / 2)
    return 0
}


// all slide at the same point in time, does not halt playback
float make_note_slide_coor(float init_pitch, float start_time, float dur, float amp, float abs_slide_start)
{
    rel_slide_start = abs_slide_start - start_time

    if ((rel_slide_start < 0) || (rel_slide_start > (dur - 1)))
    {
        STRUM2(start_time, dur, 20000, init_pitch, get_hardness(), dur / upper_decay, (rand() + 1) / 2)
        return 0
    }

    pitch = maketable("line", "nonorm", "nointerp", 20 * dur, 0, init_pitch, rel_slide_start, init_pitch, rel_slide_start + 1, init_pitch * 4, dur, init_pitch * 4)
    STRUM2(start_time, dur, amp, pitch, get_hardness(), dur / upper_decay, (rand() + 1) / 2)
    return 0
}

float make_note_slide_coor_stop(float init_pitch, float start_time, float dur, float amp, float abs_slide_start)
{
    rel_slide_start = abs_slide_start - start_time

    if (rel_slide_start < 0)
    {

    }
    else if (rel_slide_start > (dur - 1))
    {
        STRUM2(start_time, dur, amp, init_pitch, get_hardness(), dur / upper_decay, (rand() + 1) / 2)
    }
    else
    {
        pitch = maketable("line", "nonorm", "nointerp", amp, 0, init_pitch, rel_slide_start, init_pitch, rel_slide_start + 1, init_pitch * 4, dur, init_pitch * 4)
        STRUM2(start_time, dur, 20000, pitch, get_hardness(), dur / upper_decay, (rand() + 1) / 2)
    }

    return 0
}



////////////
// REVERB //
////////////

bus_config("STRUM2", "aux 0-1 out")
bus_config("FREEVERB", "aux 0-1 in", "out 0-1")

outskip = 0
inskip = 0
dur = max_dur
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
    float left_gate_maxdepth,
    float left_index,
    float right_gate_mindepth,
    float right_index
}

float make_note_node(float rest, float len, float midi_pitch, float left_gate_maxdepth, float left_index, float right_gate_mindepth, float right_index)
{
    struct StrumNoteNode note;
    note.rest = rest;
    note.len = len;
    note.midi_pitch = midi_pitch;
    note.left_gate_maxdepth = left_gate_maxdepth;
    note.left_index = left_index;
    note.right_gate_mindepth = right_gate_mindepth;
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

float choose_gliss(float start, float dur, float amp, float freq)
{
    if (gliss_mode == "none")
    {
        STRUM2(start, dur, amp, freq, get_hardness(), dur / upper_decay, (rand() + 1) / 2);
        if (note.midi_pitch < 40)
        {
            STRUM2(start, dur, amp + 2000, freq / 2, get_hardness(), dur / upper_decay, (rand() + 1) / 2);
        }
    }
    else if (gliss_mode == "uncoor")
    {
        make_note_slide_uncoor(freq, start, dur, amp, .2, 2)
    }
    else if (gliss_mode == "coor_stop")
    {
        make_note_slide_coor_stop(freq, start, dur, amp, coor_stop_time)
    }
    return 0
}

float play_note(struct StrumNotePlay note)
{
    freq = cpsmidi(note.midi_pitch) + (rand() * note.start_time / 15);
    offset = (rand() + 1) / 4;
    amp = rand() * 8000 + 12000;
    choose_gliss(note.start_time + offset, note.dur, amp, freq)
    return 0;
}



////////////////
//// CURSOR ////
////////////////

all_cursors = {};
new_cursors = {};

list limit_cursors(float depth)
{
    // Remove unneeded cursors
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


    // Determine the max number of cursors
    m_curs = max_curs;

    limit_depth_index = index(limit_depths, depth);

    if (limit_depth_index > -1)
    {
        m_curs = limit_max_cursor_nums[limit_depth_index];
    }


    if (len(all_cursors) > m_curs)
    {
        limited_cursors = {}
        for (i = 0; i < m_curs; i = i + 1)
        {
            curs = all_cursors[i];
            limited_cursors[len(limited_cursors)] = curs
        }

        all_cursors = limited_cursors
    }


    return all_cursors;
}

struct CursorStatus
{
    float index,
    float current_time,
    float depth
}

float new_cursor(float index, float current_time, float depth)
{
    struct CursorStatus ncurs;
    ncurs.index = index;
    ncurs.current_time = current_time;
    ncurs.depth = depth;
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

    // if the note will last longer than the max dur, end this cursor
    if (current_time + note_len > max_dur)
    {
        cursor.index = -1
        return 0
    }

    schedule_note(current_time, note_len, node.midi_pitch)

    //// Make a new cursor for the right branch
    if (cursor.depth >= node.right_gate_mindepth)
    {
        new_cursor(node.right_index, current_time, cursor.depth + 1)
    }


    //// The cursor stays on the left branch

    cursor.current_time = current_time;
    cursor.depth = cursor.depth + 1;


    if (cursor.depth < node.left_gate_maxdepth)
    {
        cursor.index = node.left_index;

    }

    else
    {
        cursor.index = -1;
    }





    return 0;
}

// BUILD HERE





//// set up inital cursors
for (i=0; i < init_curs; i += 1)
{
    struct CursorStatus cursor;
    cursor.index = i;
    cursor.current_time = 0;
    depth = 0;
    all_cursors[i] = cursor;
}
quit = 0


//// Traverse the tree

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

    if (depth > max_depth)
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