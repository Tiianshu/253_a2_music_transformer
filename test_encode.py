import pretty_midi
from third_party.midi_processor.processor import _control_preprocess, _note_preprocess, _divide_note, _make_time_sift_events, _snote2events


def encode_midi(file_path):
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)
    print(f">>> Loaded MIDI: {file_path}")
    print(f"    # of instruments: {len(mid.instruments)}")
    for idx, inst in enumerate(mid.instruments):
        print(f"    Instrument {idx}  (program={inst.program}, is_drum={inst.is_drum}):")
        print(f"        # notes = {len(inst.notes)}")
        print(f"        # control_changes = {len(inst.control_changes)}")
    # 后续原本的逻辑……
    events = []
    notes = []
    for inst in mid.instruments:
        ctls_raw = [ctrl for ctrl in inst.control_changes if ctrl.number == 64]
        print(f"    Track {inst.program} sustain-ctrl64 count = {len(ctls_raw)}")
        ctrls = _control_preprocess(ctls_raw)
        inst_notes = inst.notes
        notes += _note_preprocess(ctrls, inst_notes)
    print(f">>> after _note_preprocess, total notes count = {len(notes)}")
    dnotes = _divide_note(notes)
    print(f">>> after _divide_note, total dnotes = {len(dnotes)}")
    # ……继续原本循环，把 events 填进去
    dnotes.sort(key=lambda x: x.time)
    cur_time = 0
    cur_vel = 0
    for snote in dnotes:
        events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)
        events += _snote2events(snote=snote, prev_vel=cur_vel)
        cur_time = snote.time
        cur_vel = snote.velocity
    print(f">>> final events count = {len(events)}")
    return [e.to_int() for e in events]



# encode_midi("maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi")
encode_midi("nesmdb_midi_processed/000_10_YardFight_00_01GameStart.mid")

