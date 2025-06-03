# %% [markdown]
# ## Homework 3: Symbolic Music Generation Using Markov Chains

# %% [markdown]
# **Before starting the homework:**
# 
# Please run `pip install miditok` to install the [MiDiTok](https://github.com/Natooz/MidiTok) package, which simplifies MIDI file processing by making note and beat extraction more straightforward.
# 
# You’re also welcome to experiment with other MIDI processing libraries such as [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) and [miditoolkit](https://github.com/YatingMusic/miditoolkit). However, with these libraries, you’ll need to handle MIDI quantization yourself, for example, converting note-on/note-off events into beat positions and durations.

# %%
# run this command to install MiDiTok
# ! pip install miditok

# %%
# import required packages
import random
from glob import glob
from collections import defaultdict

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile

# %%
# You can change the random seed but try to keep your results deterministic!
# If I need to make changes to the autograder it'll require rerunning your code,
# so it should ideally generate the same results each time.
random.seed(42)

# %% [markdown]
# ### Load music dataset
# We will use a subset of the [PDMX dataset](https://zenodo.org/records/14984509). 
# 
# Please find the link in the homework spec.
# 
# All pieces are monophonic music (i.e. one melody line) in 4/4 time signature.

# %%

# Task 1
midi_files = glob('nesmdb_midi_processed/*.mid')
# Task 2
midi_files = glob('emopia_midis/*.mid')
len(midi_files)

# %% [markdown]
# ### Train a tokenizer with the REMI method in MidiTok

# %%
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)

# %% [markdown]
# ### Use the trained tokenizer to get tokens for each midi file
# In REMI representation, each note will be represented with four tokens: `Position, Pitch, Velocity, Duration`, e.g. `('Position_28', 'Pitch_74', 'Velocity_127', 'Duration_0.4.8')`; a `Bar_None` token indicates the beginning of a new bar.

# %%
# e.g.:
midi = Score(midi_files[0])
tokens = tokenizer(midi)[0].tokens
tokens[:10]

# %% [markdown]
# 1. Write a function to extract note pitch events from a midi file; and another extract all note pitch events from the dataset and output a dictionary that maps note pitch events to the number of times they occur in the files. (e.g. {60: 120, 61: 58, …}).
# 
# `note_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of note pitch events (e.g. [60, 62, 61, ...])
# 
# `note_frequency()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to the number of times they occur, e.g {60: 120, 61: 58, …}

# %%
def note_extraction(midi_file):
    # Q1a: Your code goes here
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    pitches = []
    for token in tokens:
        if token.startswith('Pitch_'):
            pitch = int(token.split('_')[1])
            pitches.append(pitch)
    return pitches

# print(note_extraction(midi_files[0]))

# %%
def note_frequency(midi_files):
    # Q1b: Your code goes here
    note_freq = defaultdict(int)
    for midi_file in midi_files:
        pitches = note_extraction(midi_file)
        for pitch in pitches:
            note_freq[pitch] += 1
    return dict(note_freq)

# print(note_frequency(midi_files))

# %% [markdown]
# 2. Write a function to normalize the above dictionary to produce probability scores (e.g. {60: 0.13, 61: 0.065, …})
# 
# `note_unigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to probabilities, e.g. {60: 0.13, 61: 0.06, …}

# %%
def note_unigram_probability(midi_files):
    note_counts = note_frequency(midi_files)
    unigramProbabilities = {}
    
    total_notes = sum(note_counts.values())
    # Q2: Your code goes here
    for note, count in note_counts.items():
        unigramProbabilities[note] = count / total_notes
    
    return unigramProbabilities

# print(note_unigram_probability(midi_files))

# %% [markdown]
# 3. Generate a table of pairwise probabilities containing p(next_note | previous_note) values for the dataset; write a function that randomly generates the next note based on the previous note based on this distribution.
# 
# `note_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramTransitions`: key: previous_note, value: a list of next_note, e.g. {60:[62, 64, ..], 62:[60, 64, ..], ...} (i.e., this is a list of every other note that occured after note 60, every note that occured after note 62, etc.)
# 
#   - `bigramTransitionProbabilities`: key:previous_note, value: a list of probabilities for next_note in the same order of `bigramTransitions`, e.g. {60:[0.3, 0.4, ..], 62:[0.2, 0.1, ..], ...} (i.e., you are converting the values above to probabilities)
# 
# `sample_next_note()`
# - **Input**: a note
# 
# - **Output**: next note sampled from pairwise probabilities

# %%
def note_bigram_probability(midi_files):
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)

    # Q3a: Your code goes here
    bi_gram_pairs = defaultdict(dict)
    for midi_file in midi_files:
        pitches = note_extraction(midi_file)
        for i in range(len(pitches) - 1):
            prev_note = pitches[i]
            next_note = pitches[i + 1]
            if next_note not in bi_gram_pairs[prev_note]:
                bi_gram_pairs[prev_note][next_note] = 0
            bi_gram_pairs[prev_note][next_note] += 1

    for prev_note, next_notes in bi_gram_pairs.items():
        total_count = sum(next_notes.values())
        for next_note, count in next_notes.items():
            bigramTransitions[prev_note].append(next_note)
            bigramTransitionProbabilities[prev_note].append(count / total_count)

    return bigramTransitions, bigramTransitionProbabilities

# print(note_bigram_probability(midi_files))

# %%
def sample_next_note(note):
    # Q3b: Your code goes here
    trans, probs = note_bigram_probability(midi_files)
    next_notes_choice = trans.get(note, [])
    next_probs_choice = probs.get(note, [])

    next_note = choice(next_notes_choice, p=next_probs_choice)
    return next_note

# print(sample_next_note(74))  # Example note, replace with your own

# %% [markdown]
# 4. Write a function to calculate the perplexity of your model on a midi file.
# 
#     The perplexity of a model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-1})))$
# 
#     where $p(w_1|w_0) = p(w_1)$, $p(w_i|w_{i-1}) (i>1)$ refers to the pairwise probability p(next_note | previous_note).
# 
# `note_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_bigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    
    # Q4: Your code goes here
    # Can use regular numpy.log (i.e., natural logarithm)
    pitches = note_extraction(midi_file)
    
    log_probs = []
    # first one
    first_note = pitches[0]
    log_probs.append(np.log(unigramProbabilities.get(first_note, 1e-10)))

    # rest of them
    for i in range(1, len(pitches)):
        prev_note = pitches[i - 1]
        curr_note = pitches[i]
        if prev_note in bigramTransitions and curr_note in bigramTransitions[prev_note]:
            prob = bigramTransitionProbabilities[prev_note][bigramTransitions[prev_note].index(curr_note)]
            log_probs.append(np.log(prob))
        else:
            log_probs.append(np.log(1e-10))  # small value for unseen transitions

    # Calculate perplexity
    log_prob_sum = sum(log_probs)
    perplexity = np.exp(-log_prob_sum / len(log_probs))
    return perplexity

# print(note_bigram_perplexity(midi_files[0]))

# %% [markdown]
# 5. Implement a second-order Markov chain, i.e., one which estimates p(next_note | next_previous_note, previous_note); write a function to compute the perplexity of this new model on a midi file. 
# 
#     The perplexity of this model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-2}, w_{i-1})))$
# 
#     where $p(w_1|w_{-1}, w_0) = p(w_1)$, $p(w_2|w_0, w_1) = p(w_2|w_1)$, $p(w_i|w_{i-2}, w_{i-1}) (i>2)$ refers to the probability p(next_note | next_previous_note, previous_note).
# 
# 
# `note_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramTransitions`: key - (next_previous_note, previous_note), value - a list of next_note, e.g. {(60, 62):[64, 66, ..], (60, 64):[60, 64, ..], ...}
# 
#   - `trigramTransitionProbabilities`: key: (next_previous_note, previous_note), value: a list of probabilities for next_note in the same order of `trigramTransitions`, e.g. {(60, 62):[0.2, 0.2, ..], (60, 64):[0.4, 0.1, ..], ...}
# 
# `note_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_trigram_probability(midi_files):
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)
    
    # Q5a: Your code goes here
    tri_gram_pairs = defaultdict(dict)
    for midi_file in midi_files:
        pitches = note_extraction(midi_file)
        for i in range(len(pitches) - 2):
            prev_note = pitches[i]
            next_note = pitches[i + 1]
            next_next_note = pitches[i + 2]
            if next_next_note not in tri_gram_pairs[(prev_note, next_note)]:
                tri_gram_pairs[(prev_note, next_note)][next_next_note] = 0
            tri_gram_pairs[(prev_note, next_note)][next_next_note] += 1

    for (prev_note, next_note), next_notes in tri_gram_pairs.items():
        total_count = sum(next_notes.values())
        for next_next_note, count in next_notes.items():
            trigramTransitions[(prev_note, next_note)].append(next_next_note)
            trigramTransitionProbabilities[(prev_note, next_note)].append(count / total_count)

    return trigramTransitions, trigramTransitionProbabilities

# %%
def note_trigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    
    # Q5b: Your code goes here
    pitches = note_extraction(midi_file)
    log_probs = []

    # first one
    first_note = pitches[1]
    log_probs.append(np.log(unigramProbabilities.get(first_note, 1e-10)))

    # second one
    second_note = pitches[2]
    log_probs.append(np.log(bigramTransitionProbabilities[pitches[1]][bigramTransitions[pitches[1]].index(second_note)]))

    # rest of them
    for i in range(2, len(pitches)):
        prev_note = pitches[i - 2]
        curr_note = pitches[i - 1]
        next_note = pitches[i]
        if (prev_note, curr_note) in trigramTransitions and next_note in trigramTransitions[(prev_note, curr_note)]:
            prob = trigramTransitionProbabilities[(prev_note, curr_note)][trigramTransitions[(prev_note, curr_note)].index(next_note)]
            log_probs.append(np.log(prob))
        else:
            log_probs.append(np.log(1e-10))  # small value for unseen transitions

    # Calculate perplexity
    log_prob_sum = sum(log_probs)
    perplexity = np.exp(-log_prob_sum / len(log_probs))
    return perplexity

# %% [markdown]
# 6. Our model currently doesn’t have any knowledge of beats. Write a function that extracts beat lengths and outputs a list of [(beat position; beat length)] values.
# 
#     Recall that each note will be encoded as `Position, Pitch, Velocity, Duration` using REMI. Please keep the `Position` value for beat position, and convert `Duration` to beat length using provided lookup table `duration2length` (see below).
# 
#     For example, for a note represented by four tokens `('Position_24', 'Pitch_72', 'Velocity_127', 'Duration_0.4.8')`, the extracted (beat position; beat length) value is `(24, 4)`.
# 
#     As a result, we will obtain a list like [(0,8),(8,16),(24,4),(28,4),(0,4)...], where the next beat position is the previous beat position + the beat length. As we divide each bar into 32 positions by default, when reaching the end of a bar (i.e. 28 + 4 = 32 in the case of (28, 4)), the beat position reset to 0.

# %%
duration2length = {
    '0.2.8': 2,  # sixteenth note, 0.25 beat in 4/4 time signature
    '0.4.8': 4,  # eighth note, 0.5 beat in 4/4 time signature
    '1.0.8': 8,  # quarter note, 1 beat in 4/4 time signature
    '2.0.8': 16, # half note, 2 beats in 4/4 time signature
    '4.0.4': 32, # whole note, 4 beats in 4/4 time signature
}

# %% [markdown]
# `beat_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of (beat position; beat length) values

# %%
def beat_extraction(midi_file):
    # Q6: Your code goes here
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens

    beats = [] # (list of tuples (beat, duration))
    i = 0
    while i < len(tokens):
        # a note has four tokens
        if tokens[i].startswith('Position_'):
            position = int(tokens[i].split('_')[1])
            duration = duration2length[(tokens[i + 3].split('_')[1])]
            # print(position, duration)
            beats.append((position, duration))
            i += 4
        else:
            i += 1

    return beats

# print(beat_extraction(midi_files[0]))

# %% [markdown]
# 7. Implement a Markov chain that computes p(beat_length | previous_beat_length) based on the above function.
# 
# `beat_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatTransitions`: key: previous_beat_length, value: a list of beat_length, e.g. {4:[8, 2, ..], 8:[8, 4, ..], ...}
# 
#   - `bigramBeatTransitionProbabilities`: key - previous_beat_length, value - a list of probabilities for beat_length in the same order of `bigramBeatTransitions`, e.g. {4:[0.3, 0.2, ..], 8:[0.4, 0.4, ..], ...}

# %%
def beat_bigram_probability(midi_files):
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)
    
    bi_gram_pairs = defaultdict(dict)
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for i in range(len(beats) - 1):
            prev_beat = beats[i][1]
            next_beat = beats[i + 1][1]
            if next_beat not in bi_gram_pairs[prev_beat]:
                bi_gram_pairs[prev_beat][next_beat] = 0
            bi_gram_pairs[prev_beat][next_beat] += 1

    for prev_beat, next_beats in bi_gram_pairs.items():
        total_count = sum(next_beats.values())
        for next_beat, count in next_beats.items():
            bigramBeatTransitions[prev_beat].append(next_beat)
            bigramBeatTransitionProbabilities[prev_beat].append(count / total_count)
    
    return bigramBeatTransitions, bigramBeatTransitionProbabilities

# print(beat_bigram_probability(midi_files[0:5]))

# %% [markdown]
# 8. Implement a function to compute p(beat length | beat position), and compute the perplexity of your models from Q7 and Q8. For both models, we only consider the probabilities of predicting the sequence of **beat lengths**.
# 
# `beat_pos_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatPosTransitions`: key - beat_position, value - a list of beat_length
# 
#   - `bigramBeatPosTransitionProbabilities`: key - beat_position, value - a list of probabilities for beat_length in the same order of `bigramBeatPosTransitions`
# 
# `beat_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: two perplexity values correspond to the models in Q7 and Q8, respectively

# %%
def beat_pos_bigram_probability(midi_files):
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)
    
    bi_gram_pairs = defaultdict(dict) # key: pos, value: length
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for i in range(len(beats)):
            pos = beats[i][0]
            length = beats[i][1]
            if length not in bi_gram_pairs[pos]:
                bi_gram_pairs[pos][length] = 0
            bi_gram_pairs[pos][length] += 1

    for pos, lengths in bi_gram_pairs.items():
        total_count = sum(lengths.values())
        for length, count in lengths.items():
            bigramBeatPosTransitions[pos].append(length)
            bigramBeatPosTransitionProbabilities[pos].append(count / total_count)
    
    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities

# print(beat_pos_bigram_probability(midi_files[0:5]))

# %%
def beat_bigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    # Q8b: Your code goes here
    # Hint: one more probability function needs to be computed

    # calculate unigram probabilities for beat lengths
    beat_lengths = defaultdict(int)
    for midi_file_ in midi_files:
        beats = beat_extraction(midi_file_)
        for _, length in beats:
            beat_lengths[length] += 1

    total_beats = sum(beat_lengths.values())
    beat_unigramProbabilities = {length: count / total_beats for length, count in beat_lengths.items()}

    beats = beat_extraction(midi_file)

    beat_lengths = [length for _, length in beats]
    beat_positions = [pos for pos, _ in beats]
    
    # --- Q7 perplexity: p(next_beat_length | prev_beat_length)
    log_probs_q7 = []

    # first one
    first_beat = beat_lengths[0]
    log_probs_q7.append(np.log(beat_unigramProbabilities.get(first_beat, 1e-10)))

    for i in range(1, len(beat_lengths)):
        prev = beat_lengths[i - 1]
        curr = beat_lengths[i]
        if prev in bigramBeatTransitions and curr in bigramBeatTransitions[prev]:
            idx = bigramBeatTransitions[prev].index(curr)
            prob = bigramBeatTransitionProbabilities[prev][idx]
            log_probs_q7.append(np.log(prob))
        else:
            log_probs_q7.append(np.log(1e-10))  # fallback small prob

    perplexity_Q7 = np.exp(-sum(log_probs_q7) / len(log_probs_q7))

    # --- Q8 perplexity: p(beat_length | beat_position)
    log_probs_q8 = []
    for i in range(len(beat_positions)):
        pos = beat_positions[i]
        length = beat_lengths[i]
        if pos in bigramBeatPosTransitions and length in bigramBeatPosTransitions[pos]:
            idx = bigramBeatPosTransitions[pos].index(length)
            prob = bigramBeatPosTransitionProbabilities[pos][idx]
            log_probs_q8.append(np.log(prob))
        else:
            log_probs_q8.append(np.log(1e-10))  # fallback small prob

    perplexity_Q8 = np.exp(-sum(log_probs_q8) / len(log_probs_q8))

    return perplexity_Q7, perplexity_Q8

print(beat_bigram_perplexity(midi_files[0]))

# %% [markdown]
# 9. Implement a Markov chain that computes p(beat_length | previous_beat_length, beat_position), and report its perplexity. 
# 
# `beat_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramBeatTransitions`: key: (previous_beat_length, beat_position), value: a list of beat_length
# 
#   - `trigramBeatTransitionProbabilities`: key: (previous_beat_length, beat_position), value: a list of probabilities for beat_length in the same order of `trigramBeatTransitions`
# 
# `beat_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def beat_trigram_probability(midi_files):
    trigramBeatTransitions = defaultdict(list)
    trigramBeatTransitionProbabilities = defaultdict(list)

    # Q9a: Your code goes here
    tri_gram_pairs = defaultdict(dict)  # key: (prev_length, pos), value: next_length counts

    for midi_file in midi_files:
        beats = beat_extraction(midi_file)  # [(pos, length)]
        for i in range(1, len(beats)):
            prev_length = beats[i - 1][1]
            curr_pos = beats[i][0]
            curr_length = beats[i][1]

            key = (prev_length, curr_pos)
            if curr_length not in tri_gram_pairs[key]:
                tri_gram_pairs[key][curr_length] = 0
            tri_gram_pairs[key][curr_length] += 1

    for (prev_length, pos), next_lengths in tri_gram_pairs.items():
        total_count = sum(next_lengths.values())
        for next_length, count in next_lengths.items():
            trigramBeatTransitions[(prev_length, pos)].append(next_length)
            trigramBeatTransitionProbabilities[(prev_length, pos)].append(count / total_count)
   
    return trigramBeatTransitions, trigramBeatTransitionProbabilities

# print(beat_trigram_probability(midi_files[0:5]))

# %%
def beat_trigram_perplexity(midi_file):
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)
    # Q9b: Your code goes here

    beats = beat_extraction(midi_file)  # [(pos, length)]
    beat_lengths = [length for _, length in beats]
    beat_positions = [pos for pos, _ in beats]

    log_probs = []
    # first one, get prob(len|pos)
    first_pos = beat_positions[0]
    first_length = beat_lengths[0]
    log_probs.append(np.log(bigramBeatPosTransitionProbabilities[first_pos][bigramBeatPosTransitions[first_pos].index(first_length)]))

    for i in range(1, len(beats)):
        prev_length = beat_lengths[i - 1]
        curr_pos = beat_positions[i]
        curr_length = beat_lengths[i]

        key = (prev_length, curr_pos)
        if key in trigramBeatTransitions and curr_length in trigramBeatTransitions[key]:
            idx = trigramBeatTransitions[key].index(curr_length)
            prob = trigramBeatTransitionProbabilities[key][idx]
            log_probs.append(np.log(prob))
        else:
            log_probs.append(np.log(1e-10))  # fallback small prob

    if log_probs:
        perplexity = np.exp(-sum(log_probs) / len(log_probs))
    else:
        perplexity = float('inf')

    return perplexity    

# print(beat_trigram_perplexity(midi_files[0]))

# %% [markdown]
# 10. Use the model from Q5 to generate N notes, and the model from Q8 to generate beat lengths for each note. Save the generated music as a midi file (see code from workbook1) as q10.mid. Remember to reset the beat position to 0 when reaching the end of a bar.
# 
# `music_generate`
# - **Input**: target length, e.g. 500
# 
# - **Output**: a midi file q10.mid
# 
# Note: the duration of one beat in MIDIUtil is 1, while in MidiTok is 8. Divide beat length by 8 if you use methods in MIDIUtil to save midi files.

# %%
def music_generate(length):
    # sample notes
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    beatPosTransitions, beatPosProbabilities = beat_pos_bigram_probability(midi_files)
    
    # Q10: Your code goes here ...
    
    # Initialize the first note and beat
    current_pos = 0
    
    sampled_notes = []
    sampled_beats = []

    for i in range(length):
        # Sample the next note using the bigram model
        if i == 0:
            next_note = choice(list(unigramProbabilities.keys()), p=list(unigramProbabilities.values()))
        elif i == 1:
            current_note = sampled_notes[-1]
            next_note = choice(bigramTransitions[current_note], p=bigramTransitionProbabilities[current_note])
        else:
            current_note = sampled_notes[-1]
            prev_note = sampled_notes[-2]
            next_note = choice(trigramTransitions[(prev_note, current_note)], p=trigramTransitionProbabilities[(prev_note, current_note)])
        
        sampled_notes.append(next_note)

        # Sample the next beat using the bigram model
        next_len = choice(beatPosTransitions[current_pos], p=beatPosProbabilities[current_pos])
        
        sampled_beats.append((current_pos, next_len))

        current_pos = (current_pos + next_len) % 32
    
    # save the generated music as a midi file
    # print(sampled_notes)
    # print(sampled_beats)

    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)

    time = 0  # reset time counter
    for note, (_, beat_len) in zip(sampled_notes, sampled_beats):
        midi.addNote(0, 0, note, time, beat_len / 8, 100)
        time += beat_len / 8  # advance global time

    with open('q10.mid', 'wb') as f:
        midi.writeFile(f)


music_generate(500)

# %%



