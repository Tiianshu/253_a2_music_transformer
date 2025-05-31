#! /bin/bash

mkdir -p nesmdb_midi_processed

cp nesmdb_midi/train/*.mid nesmdb_midi_processed/ 
cp nesmdb_midi/valid/*.mid nesmdb_midi_processed/
cp nesmdb_midi/test/*.mid nesmdb_midi_processed/
