



python generate.py \
    -target_seq_length 1280 \
    -num_prime 384 \
    -beam 0 \
    -midi_root dataset/e_piano1 \
    -output_dir gen_output \
    -model_weights outputs/2025-06-01_17-55-16/weights/epoch_0036.pickle \
    --rpr \
    -primer_file gen_output/saved/primer_3.mid


# python generate2.py \
#     -target_seq_length 1280 \
#     -num_prime 384 \
#     -beam 0 \
#     -midi_root dataset/e_piano1 \
#     -output_dir gen_output \
#     -model_weights outputs/2025-06-01_17-55-16/weights/epoch_0036.pickle \
#     --rpr \
#     # -primer_file nesmdb_midi_processed/322_SuperMarioBros__00_01RunningAbout.mid \