import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

# main
def main():
    args = parse_generate_args()
    print_generate_args(args)

    if args.force_cpu:
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    # Initialize model
    model = MusicTransformer(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        max_sequence=args.max_sequence,
        rpr=args.rpr
    ).to(get_device())
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()

    # Randomly select 20 unique primers
    chosen_indices = random.sample(range(len(dataset)), 20)

    with torch.no_grad():
        for i, idx in enumerate(chosen_indices):
            primer, _ = dataset[idx]
            primer = primer.to(get_device())

            print(f"[{i+1}/20] Using primer index {idx}: {dataset.data_files[idx]}")

            # Save primer MIDI
            primer_path = os.path.join(args.output_dir, f"primer_{i}.mid")
            decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=primer_path)

            # Generate music
            if args.beam > 0:
                print("    → Generating with BEAM search...")
                seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)
                out_path = os.path.join(args.output_dir, f"beam_{i}.mid")
            else:
                print("    → Generating with random sampling...")
                seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)
                out_path = os.path.join(args.output_dir, f"rand_{i}.mid")

            decode_midi(seq[0].cpu().numpy(), file_path=out_path)

if __name__ == "__main__":
    main()
