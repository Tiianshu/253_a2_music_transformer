
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")

folder_name="$timestamp"

python train.py \
    -n_workers 4 \
    -batch_size 4 \
    -epochs 80 \
    -input_dir dataset/e_piano1 \
    -output_dir outputs/$folder_name \
    --rpr \
    -print_modulus 100 \
    -weight_modulus 3 \
    -continue_weights outputs/2025-05-31_16-46-42/weights/epoch_0030.pickle \
    -continue_epoch 30 \

