
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")

folder_name="$timestamp"

python train.py \
    -n_workers 4 \
    -batch_size 4 \
    -input_dir dataset/e_piano1 \
    -output_dir outputs/$folder_name \
    --rpr \
    -print_modulus 100 \
    -weight_modulus 5 \

