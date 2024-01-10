# Description: Script to train the model
export CUDA_VISIBLE_DEVICES=7
python train_lyt.py \
    --split_seed 101 \
    --train_seed 8421 \
    --train_frac 0.80 \
    --num_epochs 100 \
    --batch_size 7680 \
    --learning_rate 1e-5 \
    --precision float32 \
    --output_dir data/llama-65b-hf \
    --hidden_states_filename data/llama-65b-hf/hidden_states.pt \
    --text_generations_filename data/llama-65b-hf/text_generations.csv \
    --device cuda