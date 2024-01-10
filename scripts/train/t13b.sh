# Description: Script to train the model
export CUDA_VISIBLE_DEVICES=6
model="llama-13b-hf"
num_epochs=100
batch_size=7680
learning_rate=1e-5
wandb_run_name=$model-ep$num_epochs-bsz$batch_size-lr$learning_rate

python train.py \
    --split_seed 101 \
    --train_seed 8421 \
    --train_frac 0.80 \
    --num_epochs $num_epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --precision float32 \
    --output_dir data/$model \
    --hidden_states_filename data/$model/hidden_states.pt \
    --text_generations_filename data/$model/text_generations.csv \
    --device cuda \
    --use_wandb \
    --wandb_run_name $wandb_run_name