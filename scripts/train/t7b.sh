export CUDA_VISIBLE_DEVICES=0
model="llama-7b-hf"
num_epochs=150
batch_size=64
learning_rate=1e-5
warmup_ratio=0.1
training_type='direct' # Set training type here
wandb_run_name=$model-rebalance-mlp-ep$num_epochs-bsz$batch_size-lr$learning_rate-wr$warmup_ratio-$training_type-full_hidden
dataset="trivia_qa"

output_dir=data/$dataset/results/$model/$wandb_run_name
mkdir -p $output_dir
echo "Output directory: $output_dir"

# Choose the training script based on training_type
if [ "$training_type" = "indirect" ]; then
    training_script="train.py"
else
    training_script="train_direct.py"
fi

python $training_script \
    --train_frac 0.80 \
    --num_epochs $num_epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --warmup_ratio $warmup_ratio \
    --precision float32 \
    --output_dir  $output_dir \
    --hidden_states_filename data/$dataset/results/$model/mlp_act.pt \
    --text_generations_filename data/$dataset/results/$model/text_generations_alias.csv \
    --device cpu \
    --logging_steps 100 \
    --wandb_run_name $wandb_run_name \
    --model_layer_idx 28 \
