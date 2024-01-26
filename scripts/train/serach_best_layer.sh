export CUDA_VISIBLE_DEVICES=0
model="llama-7b-hf"
num_epochs=150
batch_size=64
learning_rate=1e-5
warmup_ratio=0.1
training_type='direct' # Set training type here

# Choose the training script based on training_type
if [ "$training_type" = "indirect" ]; then
    training_script="train.py"
else
    training_script="train_direct.py"
fi

# Loop over all layers
# for model_layer_idx in {0..32}
# do
#     wandb_run_name="${model}-layer${model_layer_idx}-mlp-ep${num_epochs}-bsz${batch_size}-lr${learning_rate}-wr${warmup_ratio}-${training_type}-full_hidden"

#     output_dir="data/$model/search_layer/$wandb_run_name"
#     mkdir -p "$output_dir"
#     echo "Output directory: $output_dir"

#     python $training_script \
#         --train_frac 0.80 \
#         --num_epochs $num_epochs \
#         --batch_size $batch_size \
#         --learning_rate $learning_rate \
#         --warmup_ratio $warmup_ratio \
#         --precision float32 \
#         --output_dir  $output_dir \
#         --hidden_states_filename "data/$model/mlp_act.pt" \
#         --text_generations_filename "data/$model/text_generations_alias.csv" \
#         --device cuda \
#         --logging_steps 100 \
#         --wandb_run_name $wandb_run_name \
#         --model_layer_idx $model_layer_idx
# done


wandb_run_name="${model}-full-layers-mlp-ep${num_epochs}-bsz${batch_size}-lr${learning_rate}-wr${warmup_ratio}-${training_type}-full_hidden"

output_dir="data/$model/search_layer/$wandb_run_name"
mkdir -p "$output_dir"
echo "Output directory: $output_dir"

python $training_script \
    --train_frac 0.80 \
    --num_epochs $num_epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --warmup_ratio $warmup_ratio \
    --precision float32 \
    --output_dir  $output_dir \
    --hidden_states_filename "data/trivia_qa/$model/mlp_act.pt" \
    --text_generations_filename "data/trivia_qa/$model/text_generations_alias.csv" \
    --device cuda \
    --logging_steps 100 \
    --wandb_run_name $wandb_run_name