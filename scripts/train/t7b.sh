export CUDA_VISIBLE_DEVICES=0
model="Mistral-7B-Instruct-v0.2"
num_epochs=16
batch_size=256
learning_rate=1e-5
warmup_ratio=0.1
train_type='direct' # Set training type here
rebalance=false  # Use lowercase 'true' or 'false'
template=icl
train_var=hidden_states
model_layer_idx=28,29,30,31
probe_model=mlp
dataset="trivia_qa_wiki"
debug=false


echo "Output directory: $output_dir"

# Add --rebalance flag conditionally based on rebalance variable
rebalance_flag=""
if [ "$rebalance" = "true" ]; then
    rebalance_flag="--rebalance"
fi
echo "Rebalance flag: $rebalance_flag"

wandb_run_name=adam-probe-100-$probe_model-$train_var-$train_type-re$rebalance-layer-$model_layer_idx-ep$num_epochs-bsz$batch_size-lr$learning_rate-wr$warmup_ratio
output_dir=data/$dataset/results/$model/$wandb_run_name
mkdir -p $output_dir


train_type_flag=""
if [ "$train_type" = "direct" ]; then
    train_type_flag="--direct"
fi
echo "Train type flag: $train_type_flag"

probe_model_flag=""
if [ "$probe_model" = "mlp" ]; then
    probe_model_flag="--mlp"
fi

debug_flag=""
if [ "$debug" = "true" ]; then
    debug_flag="--debug"
fi

python train.py \
    --train_frac 0.80 \
    --num_epochs $num_epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --warmup_ratio $warmup_ratio \
    --precision float32 \
    --output_dir  $output_dir \
    --hidden_states_filename data/$dataset/results/$model/$train_var\_$template.pt \
    --text_generations_filename data/$dataset/results/$model/text_generations_$template.json \
    --device cuda \
    --logging_steps 10 \
    --wandb_run_name $wandb_run_name \
    --model_layer_idx $model_layer_idx \
    $rebalance_flag \
    $train_type_flag \
    $probe_model_flag \
    $debug_flag \

