export CUDA_VISIBLE_DEVICES=3

model=llama-7b-hf
dataset=gsm8k
template=cot
example_file=data/gsm8k/cot_example.json
datafolder=data/$dataset/$model
mkdir -p $datafolder
python generation_lyt.py \
    --model_checkpoint MODELS/$model \
    --n_answers_per_question 30 \
    --temperature 1 \
    --pad_token_id 50256 \
    --keep_all_hidden_layers \
    --hidden_states_filename $datafolder/mlp_act_$template.pt \
    --text_generations_filename $datafolder/text_generations_$template.json \
    --template $template \
    --dataset $dataset \
    --example_file $example_file \
    --max_new_tokens 96 \
    --mlp \
    --debug