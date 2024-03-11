export CUDA_VISIBLE_DEVICES=0,1

model=Mistral-7B-v0.1
dataset=commonsense_qa
template=mcq_cmqa
example_file=data/$dataset/example.json
datafolder=data/$dataset/$model
mkdir -p $datafolder
python generation.py \
    --model_checkpoint MODELS/$model \
    --n_answers_per_question 30 \
    --temperature 1 \
    --pad_token_id 50256 \
    --keep_all_hidden_layers \
    --hidden_states_filename $datafolder/hidden_states_$template.pt \
    --text_generations_filename $datafolder/text_generations_$template.json \
    --template $template \
    --dataset $dataset \
    --example_file $example_file \
    --max_new_tokens 96 \
    --shot 0