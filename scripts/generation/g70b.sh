export CUDA_VISIBLE_DEVICES=4,5,6,7

model=llama-2-70b
datafolder=data/$model
mkdir -p $datafolder
python generation_lyt.py \
    --model_checkpoint MODELS/$model \
    --n_answers_per_question 30 \
    --max_new_tokens 16 \
    --temperature 1 \
    --pad_token_id 50256 \
    --keep_all_hidden_layers \
    --hidden_states_filename $datafolder/hidden_states.pt \
    --text_generations_filename $datafolder/text_generations.csv
    