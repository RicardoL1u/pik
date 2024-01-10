export CUDA_VISIBLE_DEVICES=3

model=llama-13b-hf
mkdir -p data/$model
python generate.py \
    --n_questions 10 \
    --dataset_seed 420 \
    --generation_seed 1337 \
    --model_checkpoint MODELS/$model \
    --precision float16 \
    --n_answers_per_question 30 \
    --max_new_tokens 16 \
    --temperature 1 \
    --pad_token_id 50256 \
    --keep_all_hidden_layers \
    --save_frequency 99999 \
    --data_folder data/$model \
    --hidden_states_filename hidden_states.pt \
    --text_generations_filename text_generations.csv \
    --qa_pairs_filename qa_pairs.csv