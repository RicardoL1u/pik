export CUDA_VISIBLE_DEVICES=3

python evaluate.py \
    --dataset gsm8k \
    --model_ckpt_path data/trivia_qa/results/llama-7b-hf/llama-7b-hf-rebalance-mlp-ep150-bsz64-lr1e-5-wr0.1-direct-full_hidden/best_model.pt \
    --hidden_states_filename data/gsm8k/llama-7b-hf/mlp_act.pt \
    --text_generations_filename data/gsm8k/llama-7b-hf/text_generations_alias.json \
    --model_layer_idx 28 \
    --device cpu