import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from pik.utils import normalize_answer
from vllm import LLM, SamplingParams
class Model:
	'''
	Loads a language model from HuggingFace.
	Implements methods to extract the hidden states and generate text from a given input.
	'''
	def __init__(self, model_checkpoint ,generation_options, precision=torch.float16, device='cuda'):
		if model_checkpoint == 'EleutherAI/gpt-j-6B':
			config = AutoConfig.from_pretrained(model_checkpoint)
			with init_empty_weights():
				self.model = AutoModelForCausalLM.from_config(config)
			self.model = load_checkpoint_and_dispatch(
				self.model,
				'sharded-gpt-j-6B',
				device_map='auto',
				dtype=precision,
				no_split_module_classes=['GPTJBlock'],
			)
		else:
			self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
		self.vllm_model = LLM(model=model_checkpoint, gpu_memory_utilization=0.4)
		self.sampling_params = SamplingParams(
			n=generation_options.get('n', 1),
			max_tokens=generation_options.get('max_new_tokens', 16),
			temperature=generation_options.get('temperature', 1),
			top_k=generation_options.get('top_k', 50),
			top_p=generation_options.get('top_p', 1),
			# pad_token_id=generation_options.get('pad_token_id', 50256),
			# eos_token_id=generation_options.get('eos_token_id', 198),
		)
		self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

	def get_hidden_states(self, text_input, keep_all=True):
		with torch.inference_mode():
			encoded_input = self.tokenizer(text_input, return_tensors='pt').to(self.model.device)
			output = self.model(encoded_input['input_ids'], output_hidden_states=True)
		if keep_all:
			# Stack all layers
			hidden_states = torch.stack(output.hidden_states, dim=0)
			# Keep only last token
			hidden_states = hidden_states[:, :, -1, :].squeeze()
		else:
			# Keep only last layer + last token
			hidden_states = output.hidden_states[-1][0, -1]
		return hidden_states

	def get_text_generation(self, text_input, normalize=False):
		# with torch.inference_mode():
		# use vllm to generate text
		generation = self.vllm_model.generate(text_input, self.sampling_params)

		# extract 
		output_text_list = [
			output.text
			for output in generation[0].outputs
		]
		if normalize:
			return [normalize_answer(output_text) for output_text in output_text_list]
		return output_text_list

	def parameters(self):
		return self.model.parameters()
