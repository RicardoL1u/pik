import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM
from pik.utils.utils import normalize_answer
from vllm import LLM, SamplingParams
from .hook_tool import HookMLPActivation
import re
import tracemalloc
from typing import List
from tqdm import tqdm
import logging
import gc
def is_large_model(model_checkpoint):
    model_checkpoint = model_checkpoint.lower()
    try:
        # Using regex to extract the number which may have a decimal part
        model_size = re.search(r'\d+(\.\d+)?b', model_checkpoint).group(0)
        # Convert the extracted string to float
        model_size = float(model_size[:-1])
        logging.info(f'{model_checkpoint} is a {model_size}B model')
        return model_size > 2
    except:
        logging.warning(f'Cannot determine the size of {model_checkpoint}. Assuming it is a small model')
        return False

def load_model(model_checkpoint, is_vllm=False):
    if is_large_model(model_checkpoint):
        if is_vllm:
            logging.debug(f'Loading Model with tensor parallelism: {torch.cuda.device_count()} GPUs')
            return  LLM(model=model_checkpoint, tensor_parallel_size=torch.cuda.device_count())
        else:
            return AutoModelForCausalLM.from_pretrained(model_checkpoint, 
                                                        device_map='auto',
                                                        attn_implementation="flash_attention_2",
                                                        torch_dtype=torch.float16)
    else:
        if is_vllm:
            return LLM(model=model_checkpoint)
        else:
            return AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                        attn_implementation="flash_attention_2",
                                                        torch_dtype=torch.float16).to('cuda')

class Model:
    '''
    Loads a language model from HuggingFace.
    Implements methods to extract the hidden states and generate text from a given input.
    '''
    def __init__(self, model_checkpoint ,generation_options, is_chat_model=False, is_low_memory=True):
        self.sampling_params = SamplingParams(
            n=generation_options.get('n', 1),
            max_tokens=generation_options.get('max_new_tokens', 16),
            temperature=generation_options.get('temperature', 1),
            top_k=generation_options.get('top_k', 50),
            top_p=generation_options.get('top_p', 1),
            stop=['Question','Q:']
            # pad_token_id=generation_options.get('pad_token_id', 50256),
            # eos_token_id=generation_options.get('eos_token_id', 198),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,
                                                       padding_side='left')
        # set padding token id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.is_chat_model = is_chat_model
        self.is_low_memory = is_low_memory
        # Load model when needed
        self.model_checkpoint = model_checkpoint
        self.model = None
        self.vllm_model = None

    
    def preprocess_text(self, text: List[str] | str) -> List[str] | str:
        # add chat template for chat model
        if self.is_chat_model:
            TEMPLATE = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": None},
            ]
            if isinstance(text, list):
                for i in range(len(text)):
                    temp_text = text[i]
                    template_copy = TEMPLATE.copy()
                    template_copy[1]["content"] = temp_text
                    text[i] = self.tokenizer.apply_chat_template(template_copy, tokenize=False, add_generation_prompt=True)
            else:
                template_copy = TEMPLATE.copy()
                template_copy[1]["content"] = text
                text = self.tokenizer.apply_chat_template(template_copy, tokenize=False, add_generation_prompt=True)
            
            test_text = text if isinstance(text, str) else text[0]
            # print(test_text)
            # assert all([special_token in test_text for special_token in self.tokenizer.special_tokens_map['additional_special_tokens']]), 'Chat template not applied correctly'
            
        return text
            
            
    
    def get_batch_MLP_activations(self, texts, batch_size=32, keep_all=True):
        
        texts = self.preprocess_text(texts)
        
        if self.model is None:
            self.model: AutoModelForCausalLM = load_model(self.model_checkpoint, is_vllm=False)
            # add hook to the model
            self.hook = HookMLPActivation(self.model)
            
        mlp_activations_list = []
        tracemalloc.start()  # Start tracing memory allocation
        for i in tqdm(range(0, len(texts), batch_size), 
                      desc='Generating MLP activations',
                      total=len(texts)//batch_size,
                      ncols=100):
            batch_texts = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch_texts, 
                                           padding=True, 
                                           return_tensors='pt',
                                           ).to(self.model.device)
            with torch.no_grad():
                output = self.model(**encoded_input)
            
            # concat all the activations into a mat (bsz, n_layers, seq_len, hidden_size)
            # each value in self.hook.activations is (bsz, seq_len, hidden_size)
            act_mat = torch.cat([act.unsqueeze(dim=1).cpu() for act in self.hook.activations.values()], dim=1)
            logging.debug(f'act_mat shape: {act_mat.shape}')
            # only keep the last token
            act_mat = act_mat[:, :, -1, :].squeeze() # (bsz, n_layers, hidden_size)
            logging.debug(f'act_mat shape after keep last token: {act_mat.shape}')
            # if keep all the layers
            if keep_all:
                mlp_activations_list.append(act_mat)
            else:
                # only keep the last layer
                mlp_activations_list.append(act_mat[:, -1, :].squeeze())
            # release memory
            del act_mat, output, encoded_input, self.hook.activations
            self.hook.activations = {}
            torch.cuda.empty_cache()
            
            gc.collect()
            if len(mlp_activations_list) % 10 == 0:
                # print the memory usage of mlm_activations_list in cpu in gigabytes
                def calculate_memory_cost(tensor_list):
                    total_memory_bytes = sum([t.element_size() * t.nelement() for t in tensor_list])
                    total_memory_gb = total_memory_bytes / (1024**3)  # Convert bytes to gigabytes
                    return total_memory_gb
                logging.info(f'mlp_activations_list memory cost: {calculate_memory_cost(mlp_activations_list)} GB')
                
                
        # release memory
        self.model = None
        return torch.cat(mlp_activations_list, dim=0)
            
            
    def get_batch_hidden_states(self, texts, batch_size=4, keep_all=True):
        
        texts = self.preprocess_text(texts)
        
        # Method to process texts in batches
        if self.model is None:
            self.model: AutoModelForCausalLM = load_model(self.model_checkpoint, is_vllm=False)

        hidden_states_list = []
        # for i in range(0, len(texts), batch_size):
        for i in tqdm(range(0, len(texts), batch_size), 
                      desc='Generating batch hidden states',
                      total=len(texts)//batch_size,
                      ncols=100):
            batch_texts = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch_texts, 
                                           padding=True, 
                                           return_tensors='pt',
                                           ).to(self.model.device)

            with torch.no_grad():  # Reduces memory usage
                output = self.model(**encoded_input, output_hidden_states=True)

            # move output.hidden_states to cpu
            output.hidden_states = [layer.cpu() for layer in output.hidden_states]
            
            logging.debug(f'layers: {len(output.hidden_states)}')
            logging.debug(f'shape of output.hidden_states: {output.hidden_states[-1].shape}')
            
            if not keep_all:
                # Keep only last layer of last token
                batch_states = output.hidden_states[-1][:, -1]
            else:
                # Stack all layers of last token
                # torch.stack(output.hidden_states, dim=1) (bsz, n_layers, seq_len, hidden_size)
                # batch_states (bsz, n_layers, hidden_size)
                batch_states = torch.stack(output.hidden_states, dim=1)[:, :, -1, :].squeeze()
            
            logging.debug(f'shape of batch_states: {batch_states.shape}')
            
            hidden_states_list.append(batch_states.cpu())
            # release memory
            batch_states = None
        # release memory
        self.model = None
        return torch.cat(hidden_states_list, dim=0)
    
    def get_hidden_states(self, text_input, keep_all=True):
        
        text_input = self.preprocess_text(text_input)
        
        # lazy load model
        if self.model is None:
            self.model = load_model(self.model_checkpoint, is_vllm=False)
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
        
        text_input = self.preprocess_text(text_input)
        
        # use vllm to generate text
        if self.vllm_model is None:
            self.vllm_model = load_model(self.model_checkpoint, is_vllm=True)
        generation = self.vllm_model.generate(text_input, self.sampling_params,use_tqdm=False)

        # extract 
        output_text_list = [
            output.text
            for output in generation[0].outputs
        ]
        if normalize:
            return [normalize_answer(output_text) for output_text in output_text_list]
        return output_text_list

    def get_batch_text_generation(self, text_input_list:List[str], normalize=False):
        
        text_input_list = self.preprocess_text(text_input_list)
        
        if self.vllm_model is None:
            self.vllm_model = load_model(self.model_checkpoint, is_vllm=True)
        # use vllm to generate text
        generation = self.vllm_model.generate(text_input_list, self.sampling_params)

        # extract 
        output_text_list = [
            [
                output.text
                for output in gene.outputs
            ]
            for gene in generation
        ]
        if normalize:
            return [
                [normalize_answer(output_text) for output_text in output_text_list]
                for output_text_list in output_text_list
            ]
        if self.is_low_memory:
            # release memory
            self.vllm_model = None
        return output_text_list
    
    def parameters(self):
        return self.model.parameters()
