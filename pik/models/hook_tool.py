import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    GPT2LMHeadModel,
    AutoConfig,
    AutoTokenizer
)

class HookMLPActivation():
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.register_mlp_activation_hook()

    def register_mlp_activation_hook(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        def capture_input(name):
            def hook(module, input):
                self.activations[name] = input[0].detach()
            return hook

        # Check if the model is a LlamaForCausalLM instance
        if isinstance(self.model, LlamaForCausalLM):
            for i, layer in enumerate(self.model.model.layers):
                layer_name = f'mlp_act_layer_{i}'
                layer.mlp.down_proj.register_forward_pre_hook(capture_input(layer_name))
        
        # Check if the model is a GPT2LMHeadModel instance
        elif isinstance(self.model, GPT2LMHeadModel):
            for i, layer in enumerate(self.model.transformer.h):
                layer_name = f'mlp_act_layer_{i}'
                layer.mlp.act.register_forward_hook(get_activation(layer_name))

            
if __name__ == '__main__':
    
    ckpt_path = '/data3/MODELS/llama-65b-hf'
    
    config = AutoConfig.from_pretrained(ckpt_path)
    # HACK: set the activation to Rel
    config.activation_function = 'relu'
    # Load model from config
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, 
                                                 config=config,
                                                 device_map='auto',
                                                attn_implementation="flash_attention_2",
                                                torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    hooker = HookMLPActivation(model)
    hooker.register_mlp_activation_hook()
    test_text = "Hello, my dog is cute"
    input_ids = tokenizer.encode(test_text, return_tensors='pt')
    output = model(input_ids)
    print(hooker.activations['mlp_act_layer_0'])
    print("Shape of the activation: ", hooker.activations['mlp_act_layer_0'].shape)
    # clear activations
    # hooker.activations = {}
    test_text = "My cat is small"
    input_ids = tokenizer.encode(test_text, return_tensors='pt')
    output = model(input_ids)
    print(hooker.activations['mlp_act_layer_0'])
    print("Shape of the activation: ", hooker.activations['mlp_act_layer_0'].shape)
    
    
    