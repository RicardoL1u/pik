from dataclasses import dataclass, field

@dataclass
class ProbingArguments:
    model_ckpt_path: str = field(metadata={"help": "path to model checkpoint"})
    # debug: bool = field(default=False, metadata={"help": "Debug mode."})
    model_layer_idx: str = field(default=None, metadata={"help": "Model layer index(es), which layer(s) to use. None for all layers, or specify indices separated by commas (e.g., 0,2,4)."})
    input_dim: int = field(default=768, metadata={"help": "Input dimension of the probe."})
    
@dataclass
class DatasetArguments:
    dataset: str = field(metadata={"help": "dataset to use", "choices": ["gsm8k", "trivia_qa_wiki", "commonsense_qa", "bbh"]})
    text_generations_filename: str = field(default="text_generations.csv", metadata={"help": "filename for saving text generations"})
    hidden_states_filename: str = field(default="hidden_states.pt", metadata={"help": "filename for saving hidden states"})

@dataclass
class GenerateArguments:
    model_checkpoint: str = field(default='MODELS/phi-1.5b', metadata={"help": "model checkpoint to use"})
    n_answers_per_question: int = field(default=40, metadata={"help": "number of answers to generate per question"})
    max_new_tokens: int = field(default=256, metadata={"help": "maximum number of tokens to generate per answer"})
    temperature: float = field(default=1.0, metadata={"help": "temperature for generation"})
    pad_token_id: int = field(default=50256, metadata={"help": "pad token id for generation"})
    keep_all_hidden_layers: bool = field(default=True, metadata={"help": "set to False to keep only hidden states of the last layer"})
    hidden_states_filename: str = field(default='hidden_states.pt', metadata={"help": "filename for saving hidden states"})
    text_generations_filename: str = field(default='text_generations.csv', metadata={"help": "filename for saving text generations"})
    input: str = field(default='data/qa_dataset.json', metadata={"help": "filename for input dataset"})


@dataclass
class ScriptArguments:
    example_dir: str = field(required=True, metadata={"help": "directory for saving examples"})
    debug: bool = field(default=False, metadata={"help": "Debug mode."})
    mlp: bool = field(default=False, metadata={"help": "set to True to use MLP activation hook"})
    device: str = field(default='cuda', metadata={"help": "Device to use."})