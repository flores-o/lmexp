from transformers import AutoTokenizer, AutoModelForCausalLM
from lmexp.generic.activation_steering.steerable_model import SteerableModel
from lmexp.generic.tokenizer import Message, Tokenizer
import torch
import os
from dotenv import load_dotenv
from lmexp.models.model_helpers import MODEL_GEMMA_2
from transformers.models.gemma import GemmaForCausalLM
from transformers import BitsAndBytesConfig

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class Gemma2Tokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_GEMMA_2, token=HUGGINGFACE_TOKEN
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")

    def decode(self, tensor):
        return self.tokenizer.decode(tensor)

    def chat_format(
        self, messages: list[Message], add_generation_prompt: bool = False
    ) -> str:
        valid_roles = {"user", "model"}
        assert all(m["role"] in valid_roles for m in messages)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    @property
    def pad_token(self):
        return self.tokenizer.pad_token_id


class SteerableGemma2(SteerableModel):
    def __init__(self, load_in_8bit: bool = False):
        if load_in_8bit:
            self.model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
                MODEL_GEMMA_2,
                token=HUGGINGFACE_TOKEN,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )
        else:
            self.model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
                MODEL_GEMMA_2,
                token=HUGGINGFACE_TOKEN,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_GEMMA_2, token=HUGGINGFACE_TOKEN
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def n_layers(self):
        return len(self.model.model.layers)

    def get_module_for_layer(self, layer: int) -> torch.nn.Module:
        return self.model.model.layers[layer]

    def forward(self, x: torch.Tensor):
        return self.model(x).logits

    def sample(self, tokens: torch.Tensor, max_n_tokens: int) -> torch.Tensor:
        attention_mask = torch.ones_like(tokens)
        return self.model.generate(
            tokens,
            max_length=max_n_tokens,
            attention_mask=attention_mask,
            stop_strings=["<|eot_id|>"],
            pad_token_id=self.model.config.eos_token_id,
            tokenizer=self.tokenizer,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    @property
    def resid_dim(self) -> int:
        return self.model.model.layers[0].hidden_size

    @property
    def transformer_module(self) -> torch.nn.Module:
        return self.model.model
