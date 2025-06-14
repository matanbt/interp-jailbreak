
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from src.refusal_direction.pipeline.model_utils.model_base import ModelBase


def format_instruction_llama3_chat(
    tokenizer: AutoTokenizer,
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    message_dicts = []
    if system is not None:
        message_dicts.append({"role": "system", "content": system})
    else:
        message_dicts.append({"role": "user", "content": instruction})
    formatted_instruction = tokenizer.apply_chat_template(message_dicts, tokenize=False, add_generation_prompt=True)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_llama3_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
    **kwargs
):
    if outputs is not None:
        prompts = [
            format_instruction_llama3_chat(tokenizer, instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llama3_chat(tokenizer, instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        add_special_tokens=False,  # template already takes care of <bos> kind of tokens
        return_tensors="pt",
    )

    return result


LLAMA3_REFUSAL_TOKS = [40] # 'I'
LLAMA3_AFFIRM_TOKS = [40914]

class Llama3Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="cuda",
        ).eval()

        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_llama3_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # return self.tokenizer.encode(LLAMA3_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)
        pass

    def _get_refusal_toks(self):
        return LLAMA3_REFUSAL_TOKS

    def _get_affirm_toks(self):
        return LLAMA3_AFFIRM_TOKS
    
    def _get_before_after_instr_tok_count(self):
        str_before, str_after = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "DUMMY_TXT_FOR_SPLIT"}], 
                tokenize=False, add_generation_prompt=True
                ).split("DUMMY_TXT_FOR_SPLIT")
        if self.tokenizer.bos_token and str_before.startswith(self.tokenizer.bos_token):
            str_before = str_before.replace(self.tokenizer.bos_token, "")
        n_toks_before = len(self.tokenizer(str_before)["input_ids"])  # includes <bos> token
        n_toks_after = len(self.tokenizer(str_after, add_special_tokens=False)["input_ids"])
        return n_toks_before, n_toks_after
