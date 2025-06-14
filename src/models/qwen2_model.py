
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor

from src.models.model_base import ModelBase


QWEN_REFUSAL_TOKS = [40, 2121] # ['I', 'As']
QWEN_AFFIRM_TOKS = [39814]  # ['Sure']


def format_instruction_qwen_chat(
    tokenizer: AutoTokenizer,
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
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

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
    **kwargs,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(tokenizer=tokenizer, instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(tokenizer=tokenizer, instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


class Qwen2Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        model_kwargs = {}
        if 'qwen-1_8b' in model_path:
            model_kwargs.update({"use_flash_attn": True, 'trust_remote_code': True})
            if dtype != "auto":
                model_kwargs.update({
                    "bf16": dtype==torch.bfloat16,
                    "fp16": dtype==torch.float16,
                    "fp32": dtype==torch.float32,
                })

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # torch_dtype=dtype,
            torch_dtype="auto",
            device_map="auto",
            **model_kwargs,
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = 'left'

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_affirm_toks(self):
        return QWEN_AFFIRM_TOKS
    
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
