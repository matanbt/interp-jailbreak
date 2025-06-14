
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from src.models.model_base import ModelBase

# Gemma chat template is based on
# - Official Gemma documentation: https://ai.google.dev/gemma/docs/formatting

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

GEMMA_CHAT_TEMPLATE_WO_CHAT = """<start_of_turn>user
{instruction}"""


GEMMA_REFUSAL_TOKS = [235285, 1718, 107, 1] # ['I', 'It', '<end_of_turn>', '<eos>']
GEMMA_AFFIRM_TOKS = [21404, 1620, 4858, 1917, 14692, 94638]  # ['Sure', '##', 'Here', '```', 'Okay', 'Certainly']

GEMMA2_PRE_INSTRUCT_TOK_COUNT = 4  # <bos><start_of_turn>user
GEMMA_POST_INSTRUCT_TOK_COUNT = 5 # <end_of_turn>\n<start_of_turn>model\n

def format_instruction_gemma_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
    wo_tempalte_chat_suffix: bool = False,  # for special experiments, ablating the chat suffix
):
    if system is not None:
        raise ValueError("System prompts are not supported for Gemma models.")
    else:
        if wo_tempalte_chat_suffix:
            formatted_instruction = GEMMA_CHAT_TEMPLATE_WO_CHAT.format(instruction=instruction)
        else:
            formatted_instruction = GEMMA_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_gemma_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace=True,
    wo_tempalte_chat_suffix=False,
):
    if outputs is not None:
        prompts = [
            format_instruction_gemma_chat(instruction=instruction, output=output, system=system,
                                          include_trailing_whitespace=include_trailing_whitespace,
                                          wo_tempalte_chat_suffix=wo_tempalte_chat_suffix)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma_chat(instruction=instruction, system=system,
                                          include_trailing_whitespace=include_trailing_whitespace,
                                          wo_tempalte_chat_suffix=wo_tempalte_chat_suffix)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

class Gemma2Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cuda",
            attn_implementation="eager",
        ).eval()

        model.requires_grad_(False)

        assert model.dtype in [torch.float16, torch.bfloat16], f"Model dtype is (probably) too big: {model.dtype}"
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'

        assert tokenizer.chat_template is not None, "Tokenizer does not have a chat template"

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_gemma_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)
    
    def _get_refusal_toks(self):
        return GEMMA_REFUSAL_TOKS
    
    def _get_affirm_toks(self):
        return GEMMA_AFFIRM_TOKS

    def _get_before_after_instr_tok_count(self):
        # TODO can automate this (with chat template and tokenizer)
        return GEMMA2_PRE_INSTRUCT_TOK_COUNT, GEMMA_POST_INSTRUCT_TOK_COUNT
    