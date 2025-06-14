import gc
from time import sleep
from typing import List, Tuple, Union, Dict
from src.models.model_factory import construct_model_base
from transformer_lens import HookedTransformer
import torch
import plotly.graph_objects as go
import plotly.express as px
from jaxtyping import Float
import pandas as pd


def load_model(model_name):
    """Loads model with TransformerLens, and returns the model and its base."""
    # load the model:
    model_base = construct_model_base(model_name)

    # load the TL variant:
    tl_model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        hf_model=model_base.model,
        device=model_base.device,
        # dtype=torch.float32 if 'llama' not in model_name else torch.bfloat16,  # TODO remove hack
        dtype=torch.float32,
        # fold_ln=True,
    )
    tl_model.cfg.use_attn_in = True
    tl_model.cfg.use_attn_result = True  # for `attn.result`
    tl_model.cfg.ungroup_grouped_query_attention = True
    tl_model.cfg.use_hook_mlp_in = True

    # HACK to support kv-cahce (due to bug in transformer_lens, when ungrouping attn heads)
    tl_model.cfg.n_key_value_heads = tl_model.cfg.n_heads  

    model_base.tl_model = tl_model
    model_base.model = None
    torch.set_grad_enabled(False)  # as our interp work does not require gradients

    return tl_model, model_base


def to_toks(
    message: str,
    model: HookedTransformer, 
    force_output_prefix:str = None,
    add_template_if_possible=True
):
    # 1.A. Default tokenization:
    input_ids = model.tokenizer.encode(message, return_tensors="pt", add_special_tokens=True)
    start_of_gen_idx = input_ids.shape[-1]

    # 1.B. Chat-template tokenization:
    if add_template_if_possible and model.tokenizer.chat_template is not None:
        wrapped_message = [
            {"role": "user", "content": message},
        ]
        wrapped_message = model.tokenizer.apply_chat_template(wrapped_message, tokenize=False, add_generation_prompt=True)
        start_of_gen_idx = model.tokenizer.encode(wrapped_message, add_special_tokens=False, return_tensors="pt").shape[-1]

        if force_output_prefix is not None:
            wrapped_message = wrapped_message + force_output_prefix
        
        input_ids = model.tokenizer.encode(wrapped_message, add_special_tokens=False, return_tensors="pt")

        if model.tokenizer.bos_token_id not in input_ids:
            print("[WARN] BOS token not found after adding chat template. \n This is unexpected. Consider adding the chat template manually")
            
    elif not add_template_if_possible and model.tokenizer.chat_template is not None:
        print("[WARN] Chat template is available, but not used. Use `add_template_if_possible=True` to use it.")
    elif add_template_if_possible and model.tokenizer.chat_template is None:
        print("[WARN] Chat template is not available, thus not used.")
    
    return input_ids, start_of_gen_idx


def generate(  # TODO move to model
    message: str, 
    model: HookedTransformer=None,
    force_output_prefix: str = None,
    max_new_tokens: int = 256,
    add_template_if_possible: bool = True,
    return_logits: bool = False,
    use_past_kv_cache: bool = False,
) -> Tuple[List[str], str, str]:
    """
    Generates a response for `message` with `model`; 
    Returns: a tuple with
        (i) a list of token ids that includes the given message and generated response, wrapped in the chat template if added;
        (ii) the corresponding full string;
        (iii) the string of the generated response alone (includes the forced string, if provided).

    Notes:
    - `message` is added with special tokens (such as <bos>) if needed.
    - If `add_template_if_possible`, then `model.tokenizer.chat_template`(if not None) is added for 
    wrapping the message prior to generation, including the tokens before generation. 
    - Currently generation is deterministic (`do_sample=False`); can be generalized in the future [TODO].
    - If `force_output_prefix` is not None, it is used as the output prefix.
    """

    # 1. Tokenization:
    input_ids, start_of_gen_idx = to_toks(message, model,
                                          force_output_prefix=force_output_prefix, 
                                          add_template_if_possible=add_template_if_possible)
    
    # 2. Generate:
    full_chat_toks = model.generate(
        input_ids,
        # return_type="tensor",
        return_type="input",
        do_sample=False,
        max_new_tokens=max_new_tokens,
        prepend_bos=False,
        use_past_kv_cache=use_past_kv_cache,
    )
    full_chat_toks = full_chat_toks[0]  # remove batch dim

    # 3. Decode response:
    full_chat_str = model.tokenizer.decode(full_chat_toks, skip_special_tokens=False)
    # 3'. Trim just the response:
    response_toks = full_chat_toks[start_of_gen_idx:]
    response_str = model.tokenizer.decode(response_toks, skip_special_tokens=True)

    # 4. [Optional] Return logits (after output forcing):
    if return_logits:
        # logits = model(input_ids, prepend_bos=False)[0, start_of_gen_idx-1]  # without output forcing
        logits = model(input_ids, prepend_bos=False)[0, -1]  # with output forcing
        return (
            full_chat_toks.tolist(),
            full_chat_str,
            response_str,
            logits,
        )

    return (
        full_chat_toks.tolist(),
        full_chat_str,
        response_str,
    )



def get_idx_slices(
        model_base, adv_message, response_str, 
        adv_suffix_len=20,  # we mostly use GCG with 20 tokens
    ):
    # wraps `get_idx_slices`, but also calculates the affirm length before
    _affirm_slice_data = enrich_with_affirm_length(pd.DataFrame([{'response': response_str}]), model_base)
    affirm_str, affirm_tok_len = _affirm_slice_data.affirm_str.item(), _affirm_slice_data.affirm_tok_len.item()
    
    # given a model-input string (message + 20-tokens-adv-suffix, under chat template) returns the slices for the different parts (message, adv, chat, affirm, bad)
    # input_len = model_base.to_toks(adv_message).shape[1]   # DISABLED: NOTE! HF's tokenizer for Gemma _slightly_ differs from TL's one (from some reason).
    input_len = to_toks(adv_message, model_base.tl_model)[0].shape[1]
    chat_pre_len = model_base.before_instr_tok_count
    chat_suffix_len = model_base.after_instr_tok_count
    affrim_prefix_len = affrim_prefix_len or 20  # 20 mostly fits, when there is an actual `sure, etc.`
    slcs = dict(
        slc_bos=slice(0, 1),  # <BOS>
        slc_chat_pre=slice(1, chat_pre_len),
        slc_instr=slice(chat_pre_len, input_len-adv_suffix_len-chat_suffix_len), 
        slc_adv=slice(input_len-adv_suffix_len-chat_suffix_len, input_len-chat_suffix_len), 
        slc_chat=slice(input_len-chat_suffix_len, input_len), 
        slc_affirm=slice(input_len, input_len+affrim_prefix_len), 
        slc_bad=slice(input_len+affrim_prefix_len, None),

        # additional slices:
        slc_chat3_affirm3=slice(input_len - 3, input_len + 3),
        slc_chat_s2=slice(input_len - 2, input_len),
        slc_input=slice(chat_pre_len, input_len - chat_suffix_len)
        )
    slcs['slc_chat[-1]'] = slice(slcs['slc_chat'].stop - 1, slcs['slc_chat'].stop)  # for the last token in chat slice
    
    return slcs


