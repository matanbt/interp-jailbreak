from typing import List, Tuple
from src.models.model_factory import construct_model_base
from transformer_lens import HookedTransformer
import torch
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
        dtype=torch.float32,
    )
    tl_model.cfg.use_attn_in = True
    tl_model.cfg.use_attn_result = True  # for `attn.result`
    tl_model.cfg.ungroup_grouped_query_attention = True
    tl_model.cfg.use_hook_mlp_in = True

    # HACK to support kv-cahce (due to bug in transformer_lens, when ungrouping attn heads)
    tl_model.cfg.n_key_value_heads = tl_model.cfg.n_heads  

    # add more metadata:
    tl_model.cfg.before_instr_tok_count = model_base.before_instr_tok_count
    tl_model.cfg.after_instr_tok_count = model_base.after_instr_tok_count

    model_base.del_model()  # remove the original model to save memory
    del model_base
    torch.set_grad_enabled(False)  # as our interp work does not require gradients

    return tl_model


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


def generate(
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


def enrich_with_affirm_length(df, tokenizer, set_mock_affirm_prefix=False, pad_to_max_tokens=None):
    """a heuristic to get the affirmative response's prefix length"""
    def heuristic_affirm(x, min_tokens=5):
        new_x = x.split('\n')[0]
        if '.' in new_x:
            new_x = new_x.split('.')[0] + '.'
        if new_x == '':
            # since we want to avoid empty affirm, we take the first 4 tokens
            toks = tokenizer.encode(x, return_tensors="pt", add_special_tokens=False)
            new_x = tokenizer.decode(toks[0, :min_tokens])
        return new_x

    def split_by_2nd_comma_if_exists(x):
        if ':' in x[10:]:
            return x[:10] + x[10:].split(':')[0] + ':'
        if ',' in x[10:]:
            return x[:10] + x[10:].split(',')[0] + ','

        return x

    def trim_to_max_tokens(x, max_tokens=20):
        toks = tokenizer.encode(x, return_tensors="pt", add_special_tokens=False)
        if toks.shape[1] > max_tokens:  # if too long - trim
            new_x = tokenizer.decode(toks[0, :max_tokens])
            # assert x.startswith(new_x), f"trimming failed: {x} -> {new_x}"
            return new_x
        return x

    if set_mock_affirm_prefix:
        # we set a fixed affirm prefix
        # df['affirm_str'] = "Sure, here's what you need:\n"
        df['affirm_str'] = "Sure, here is exactly what you need to do.\n"
    else:
        df['affirm_str'] = df.response.apply(heuristic_affirm).apply(split_by_2nd_comma_if_exists).apply(trim_to_max_tokens)
    df['affirm_tok_len'] = df.affirm_str.apply(lambda x: tokenizer.encode(x, return_tensors="pt", 
                                                                          add_special_tokens=False).shape[1])

    # [OPTIONAL] add padding tokens for sequences shorther than `pad_to_max_tokens`, following `affirm_tok_len`
    if pad_to_max_tokens is not None:
        df.loc[df.affirm_tok_len < pad_to_max_tokens, 'affirm_str'] = df.loc[df.affirm_tok_len < pad_to_max_tokens].apply(
            # TODO [THIS IS HACKY AND UGLY] find another way!
            lambda x: x.affirm_str + ('\n-' * (pad_to_max_tokens - x.affirm_tok_len)), axis=1)  # TODO this is an ugly hack.
        df['affirm_tok_len'] = df.affirm_str.apply(lambda x: tokenizer.encode(x, return_tensors="pt", add_special_tokens=False).shape[1])

    return df


def get_idx_slices(
        model, message, suffix, response_str="", 
    ):
    # wraps `get_idx_slices`, but also calculates the affirm length before
    _affirm_slice_data = enrich_with_affirm_length(pd.DataFrame([{'response': response_str}]), model.tokenizer)
    affirm_str, affirm_tok_len = _affirm_slice_data.affirm_str.item(), _affirm_slice_data.affirm_tok_len.item()
    affirm_tok_len = affirm_tok_len or 20  # default to 20 if not set
    
    # given a model-input string (message + 20-tokens-adv-suffix, under chat template) returns the slices for the different parts (message, adv, chat, affirm, bad)
    input_len = to_toks(message + suffix, model)[0].shape[1]
    chat_pre_len = model.cfg.before_instr_tok_count
    adv_suffix_len = model.tokenizer.encode(suffix, return_tensors="pt", add_special_tokens=False).shape[1]
    chat_suffix_len = model.cfg.after_instr_tok_count
    slcs = dict(
        bos=slice(0, 1),  # <BOS>
        chat_pre=slice(1, chat_pre_len),
        instr=slice(chat_pre_len, input_len-adv_suffix_len-chat_suffix_len), 
        adv=slice(input_len-adv_suffix_len-chat_suffix_len, input_len-chat_suffix_len), 
        chat=slice(input_len-chat_suffix_len, input_len), 
        affirm=slice(input_len, input_len+affirm_tok_len), 
        bad=slice(input_len+affirm_tok_len, None),

        # additional slices:
        chat3_affirm3=slice(input_len - 3, input_len + 3),
        chat_s2=slice(input_len - 2, input_len),
        input=slice(chat_pre_len, input_len - chat_suffix_len)
        )
    slcs['chat[-1]'] = slice(slcs['chat'].stop - 1, slcs['chat'].stop)  # for the last token in chat slice
    slcs['chat[:-1]'] = slice(slcs['chat'].start, slcs['chat'].stop - 1)
    
    return slcs


