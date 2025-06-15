import gc
from time import sleep
from typing import List, Tuple, Union, Dict
from src.interp.utils import to_toks
from src.models.model_factory import construct_model_base
from transformer_lens import HookedTransformer
import torch
import plotly.graph_objects as go
import plotly.express as px
from jaxtyping import Float
import pandas as pd
from src.interp.utils import get_idx_slices


def get_model_hidden_states(
    model: HookedTransformer,
    toks: Union[List[int], str],
    return_labels: bool = False,
    calc_special_matrices: bool = False,
    add_dominance_calc: bool = False,
    force_output_prefix=None,
    Y_ablation_variant: str = '',
    calc_batch_size: int = 2,
    given_dir: Float[torch.Tensor, "n_layer d_model"] = None,  # TODO: support this

    apply_sanity_checks: bool = False,
):
    """
    Returns selected hidden states, and TL's cache object.
    Optionally, if `return_labels`, returns labels and layer count per hook.
    """
    calc_special_matrices = False  # currently done in TL and disabled
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    # 0. tokenize if needed:
    if isinstance(toks, str):
        toks, _ = to_toks(toks, model, force_output_prefix=force_output_prefix)
        toks = toks[0].clone().detach()  # remove batch dim, and clone to avoid inplace changes
    if not isinstance(toks, torch.Tensor):
        toks = torch.tensor(toks)

    # 1. Cache the relevant hidden states:
    supported_hooks = {  # translate to TransformerLens hooks:
        'resid': 'blocks.{layer}.hook_resid_post',
        'resid_pre': 'blocks.{layer}.hook_resid_pre',
        'decompose_resid': 'decompose_resid',
        
        ## attention:
        'attn': 'blocks.{layer}.hook_attn_out',
        'attn_result': 'blocks.{layer}.attn.hook_result',  # # layer, head, seq_len, hidden_size
        'attn_pattern': 'blocks.{layer}.attn.hook_pattern',  # layer, head, seq_len (dst), seq_len (src)

        ## mlp:
        'mlp': 'blocks.{layer}.hook_mlp_out',

        # fine grained attn:
        # 'resids_pre_attn': 'blocks.{layer}.ln1.hook_normalized_post', # layer, seq_len, hidden_size  #[UNUSED]
        'X_in': 'blocks.{layer}.attn.hook_X_in',  # layer, head, src_pos, hidden_size
        'X_WVO': 'blocks.{layer}.attn.hook_X_WVO', # layer, head, src_pos, d_model
        'Y': 'blocks.{layer}.hook_Y_out',  # layer, head, seq_len (dst), seq_len (src), hidden_size
    }
    out_cache, out_labels = {}, {}
    
    model.set_use_attn_fine_grained(True)  # temporarily enable fine-grained attention hooks
    _, cache = model.run_with_cache(toks)
    model.set_use_attn_fine_grained(False)
    cache = cache.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    for hook_name, hook_tl_name in supported_hooks.items():
        if hook_name == 'decompose_resid':
            hidden_states, labels = cache.decompose_resid(return_labels=True)
            hidden_states = hidden_states.squeeze(1) # layer, seq, hidden
        else:
            n_layers = model.cfg.n_layers
            hidden_states = torch.stack([
                    cache[hook_tl_name.format(layer=layer)] for layer in range(n_layers)
                    ], dim=1).squeeze(0)  # layer, seq, [head] hidden
            labels = [f"{hook_name}-l{layer}" for layer in range(n_layers)]
        
        # save all:
        out_cache[hook_name] = hidden_states.cpu()
        out_labels[hook_name] = labels

    ## save embed, mlps and attn separately
    out_cache['decompose_resid__embed'] = out_cache['decompose_resid'][0]  # embed
    out_labels['decompose_resid__embed'] = ['embed']
    out_cache['decompose_resid__attns'] = out_cache['decompose_resid'][list(range(1, 2 * n_layers, 2))]  # attns
    out_labels['decompose_resid__attns'] = [f"attn l{layer}" for layer in range(n_layers)]
    out_cache['decompose_resid__mlps'] = out_cache['decompose_resid'][list(range(2, 2 * n_layers + 1, 2))]  # mlps
    out_labels['decompose_resid__mlps'] = [f"mlp l{layer}" for layer in range(n_layers)]

    ## coarse version of decompose_resid:
    decomp_resid_coar = torch.zeros((n_layers, len(toks), model.cfg.d_model))
    decomp_resid_coar[0] = out_cache['decompose_resid__embed']  # embed
    for layer in range(0, n_layers):
        decomp_resid_coar[layer] += out_cache['decompose_resid__mlps'][layer]
        decomp_resid_coar[layer] += out_cache['decompose_resid__attns'][layer]
    out_cache['decompose_resid_coar'] = decomp_resid_coar
    out_labels['decompose_resid_coar'] = [f"decomp_resid_coar l{layer}" for layer in range(n_layers)]
    
    ## add my attention reformulations:
    if calc_special_matrices:
        out_cache['C'], out_cache['C_all'], out_cache['Y_all'] = my_attention_reformulations(
            model, 
            out_cache['resids_pre_attn'], 
            out_cache['attn_pattern'],
            ablation_variant=Y_ablation_variant,
            calc_batch_size=calc_batch_size,
        )
        ## i. Matrix C:
        out_cache['C'] = out_cache['C'].cpu()  # layer, head, seq_len[src], d_model
        
        # flatten C's first two dims, and record the labels:
        out_cache['C_flat'] = out_cache['C'].flatten(0, 1).cpu()
        out_labels['C_flat'] = [f"C l{layer}-h{head}" for layer in range(n_layers) for head in range(model.cfg.n_heads)]

        ## ii. normalized Y:
        # out_cache['Y_all']  # layer, head, seq_len[dst], seq_len[src], d_model
        out_cache['Y_all_norm'] = torch.zeros_like(out_cache['Y_all'])
        out_cache['C_all_norm'] = torch.zeros_like(out_cache['C_all'])

        for layer in range(n_layers):
            y_in_layer = out_cache['Y_all'][layer]            # head, seq_len[dst], seq_len[src], d_model
            c_in_layer = out_cache['C_all'][layer]            # head, seq_len[dst], seq_len[src], d_model
            
            # we perform RMSNorm per layer's attn (Gemma's) [https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/components/rms_norm.py#L15]
            scale = cache[f'blocks.{layer}.ln1_post.hook_scale'].to(y_in_layer.device)[0].unsqueeze(-1).unsqueeze(0)  # 1, seq_len, 1, 1  (for: head, seq_dst, seq_src, d_model)
            w = model.blocks[layer].ln1_post.w.to(y_in_layer.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1, 1, 1 d_model  (for: head, seq_dst, seq_src, d_model)

            y_in_layer = (y_in_layer / scale) * w
            c_in_layer = (c_in_layer / scale) * w

            out_cache['Y_all_norm'][layer] = y_in_layer
            out_cache['C_all_norm'][layer] = c_in_layer

        out_cache['Y_all_norm'] = out_cache['Y_all_norm'].cpu()
        out_labels['Y_all_norm'] = [f"Y_all_norm l{layer}-h{head}" for layer in range(n_layers) for head in range(model.cfg.n_heads)]

        out_cache['C_all_norm'] = out_cache['C_all_norm'].cpu()

        # an optional transpose (keeps dims consistent)
        out_cache['Y_all_norm_T'] = out_cache['Y_all_norm'].transpose(2, 3).cpu()  # layer, head, seq_len[src], seq_len[dst], d_model
        out_labels['Y_all_norm_T'] = out_labels['Y_all_norm']

        # assert torch.allclose(out_cache['decompose_resid__attns'][5], out_cache['Y_all_norm_T'][5].sum(dim=(0,1)).cuda()), "Y_all_norm calc probably gone wrong."

        ## iii. Matrix Y (changes according to the dst token in focus)
        def _calc_Y(
                _Y_dst_focus=-1
            ):
            return out_cache['Y_all_norm'][..., _Y_dst_focus, :, :].flatten(0, 1).cpu()
        
        out_cache['Y'] = _calc_Y
        out_labels['Y'] = [f"Y l{layer}-h{head}" for layer in range(n_layers) for head in range(model.cfg.n_heads)]

        if apply_sanity_checks:
            print(
                "decomp vs Y:", (out_cache['decompose_resid__attns'][5] - out_cache['Y_all_norm_T'][5].sum(dim=(0,1))).abs().max(), "\n"
                "decomp (w/ y) vs resid:", ((out_cache['decompose_resid__embed'] + out_cache['decompose_resid__mlps'].sum(dim=0) + out_cache['Y_all_norm_T'].sum(dim=(0,1,2))) - out_cache['resid'][-1]).abs().max()
            )

    # legacy names:
    out_cache['resids_pre_attn'] = out_cache['X_in']
    out_cache['Y_all_norm'] = out_cache['Y']
    out_cache['C_all_norm'] = out_cache['X_WVO'].unsqueeze(2).repeat(1, 1, out_cache['X_WVO'].shape[2], 1, 1)  # repeat dst_pos --> layer, head, src_pos, dst_pos, d_model
    out_labels['Y_all_norm'] = [f"Y_all_norm l{layer}-h{head}" for layer in range(n_layers) for head in range(model.cfg.n_heads)]

    if apply_sanity_checks:
        print(
            "decomp vs Y:", (out_cache['decompose_resid__attns'][5] - out_cache['Y'][5].sum(dim=(0,2))).abs().max(), "\n"
            "decomp (w/ y) vs resid:", ((out_cache['decompose_resid__embed'] + out_cache['decompose_resid__mlps'].sum(dim=0) + out_cache['Y'].sum(dim=(0,1,3))) - out_cache['resid'][-1]).abs().max()
        )
    
    if add_dominance_calc:
        dst_slc = slice(None, None)
        def get_dot_with_vectors(
            _out_vecs, #: Float['layer, dst, d_model'],
            replace_Y_with_XWVO=False,  # HACK to support replacing Y
        ):    
            resid = _out_vecs[:, dst_slc].unsqueeze(-2).unsqueeze(-2).transpose(1, 2)  # layer, head[1], dst, src[1], d_model
            Y = out_cache['Y_all_norm'][:, :, dst_slc]
            # Y.shape, resid.shape  # -> layer, head, dst, src, d_model
            if replace_Y_with_XWVO:
                Y = out_cache['C_all_norm'][:, :, dst_slc]

            # perform dot product of the last dimension
            dot_prod_vals = torch.einsum('lhtsd,lktkd->lhts', Y, resid)  # layer, head, dst, src
            
            # normalize per layer and (dst) token position
            dot_prod_vals = dot_prod_vals / torch.norm(resid, dim=-1).pow(2)  # layer, head, dst, src

            return dot_prod_vals

        # TODO define what to calculate
        # out_cache['Y@resid'] = get_dot_with_vectors(out_cache['resid']) # layer, head, dst, src
        out_cache['Y@attn'] = get_dot_with_vectors(out_cache['attn'])  # layer, head, dst, src
        # out_cache['(X@W_VO)@attn'] =  get_dot_with_vectors(out_cache['attn'], replace_Y_with_XWVO=True)
        # out_cache['Y@dcmp_resid'] = get_dot_with_vectors(out_cache['decompose_resid_coar'])
        out_cache['A'] = out_cache['attn_pattern']  # layer, head, dst, src

        if given_dir is not None:
            out_cache[f"Y@dir"] = torch.einsum(
                '...d, d -> ...' if len(given_dir.shape) == 1 else 'lhtsd, ld -> lhts',
                out_cache['Y_all_norm'][:, :, dst_slc],
                (given_dir / torch.norm(given_dir, dim=-1, keepdim=True)).to(out_cache['Y_all_norm'])
            )
    
    if return_labels:
        return out_cache, out_labels, cache
    return out_cache, cache


def get_dom_scores(
    model: HookedTransformer,
    msg: str, suffix: str,
    hs_dict: Dict[str, torch.Tensor] = None,
    dst_slc_name: str = 'chat[-1]',
    hijacking_metric: str ='Y@attn',  # 'Y@resid', 'Y@attn', 'X@WVO@attn', 'Y@dcmp_resid', 'Y@dir
    hijacking_metric_flavor: str = 'sum',  # 'sum', 'sum-top0.1'
) -> Float[torch.Tensor, "n_layer"]:

    slcs = get_idx_slices(
        model, msg, suffix
    )
    
    # calculate the hijacking score:
    if hs_dict is None:
        hs_dict, _ = get_model_hidden_states(
            model, msg + suffix, add_dominance_calc=True, calc_special_matrices=False,
        )

    src_slc_names = ['bos', 'chat_pre', 'instr', 'adv', 'chat[:-1]', 'chat[-1]']  # 'input',

    hijacking_score_dict = {}
    for src_slc_name in src_slc_names:
        hijacking_score_dict[src_slc_name] = (
            hs_dict[hijacking_metric]  # layer, head, dst, src
                .sum(dim=1)[:, slcs[dst_slc_name], slcs[src_slc_name]]
        )  # n_layer, dst, src

        if hijacking_metric_flavor == 'sum-top0.1':
            raise NotImplementedError("`sum-top0.1` flavor is not implemented yet.")
        elif hijacking_metric_flavor == 'sum':
            hijacking_score_dict[src_slc_name] = hijacking_score_dict[src_slc_name].sum(dim=(1, 2))  # n_layer, dst, src -> n_layer

    return hijacking_score_dict


# TODO another function to support attn@resid + mlp@resid + embed@resid
