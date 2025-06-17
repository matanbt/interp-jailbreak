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
    force_output_prefix=None,
    apply_sanity_checks: bool = False,

    # dominance score calculation:
    add_dominance_calc: bool = False,
    given_dir: Float[torch.Tensor, "n_layer d_model"] = None,
    selected_dom_scores: List[str] = ['Y@attn'],
):
    """
    Returns selected hidden states, and TL's cache object.
    Optionally, if `return_labels`, returns labels and layer count per hook.
    """
    gc.collect()
    torch.cuda.empty_cache()
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
        'X_in': 'blocks.{layer}.attn.hook_X_in',  # layer, head, src_pos, hidden_size
        'X_WVO': 'blocks.{layer}.attn.hook_X_WVO', # layer, head, src_pos, d_model
        'Y': 'blocks.{layer}.hook_Y_out',  # layer, head, seq_len (dst), seq_len (src), hidden_size
    }
    hs_dict, hs_dict_labels = {}, {}
    
    _orig_use_attn_fine_grained = model.cfg.use_attn_fine_grained
    model.set_use_attn_fine_grained(True)  # temporarily enable fine-grained attention hooks
    _, cache = model.run_with_cache(toks)
    model.set_use_attn_fine_grained(_orig_use_attn_fine_grained)  # restore original setting
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
        hs_dict[hook_name] = hidden_states.cpu()
        hs_dict_labels[hook_name] = labels

    ## save embed, mlps and attn separately
    hs_dict['decompose_resid__embed'] = hs_dict['decompose_resid'][0]  # embed
    hs_dict_labels['decompose_resid__embed'] = ['embed']
    hs_dict['decompose_resid__attns'] = hs_dict['decompose_resid'][list(range(1, 2 * n_layers, 2))]  # attns
    hs_dict_labels['decompose_resid__attns'] = [f"attn l{layer}" for layer in range(n_layers)]
    hs_dict['decompose_resid__mlps'] = hs_dict['decompose_resid'][list(range(2, 2 * n_layers + 1, 2))]  # mlps
    hs_dict_labels['decompose_resid__mlps'] = [f"mlp l{layer}" for layer in range(n_layers)]

    ## coarse version of decompose_resid:
    decomp_resid_coar = torch.zeros((n_layers, len(toks), model.cfg.d_model))
    decomp_resid_coar[0] = hs_dict['decompose_resid__embed']  # embed
    for layer in range(0, n_layers):
        decomp_resid_coar[layer] += hs_dict['decompose_resid__mlps'][layer]
        decomp_resid_coar[layer] += hs_dict['decompose_resid__attns'][layer]
    hs_dict['decompose_resid_coar'] = decomp_resid_coar
    hs_dict['decompose_resid_coar'] = [f"decomp_resid_coar l{layer}" for layer in range(n_layers)]
    
    ## legacy names:
    hs_dict['resids_pre_attn'] = hs_dict['X_in']
    hs_dict['Y_all_norm'] = hs_dict['Y']
    hs_dict['C_all_norm'] = hs_dict['X_WVO'].unsqueeze(2).repeat(1, 1, hs_dict['X_WVO'].shape[2], 1, 1)  # repeat dst_pos --> layer, head, src_pos, dst_pos, d_model
    hs_dict['A'] = hs_dict['attn_pattern']  # layer, head, dst_pos, src_pos

    if apply_sanity_checks:
        print(
            "decomp vs Y:", (hs_dict['decompose_resid__attns'][5] - hs_dict['Y'][5].sum(dim=(0,2))).abs().max(), "\n"
            "decomp (w/ y) vs resid:", ((hs_dict['decompose_resid__embed'] + hs_dict['decompose_resid__mlps'].sum(dim=0) + hs_dict['Y'].sum(dim=(0,1,3))) - hs_dict['resid'][-1]).abs().max()
        )
    
    ## dominance score calculation:
    if add_dominance_calc:
        hs_dict = _get_dominance_scores_full(
            hs_dict,
            given_dir=given_dir,
            selected_dom_scores=selected_dom_scores,
        )
    
    if return_labels:
        return hs_dict, hs_dict_labels, cache
    return hs_dict, cache


def _get_dominance_scores_full(
    hs_dict: Dict[str, torch.Tensor],
    selected_dom_scores=['Y@attn'], # list of keys to calculate dominance scores for
    given_dir: Float[torch.Tensor, "n_layer d_model"] = None,  # direction to calculate dominance in
    dst_slc: slice = slice(None, None),  # dst slice for the attention (default to all)
):
    dst_slc = slice(None, None)
    def get_dot_with_vectors(
        main_vecs: Float[torch.Tensor, 'n_layer head dst src d_model'],  # main vectors to calculate the dot product with
        ref_vecs: Float[torch.Tensor, 'n_layer dst d_model'],
    ):    
        main_vecs =main_vecs[:, :, dst_slc]
        ref_vecs = ref_vecs[:, dst_slc].unsqueeze(-2).unsqueeze(-2).transpose(1, 2)  # layer, head[1], dst, src[1], d_model
        # main_vecs.shape, ref_vecs.shape  # -> layer, head, dst, src, d_model

        # perform dot product of the last dimension
        dot_prod_vals = torch.einsum('lhtsd,lktkd->lhts', main_vecs, ref_vecs)  # layer, head, dst, src
        
        # normalize per layer and (dst) token position
        dot_prod_vals = dot_prod_vals / torch.norm(ref_vecs, dim=-1).pow(2)  # layer, head, dst, src

        return dot_prod_vals

    dom_score_to_args = {
        # TODO verify the following two:
        # 'Y@resid': {'out_vecs': hs_dict['resids_pre_attn']},  
        # 'Y@dcmp_resid': {'out_vecs': hs_dict['decompose_resid_coar']},

        'Y@attn': {'main_vecs': hs_dict['Y'], 'ref_vecs': hs_dict['attn']},
        '(X@W_VO)@attn': {'main_vecs': hs_dict['X_WVO'], 'ref_vecs': hs_dict['attn'], 'replace_Y_with_XWVO': True},
    }
    for dom_score_key in selected_dom_scores:
        if dom_score_key not in dom_score_to_args:
            raise ValueError(f"Unknown dominance score key: {dom_score_key}. Available keys: {list(dom_score_to_args.keys())}")
        hs_dict[dom_score_key] = get_dot_with_vectors(**dom_score_to_args[dom_score_key])

    if given_dir is not None:  # calculate direction-specific dominance scores
        hs_dict[f"Y@dir"] = torch.einsum(
            '...d, d -> ...' if len(given_dir.shape) == 1 else 'lhtsd, ld -> lhts',
            hs_dict['Y'][:, :, dst_slc],
            (given_dir / torch.norm(given_dir, dim=-1, keepdim=True)).to(hs_dict['Y'])
        )

    return hs_dict


def get_dominance_scores(
    model: HookedTransformer,
    msg: str, suffix: str,
    hs_dict: Dict[str, torch.Tensor] = None,
    dst_slc_name: str = 'chat[-1]',
    dominance_metric: str ='Y@attn', 
    dominance_metric_flavor: str = 'sum',  # 'sum', 'sum-top0.1'
) -> Float[torch.Tensor, "n_layer"]:
    """
    Self-contained function to calculate dominance scores for a given message and suffix.
    """

    slcs = get_idx_slices(
        model, msg, suffix
    )
    
    # calculate the hijacking score:
    if hs_dict is None:
        hs_dict, _ = get_model_hidden_states(
            model, msg + suffix,
            add_dominance_calc=True,
            selected_dom_scores=[dominance_metric]
        )

    src_slc_names = ['bos', 'chat_pre', 'instr', 'adv', 'chat[:-1]', 'chat[-1]']

    dominance_score_dict = {}
    for src_slc_name in src_slc_names:
        dominance_score_dict[src_slc_name] = (
            hs_dict[dominance_metric]  # layer, head, dst, src
                .sum(dim=1)[:, slcs[dst_slc_name], slcs[src_slc_name]]
        )  # n_layer, dst, src

        if dominance_metric_flavor == 'sum-top0.1':
            raise NotImplementedError("`sum-top0.1` flavor is not implemented yet.")
        elif dominance_metric_flavor == 'sum':
            dominance_score_dict[src_slc_name] = dominance_score_dict[src_slc_name].sum(dim=(1, 2))  # n_layer, dst, src -> n_layer

    return dominance_score_dict


# TODO another function to support attn@resid + mlp@resid + embed@resid
