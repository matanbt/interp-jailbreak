import gc
from typing import List, Tuple, Union, Dict
from src.interp.utils import to_toks
from transformer_lens import HookedTransformer
import torch
from jaxtyping import Float
from src.interp.utils import get_idx_slices


def get_model_hidden_states(
    model: HookedTransformer,
    toks: Union[List[int], str],
    return_labels: bool = False,
    force_output_prefix: str = None,
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
        'attn': 'blocks.{layer}.hook_attn_out',  # layer, seq_len, hidden_size
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
    
    ## legacy names:
    hs_dict['resids_pre_attn'] = hs_dict['X_in']
    hs_dict['X_WVO'] = hs_dict['X_WVO'].unsqueeze(2)
    hs_dict['A'] = hs_dict['attn_pattern']  # layer, head, dst_pos, src_pos

    if apply_sanity_checks:
        print(
            "decomp vs Y:", (hs_dict['decompose_resid__attns'][5] - hs_dict['Y'][5].sum(dim=(0,2))).abs().max(), "\n"
            "decomp (w/ y) vs resid:", ((hs_dict['decompose_resid__embed'] + hs_dict['decompose_resid__mlps'].sum(dim=0) + hs_dict['Y'].sum(dim=(0,1,3))) - hs_dict['resid'][-1]).abs().max()
        )
    
    ## dominance score calculation:
    if add_dominance_calc:
        hs_dict = _calculate_hooks_for_dom_scores(
            hs_dict,
            given_dir=given_dir,
            selected_dom_scores=selected_dom_scores,
        )
    
    if return_labels:
        return hs_dict, hs_dict_labels, cache
    return hs_dict, cache


def _calculate_hooks_for_dom_scores(
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

        return dot_prod_vals.cpu()

    dom_score_to_args = {
        'Y@resid': (hs_dict['Y'], hs_dict['resid']),
        'Y@dcmp_resid': (hs_dict['Y'], hs_dict['decompose_resid_coar']),
        'Y@attn': (hs_dict['Y'], hs_dict['attn']),
        '(X@W_VO)@attn': (hs_dict['X_WVO'], hs_dict['attn'])
    }
    for dom_score_key in set(selected_dom_scores) & set(dom_score_to_args.keys()):
        hs_dict[dom_score_key] = get_dot_with_vectors(*dom_score_to_args[dom_score_key])

    if given_dir is not None and 'Y@dir' in selected_dom_scores:
        # calculate direction-specific dominance scores
        hs_dict[f"Y@dir"] = torch.einsum(
            '...d, d -> ...' if len(given_dir.shape) == 1 else 'lhtsd, ld -> lhts',
            hs_dict['Y'][:, :, dst_slc],
            (given_dir / torch.norm(given_dir, dim=-1, keepdim=True)).to(hs_dict['Y'])
        ).cpu()
    
    ## norm-based scores:
    hs_dict['norm(X)'] = hs_dict['X_in'].unsqueeze(2).norm(dim=-1).cpu()  # layer, head, [1], src
    hs_dict['norm(Y)'] = hs_dict['Y'].norm(dim=-1).cpu()  # layer, head, dst, src

    return hs_dict


def get_dominance_scores(
    model: HookedTransformer,
    msg: str, suffix: str,
    hs_dict: Dict[str, torch.Tensor] = None,
    dst_slc_name: str = 'chat[-1]',
    src_slc_names: Tuple[str] = ('bos', 'chat_pre', 'instr', 'adv', 'chat[:-1]', 'chat[-1]'),
    dominance_metric: str ='Y@attn', 
    dominance_metric_flavor: str = 'sum',  # 'sum', 'sum-top_q'
    dominance_metric_flavor_q: float = 0.1,  # used only for 'sum-top0.1' flavor
    aggr_all_layers: bool = False,
) -> Dict[str, List[float]]:
    """
    Self-contained function to calculate dominance scores for a given message and suffix.
    Args:
        model: the HookedTransformer model to use.
        msg: the message string to use.
        suffix: the suffix string to use.
        hs_dict: pre-computed hidden states dictionary, if None, will be computed.
        dst_slc_name: the name of the destination slice to use (e.g. 'chat[-1]').
        src_slc_names: a tuple of source slice names to calculate dominance scores for (e.g. `('instr', 'adv')`).
        dominance_metric: the metric to use for dominance calculation (e.g. 'Y@attn').
        dominance_metric_flavor: the flavor of the metric to use (e.g. 'sum', 'sum-top_q').
        dominance_metric_flavor_q: the quantile to use for 'sum-top_q' flavor.
        aggr_all_layers: if True, will aggregate the scores across all layers, otherwise returns scores per layer.
    Return: a dict, each entry maps the source slice name (e.g. 'instr', 'adv') to a dominance score list of length n_layers (or a singleton list if `aggr_all_layers`).
    """

    # varify supported inputs:
    assert dominance_metric in ['Y@attn', 'Y@resid', 'Y@dcmp_resid', '(X@W_VO)@attn', 'norm(X)', 'norm(Y)', 'Y@dir', 'A']
    assert dominance_metric_flavor in ['sum', 'sum-top_q']

    # get prompt slices:
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

    dominance_score_dict = {}
    for src_slc_name in src_slc_names:
        
        ## skip empty slices:
        if slcs[src_slc_name].stop - slcs[src_slc_name].start <= 0:
            print(f"WARNING: Skipping empty slice {src_slc_name} with slice {slcs[src_slc_name]}")
            continue

        ## set to the current slices:
        assert hs_dict[dominance_metric].ndim == 4, f"Expects a 4D tensor of shape (n_layer, head, dst, src), got {hs_dict[dominance_metric].shape} for metric {dominance_metric}"

        dst_slc = slcs[dst_slc_name] if hs_dict[dominance_metric].shape[-2] > 1 else slice(None, None)  # if dst is not a single token, use the full slice
        src_slc = slcs[src_slc_name] if hs_dict[dominance_metric].shape[-1] > 1 else slice(None, None)  # if src is not a single token, use the full slice

        dominance_score_dict[src_slc_name] = (
            hs_dict[dominance_metric]  # layer, head, dst, src
                [:, :, dst_slc, src_slc]
        )  # n_layer, head, chosen_dst, chosen_src

        if dominance_metric_flavor == 'sum':
            dominance_metric_flavor_q = 1.0  # generalize to 'sum-top_q' flavor
        assert 0 < dominance_metric_flavor_q <= 1, "dominance_metric_flavor_param should be in (0, 1)"
        
        # whether to skip layer dim on flattening layers
        dim_to_start_reduce = 1 if not aggr_all_layers else 0   
        flatten_vecs = dominance_score_dict[src_slc_name].flatten(start_dim=dim_to_start_reduce)  # layer, head*c_dst*c_src OR layer*head*c_dst*c_src

        ## pick the top-q% values and sum them
        k = max(1, int(dominance_metric_flavor_q * flatten_vecs.shape[-1]))
        dominance_score_dict[src_slc_name] = flatten_vecs.topk(
            k=k, dim=-1, largest=True, sorted=False
            ).values.sum(dim=-1)  # n_layer, head*c_dst*c_src -> n_layer OR n_layer*head*c_dst*c_src->scalar
        dominance_score_dict[src_slc_name] = dominance_score_dict[src_slc_name].tolist() 

        if aggr_all_layers:  # make sure we return a singleton list
            dominance_score_dict[src_slc_name] = [dominance_score_dict[src_slc_name]]

    return dominance_score_dict
