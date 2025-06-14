
def heatmap_with_direction(
        model: HookedTransformer,
        toks: Union[List[int], str],
        direction: torch.Tensor = None,  # (hidden_size,)
        on_hook: str = 'resid',
        measure_name: str = 'cos',
        slcs=None,
        crop_y_axis: slice = None,

        # cached `get_model_hidden_states`:
        hidden_states=None,
        labels=None,

        # optional `get_model_hidden_states` of another variant:
        alt_hidden_states=None,
        alt_toks=None,
        alt_slcs=None,

        title="",
        return_mesaures=False,

        **fig_kwargs
) -> Tuple[go.Figure, torch.Tensor]: 
    """
    Measures cosine similarity of `model` activations (`on_hook`) with the given `direction`, for the input tokens `toks`.
    
    Available `on_hook` options: ['resid', 'attn', 'mlp', 'decomp_resid']; defaults to the final residuals ('resid').
    
    Returns: a tuple of 
    (i) a heatmap (of `go.Figure`) showing the similarities;
    (ii) a tensor with these similarities (n_hooks, seq_len).
    """
    # 1. get hidden states:
    if hidden_states is None or labels is None:
        hidden_states, labels, _ = get_model_hidden_states(model, toks, return_labels=True, slcs=slcs, calc_special_matrices=True)

    # Extract required hidden states:
    def _extract_hidden_states(hs, labels, hook, slcs):
        if hook.startswith('Y_'):
            hook, slc_name, idx = hook.split('_')
            hs = hs[hook](slcs[f'slc_{slc_name}'].start + int(idx))
        if isinstance(labels, dict): labels = labels[hook]
        if isinstance(hs, dict): hs = hs[hook]
        return hs, labels

    hidden_states, labels = _extract_hidden_states(hidden_states, labels, on_hook, slcs)

    if alt_hidden_states is not None:
        alt_hidden_states, _ = _extract_hidden_states(alt_hidden_states, labels, on_hook, alt_slcs)


    # 1'. prepare:
    if isinstance(toks, str):
        toks, _ = to_toks(toks, model)

    # 2. Compute cosine similarity with activation:
    if measure_name == 'cos':
        measures = torch.nn.functional.cosine_similarity(
            hidden_states,
            direction.to(hidden_states.device),
            dim=-1
        )
    elif measure_name == 'dot':
        measures = torch.einsum('lsh,h->ls', hidden_states.float(), direction.to(hidden_states.device).float())  # TODO validate
    elif measure_name == 'norm':
        measures = torch.norm(hidden_states, p=2, dim=-1)
    elif measure_name == 'norm(hs)/norm(alt_hs)':  
        title="Norm(resids) / Norm(resids alt)"
        measures = torch.norm(hidden_states, p=2, dim=-1) / torch.norm(alt_hidden_states, p=2, dim=-1)
    elif measure_name == 'cos(hs,alt_hs)':
        title = f"Cos(HS, AltHS) of HS={on_hook}"
        measures = torch.nn.functional.cosine_similarity(
            hidden_states,
            alt_hidden_states.to(hidden_states.device),
            dim=-1
        )
    elif measure_name == 'cos(hs,dir)-cos(alt_hs,dir)':
        title = f"Cos(HS, Dir) - Cos(AltHS, Dir) of HS={on_hook} with given direction"
        measures = torch.nn.functional.cosine_similarity(
            hidden_states,
            direction.to(hidden_states.device),
            dim=-1
        ) - torch.nn.functional.cosine_similarity(
            alt_hidden_states,
            direction.to(hidden_states.device),
            dim=-1
        )
    elif measure_name == 'cos(hs-alt_hs,dir)':
        title = f"Cos(HS-AltHS, Dir) of HS={on_hook} with given direction"
        measures = torch.nn.functional.cosine_similarity(
            hidden_states - alt_hidden_states,
            direction.to(hidden_states.device),
            dim=-1
        )
    elif measure_name == 'dot_alt_diff':
        print("shape:", hidden_states.shape, direction.shape)
        measures = torch.einsum('lsh,h->ls', hidden_states, direction.to(hidden_states.device)) - torch.einsum('lsh,h->ls', alt_hidden_states, direction)

    if crop_y_axis is not None:
        measures = measures[crop_y_axis]
        labels = labels[crop_y_axis]

    # 3. Plot in heatmap, then show the plot
    fig = px.imshow(measures.float().cpu().numpy(), title=title, **fig_kwargs)
    
    hover_data = []
    toks_str = model.to_str_tokens(torch.tensor(toks))
    if alt_toks is not None:
        alt_toks = model.to_str_tokens(torch.tensor(alt_toks))
    for layer in range(len(labels)):
        if alt_toks:
            hover_data.append([f"{measure_name.title()}: {measures[layer, i] :.3f} <br> {labels[layer]} <br>Token-{i}: {tok} <br>Alt-Token-{i}: {alt_tok}" for i, (tok, alt_tok) in enumerate(zip(toks_str, alt_toks))])
        else:   
            hover_data.append([f"{measure_name.title()}: {measures[layer, i] :.3f} <br> {labels[layer]} <br>Token-{i}: {tok}" for i, tok in enumerate(toks_str)])
    fig.update_traces(hovertemplate='%{customdata}', customdata=hover_data)
    
    if slcs is not None:  # add vertical lines for slices
        for slc_name, slc in slcs.items():
            if slc_name not in ["slc_instr", "slc_adv", "slc_chat", "slc_affirm"]:
                continue
            fig.add_shape(
               type="line", x0=slc.start-0.5, y0=0, x1=slc.start-0.5, y1=len(labels), line=dict(color="red", width=2)
            )

    fig.update_xaxes(tickvals=list(range(len(toks_str))), ticktext=toks_str, tickangle=50)
    fig.update_yaxes(title_text="Layer", tickvals=list(range(len(labels))), ticktext=labels)
    fig.update_layout(width=800)
    
    if return_mesaures:
        return measures, labels
    else:
        fig.show()


def plot_conrib_bar(
    model: HookedTransformer,
    hs_dict: dict,
    slcs: Dict[str, slice],
    toks: List[int],
    up_to_layer=18,
    dir_to_inspect=None,  # defaults to resid[L18]; `d_model`
    pos_to_inspect=None,  # default to chat[-1]

    measure = 'dot',  # 'dot', 'cos', 'norm'
    standardize_measure=False,
    show_plot=True,
    normalize_dir=False,
    aggr_over_slcs=False,

    ablation_variant: str = "",  # attn_scores_only
):
    # hs_dict['decompose_resid__embed'] + hs_dict['decompose_resid__mlps'].sum(dim=0) + hs_dict['Y_all_norm_T'].sum(dim=(0,1,2)).cuda()) - 

    pos_to_inspect = pos_to_inspect if pos_to_inspect is not None else slcs['slc_chat'].stop - 1
    dir_to_inspect = dir_to_inspect if dir_to_inspect is not None else hs_dict['resid'][up_to_layer-1, pos_to_inspect]
    slcs['slc_self'] = slice(pos_to_inspect, pos_to_inspect + 1)  # add self-slice
    
    if normalize_dir:
        dir_to_inspect = dir_to_inspect / torch.norm(dir_to_inspect, p=2, dim=-1, keepdim=True)

    n_heads = model.cfg.n_heads
    str_toks =  model.to_str_tokens(torch.tensor(toks))

    def _get_slc_name(slcs, ix):
        if ix == pos_to_inspect:
            return 'slc_self'
        for k, v in slcs.items():
            l, u = v.start, v.stop
            if u is None: u = len(str_toks)
            if l <= ix < u:
                return k
        return None
    
    def _measure_func(_hs, _dir, layer=None):
        if measure == 'dot':
            return (_hs @ _dir.to(_hs)).item()
        elif measure == 'cos':
            return torch.nn.functional.cosine_similarity(_hs, _dir.to(_hs), dim=-1).item()
        elif measure == 'norm':
            return torch.norm(_hs, p=2, dim=-1).item()
        elif measure.startswith('dot_per_layer__'):
            _dir_name = measure.split('__')[1]  # e.g., 'resid'
            # _dir is ignored, we extract the layer's resid instead:
            _dir = hs_dict[_dir_name][layer, pos_to_inspect]
            # then, dot product as usual:
            return (_hs @ _dir.to(_hs)).item() / _dir.norm(p=2, dim=-1).pow(2).item()

        else:
            raise ValueError(f"Unknown measure: {measure}")
    
    _df = []

    ## embed:
    _embed = hs_dict['decompose_resid__embed'][pos_to_inspect].float()  # d_model
    _df.append(dict(
        label="embed",
        comp_type="embed",
        measure=_measure_func(_embed, dir_to_inspect, layer=0),
        src_slc_name=_get_slc_name(slcs, pos_to_inspect),  # this is a self-component
        str_tok=str_toks[pos_to_inspect],
        layer=0, head=0,
        ))

    ## MLPs:
    _mlps = hs_dict['decompose_resid__mlps'][:, pos_to_inspect]  # n_layers, d_model
    for layer in range(up_to_layer):
        _df.append(dict(
            label=f"mlp[L{layer}]",
            comp_type="mlp",
            measure=_measure_func(_mlps[layer], dir_to_inspect, layer=layer),
            src_slc_name=_get_slc_name(slcs, pos_to_inspect),  # this is a self-component
            str_tok=str_toks[pos_to_inspect],
            layer=layer, head=0,
        ))

    ## attns:
    _attns = hs_dict['Y_all_norm_T'][..., pos_to_inspect, :]  # layer, head, seq_len[src], [1 - seq_len[dst]], d_model

    if ablation_variant == 'attn_scores_only':
        print(hs_dict['attn_pattern'].shape, pos_to_inspect)
        _attns = hs_dict['attn_pattern'][:, :, pos_to_inspect]  # layer, head, seq_len[src]
        def _measure_func(_hs, _dir):
            return _hs.item()

    for layer in range(up_to_layer):
        for head in range(n_heads):
            sum_per_slcs = {slc_name: 0.0 for slc_name in ['slc_bos', 'slc_chat_pre', 'slc_instr', 'slc_adv', 'slc_chat',  'slc_affirm', 'slc_bad', 'slc_self']}
            for src_pos in range(_attns.shape[2]):
                src_slc_name = _get_slc_name(slcs, src_pos)
                _measure_val = _measure_func(_attns[layer, head, src_pos], dir_to_inspect, layer=layer)

                # [OPTION I] - fine
                if not aggr_over_slcs:
                    _df.append(dict(
                        label=f"attn[L{layer},H{head},P{src_pos}]",
                        comp_type="attn",
                        measure=_measure_val,
                        src_slc_name=src_slc_name,  # this is a self-component
                        str_tok=str_toks[src_pos],
                        Y_idx=(layer, head, src_pos),
                        layer=layer,
                        head=head,
                        relative_src_pos=src_pos - slcs[src_slc_name].start if src_slc_name is not None else None,
                    ))

                # [OPTION II] - coarse
                sum_per_slcs[src_slc_name] += _measure_val

            if aggr_over_slcs:  # [OPTION II] - coarse
                for slc_name, _measure_val in sum_per_slcs.items():
                    _df.append(dict(
                        label=f"attn[L{layer},H{head},{slc_name}]",
                        comp_type="attn",
                        measure=_measure_val,
                        src_slc_name=slc_name,
                        Y_idx=(layer, head, slcs[slc_name].start),
                        layer=layer,
                        head=head,
                    ))

    import plotly.express as px
    _df = pd.DataFrame(_df).sort_values('measure', ascending=False)
    _df['orig_measure'] = _df['measure']
    if standardize_measure:
        _df['measure'] = _df['measure'] / _df['measure'].sum().item()
    _df = _df
    fig = px.bar(_df, 
                y='measure', 
                x='src_slc_name',  
                color='comp_type',
                barmode='relative',
                hover_data=['label', 'str_tok'],
                color_discrete_map={ 'attn': 'red', 'mlp': 'blue', 'embed': 'green'},
                category_orders={'src_slc_name': ['slc_bos', 'slc_chat_pre', 'slc_instr', 'slc_adv', 'slc_chat',  'slc_affirm', 'slc_bad', 'slc_self']},
                )
    fig.update_traces(marker_line_color='black', marker_line_width=0.25, opacity=0.8)
    # fig.update_yaxes(range=[None, 1])
    fig.update_layout(width=800, height=600)

    if show_plot:
        fig.show()

    # print(f"sanity check (should be close):{ _df['orig_measure'].sum().item()} == {_measure_func(dir_to_inspect, dir_to_inspect)}")

    return _df


def get_top_attn_scores(
        model, cache, slcs, 
        top_k=None, attn_slice_pooling='sum',
        dst_slc=None,  # ignore
        src_slc=None,  # slice of the source token to consider
        topk_sorted_arg=True,
        tensor_instead_of_attn=None,
        limit_to_layers=None,
        hs_dict=None,
        score_based_on='attn_pattern',
        dir_dim_for_the_score=None,  # layer, d_model; for the dot product
    ):  # in adv->chat+i

    if dst_slc is None:
        dst_slc = slice(slcs['slc_chat'].start, slcs['slc_chat'].stop + 5)
    if src_slc is None:
        src_slc = slcs['slc_adv']

    if isinstance(dst_slc, int):
        dst_slc = slice(dst_slc, dst_slc + 1)

    if score_based_on == 'attn_pattern':
        scores = torch.stack([cache[f'blocks.{layer}.attn.hook_pattern'] for layer in range(model.cfg.n_layers)], dim=1).squeeze(0)  # layer, head, dst, src
        
        ## adv->chat
        scores = scores[:, :, dst_slc, src_slc]
        
    elif score_based_on == 'Y@attn':
        resid = hs_dict['attn'] # layer, dst, d_model
        resid = resid.unsqueeze(-2).unsqueeze(-2).transpose(1, 2)  # layer, head[1], dst, src[1], d_model
        Y = hs_dict['Y_all_norm']
        # Y.shape, resid.shape  # -> layer, head, dst, src, d_model

        # perform dot product of the last dimension
        dot_prod_vals = torch.einsum('lhtsd,lktkd->lhts', Y, resid)  # layer, head, dst, src
        
        # normalize per layer and (dst) token position
        scores = dot_prod_vals / torch.norm(resid, dim=-1).pow(2)  # layer, head, dst, src

        ## adv->chat
        scores = scores[:, :, dst_slc, src_slc]

    elif score_based_on == 'Y@dir':  # assumes `chat`!
        dst_slc_len = dst_slc.stop - dst_slc.start
        assert dst_slc_len == 5, f"dst_slc_len should be 5, but got {dst_slc_len} instead." 
        resid = dir_dim_for_the_score[:, :dst_slc_len]  # layer, dst_slc_len, d_model
        resid = resid.unsqueeze(-2).unsqueeze(-2).transpose(1, 2)  # layer, head[1], dst, src[1], d_model
        resid = resid.to(hs_dict['Y_all_norm'])  # layer, head[1], dst, src[1], d_model
        Y = hs_dict['Y_all_norm'][:, :, dst_slc, :]
        # Y.shape, resid.shape  # -> layer, head, dst_slc_len, src, d_model

        # perform dot product of the last dimension
        dot_prod_vals = torch.einsum('lhtsd,lktkd->lhts', Y, resid)  # layer, head, dst, src
        
        # normalize per layer and (dst) token position
        scores = dot_prod_vals / torch.norm(resid, dim=-1).pow(2)  # layer, head, dst, src
        scores = scores[:, :, :, src_slc]

    if limit_to_layers is not None:
        scores = scores[limit_to_layers]
    if tensor_instead_of_attn is not None:
        scores = tensor_instead_of_attn

    if top_k is None:
        top_k = scores.numel()  # simply consider all heads
    elif 0.0 < top_k <= 1.0:
        top_k = int(scores.numel() * top_k)

    if attn_slice_pooling == 'sum':
        scores = scores.sum(dim=(-1,-2))
    elif attn_slice_pooling == 'mean':
        scores = scores.mean(dim=(-1,-2))
    elif attn_slice_pooling == 'max':
        scores = scores.amax(dim=(-1,-2))
    elif attn_slice_pooling == 'top5_sum':
        scores = scores.flatten(-2,-1).topk(5, dim=-1).values.sum(dim=-1)
    elif attn_slice_pooling == 'sum__keep_dst':
        scores = scores.sum(dim=-2)
    elif attn_slice_pooling == 'no_pooling':
        pass
    else:
        raise NotImplementedError(f"Unknown attn_slice_pooling {attn_slice_pooling=}")
    
    ## plot bars with the top 20 heads, with their name LayerHHead; use plotly express  
    if attn_slice_pooling == 'no_pooling':
        top_indices = torch.topk(scores.flatten(), k=top_k, sorted=topk_sorted_arg).indices
        top_heads = []
        for idx in top_indices:
            layer, head, dst_pos, src_pos = (
                idx // (scores.shape[1] * scores.shape[2] * scores.shape[3]),
                (idx // (scores.shape[2] * scores.shape[3])) % scores.shape[1],
                (idx // scores.shape[3]) % scores.shape[2],
                idx % scores.shape[3],
            )
            # assert scores.flatten()[idx] == scores[layer, head, dst_pos, src_pos], f"Mismatch at index {idx}"
            top_heads.append((layer.item(), head.item(), dst_pos.item(), src_pos.item()))
    elif attn_slice_pooling == 'sum__keep_dst':
        top_indices = torch.topk(scores.flatten(start_dim=0), k=top_k).indices
        top_heads = []
        for idx in top_indices:
            layer, head, dst_pos = (
            idx // (scores.shape[1] * scores.shape[2]),
            (idx // scores.shape[2]) % scores.shape[1],
            idx % scores.shape[2],
            )
            top_heads.append((layer.item(), head.item(), dst_pos.item()))
    else:
        top_heads = [(idx // scores.shape[1], idx % scores.shape[1]) 
            for idx in torch.topk(scores.flatten(), k=top_k).indices]

    # TODO make a list
    return top_heads, scores  # List, torch.Tensor["layer, head"]