import random
from typing import List
import pandas as pd
import torch
from src.evaluate.utils import categorize_suffixes
from transformer_lens import HookedTransformer
from src.refusal_direction.pipeline.model_utils.model_factory import construct_model_base
import ast

def prepare_df_for_specific_analysis(
        df: pd.DataFrame, 
        n_samples: int,
        random_seed: int = 2,
        include_groups: List[str] = ['reg_best'],
    ) -> pd.DataFrame:  

    df = categorize_suffixes(df)

    # get suffix stats:
    avg_score_per_suffix = df[df.suffix_cat == 'reg'].groupby("suffix_id")["strongreject_finetuned"].mean()
    sorted_suffixes = avg_score_per_suffix.sort_values(ascending=False).index.tolist()
    avg_score_per_suffix_mult = df[df.suffix_cat == 'reg_mult'].groupby("suffix_id")["strongreject_finetuned"].mean()
    sorted_suffixes_mult = avg_score_per_suffix_mult.sort_values(ascending=False).index.tolist()

    # choose suffixes:
    chosen_suffixes = []

    if 'reg_best' in include_groups:
        chosen_suffixes += [
            ### Regular, top 3:
            ("best1", sorted_suffixes[0]),
            ("best2", sorted_suffixes[1]),
            ("best3", sorted_suffixes[2]),
        ]
    
    if 'reg_best2' in include_groups:
        chosen_suffixes += [
            ### Regular, from top 10:
            ("best4", sorted_suffixes[3]),
            ("best6", sorted_suffixes[5]),
            ("best8", sorted_suffixes[7]),
        ]
    
    if 'reg_best3' in include_groups:
        chosen_suffixes += [
            ### Regular, from top 10 (the remaining):
            ("best5", sorted_suffixes[4]),
            ("best7", sorted_suffixes[6]),
            ("best9", sorted_suffixes[8]),

        ]

    if 'reg_mid' in include_groups:
        chosen_suffixes += [
            ("middle1", sorted_suffixes[int(len(sorted_suffixes) * 0.2)]),
            ("middle2", sorted_suffixes[int(len(sorted_suffixes) * 0.25)]),
            ("middle3", sorted_suffixes[int(len(sorted_suffixes) * 0.3)]),
    ]

    if 'reg_mid2' in include_groups:
        chosen_suffixes += [
            ("middle4", sorted_suffixes[int(len(sorted_suffixes) * 0.45)]),
            ("middle5", sorted_suffixes[int(len(sorted_suffixes) * 0.5)]),
            ("middle6", sorted_suffixes[int(len(sorted_suffixes) * 0.55)]),
    ]
        
    if 'reg_mid3' in include_groups:
        chosen_suffixes += [
            ("middle7", sorted_suffixes[int(len(sorted_suffixes) * 0.52)]),
            ("middle8", sorted_suffixes[int(len(sorted_suffixes) * 0.57)]),
            ("middle9", sorted_suffixes[int(len(sorted_suffixes) * 0.6)]),
        ]

    if 'reg_worst' in include_groups:
        chosen_suffixes += [
            ("worst3", sorted_suffixes[int(len(sorted_suffixes) * 0.75)]),
            ("worst2", sorted_suffixes[int(len(sorted_suffixes) * 0.8)]),
            ("worst1", sorted_suffixes[int(len(sorted_suffixes) * 0.85)]),
    ]

    if 'reg-mult_best' in include_groups:
        chosen_suffixes += [
            ### Mult:
            ("mult_best1", sorted_suffixes_mult[0]),
            ("mult_best2", sorted_suffixes_mult[1]),
            ("mult_best3", sorted_suffixes_mult[2]),
            # ("mult_worst1", sorted_suffixes_mult[-1]),
        ]

    if 'reg-mult_mid' in include_groups:
        chosen_suffixes += [
            ("mult_mid1", sorted_suffixes_mult[int(len(sorted_suffixes_mult) * 0.2)]),
            ("mult_mid2", sorted_suffixes_mult[int(len(sorted_suffixes_mult) * 0.35)]),
            ("mult_mid3", sorted_suffixes_mult[int(len(sorted_suffixes_mult) * 0.5)]),
        ]

    # discard duplicates, if any exists:
    new_chosen_suffixes = []
    for suffix in chosen_suffixes:
        if suffix[1] not in [s[1] for s in new_chosen_suffixes]:
            new_chosen_suffixes.append(suffix)
    chosen_suffixes = new_chosen_suffixes

    # if exists, add init suffix:
    chosen_suffixes += [("init", df[df.suffix_cat == 'init'].suffix_id.iloc[0])]

    chosen_suffix_labels, chosen_suffix_ids = zip(*chosen_suffixes)
    
    # choose message ids:
    n_chosen_m_ids =  n_samples // len(chosen_suffixes)
    random.seed(random_seed)

    # filter-out messages with successful prefill+init [TODO] enable
    non_trivial_mids = df[(df.prefilled__strongreject_finetuned < 0.25) & (df.suffix_cat == 'init')].message_id.unique()
    df = df[df.message_id.isin(non_trivial_mids)]

    # [OPTION 1] choose 10 random message ids (-> len(chosen_suffixes) * k samples)
    # chosen_message_ids = random.sample(df.message_id.unique().tolist(), k=n_chosen_m_ids)
    
    # [OPTION 2] choose messages ids that are successful in most suffixes:
    chosen_message_ids_all = df[df.suffix_id.isin(chosen_suffix_ids)].groupby("message_id")["strongreject_finetuned"].mean().sort_values(ascending=False).index.tolist()
    chosen_message_ids = []
    chosen_message_ids += random.sample(chosen_message_ids_all[:len(chosen_message_ids_all) // 7], k=3 * n_chosen_m_ids // 7)
    chosen_message_ids += random.sample(chosen_message_ids_all[len(chosen_message_ids_all) // 7: len(chosen_message_ids_all) // 3], k=3 * n_chosen_m_ids // 7)
    chosen_message_ids += random.sample(chosen_message_ids_all[len(chosen_message_ids_all) // 3:], k=n_chosen_m_ids // 7)

    df = df[(df.suffix_id.isin(chosen_suffix_ids)) & (df.message_id.isin(chosen_message_ids))]
    df['suffix_id'] = df.suffix_id.apply(lambda x: chosen_suffix_labels[chosen_suffix_ids.index(x)])
    print(f"Chose {len(df)} samples for analysis.")

    return df


def load_model(model_name):
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


def load_data(
        model_name="google/gemma-2-2b-it",

        # filters:
        filter_to_non_trivial=False,
        msg_slices=None,
        suffix_optimizers=['gcg'],
        suffix_objectives=['affirm'],
        suffix_cats=None # ['reg', 'init'],
):
    ## load data:
    if model_name == "google/gemma-2-2b-it":
        file_path = "artifacts/gcg_data/v2/suffixed_msgs_w_resp_w_eval_prfl__gemma_gcg_affirm__raw_df.bz2"
    elif model_name == "qwen/qwen2.5-1.5b-instruct":
        file_path = "artifacts/gcg_data/v2_qwen2.5/suffixed_msgs_w_resp_w_eval_prfl__qwen2.5-1.5b-instruct_gcg_affirm.bz2"
    elif model_name == "meta-llama/llama-3.1-8b-instruct":
        file_path = "artifacts/gcg_data/v2_llama3.1/suffixed_msgs_w_resp_w_eval_prfl__llama3.1_gcg_affirm.bz2"
    else:
        raise ValueError(f"Model {model_name} not supported.")

    df = pd.read_pickle(file_path, compression='bz2')

    ##### CODE TO RUN TO TRANSFORM CSV TO BZ2: ########
    # from src.evaluate.utils import enrich_with_categorization, categorize_suffixes
    # df['suffix_obj_indices'] = df.suffix_obj_indices.apply(ast.literal_eval)
    # df['suffix_id'] = df.suffix_id.apply(str)
    # df['is_mult_attack'] = df.suffix_obj_indices.apply(lambda x: len(x) > 1)
    # df['response'] = df.response.apply(str)
    # df['message_suffixed'] = df.message_suffixed.apply(str)
    # df = enrich_with_categorization(df, model_base)
    # df = categorize_suffixes(df)
    # df.to_pickle(file_path, compression='bz2')

    # filter to non-mult messages:
    df = df[~df.is_mult_attack]
    df = df[df.suffix_optimizer.isin(suffix_optimizers)]
    df = df[df.suffix_objective.isin(suffix_objectives)]
    if suffix_cats is not None:
        df = df[df.suffix_cat.isin(suffix_cats)]

    # no training messages:  # TODO!

    if filter_to_non_trivial:
        # filter to messages that fail under `init+prefill` (non-trivial ones)
        non_trivial_mids = df[(df.prefilled__strongreject_finetuned < 0.25) & (df.suffix_cat == 'init')].message_id.unique()
        df = df[df.message_id.isin(non_trivial_mids)]

    ## calc relevant stats:
    top_suffix_ids = df.groupby('suffix_id').strongreject_finetuned.mean().sort_values(ascending=False).index.tolist()
    suffix_ids_to_avg = df.groupby('suffix_id').strongreject_finetuned.mean().to_dict()

    ## add suffix stats as columns:
    df['suffix_rank'] = df.suffix_id.apply(lambda x: top_suffix_ids.index(x))
    df['univ_score'] = df.suffix_id.apply(lambda x: suffix_ids_to_avg[x])
    
    ## fetch target affirms:
    targets = pd.read_csv("data/harmful_behaviors.csv").target_response_prefix.tolist()

    ## calc the set of hard messages:
    univ_per_msg = df[df.suffix_id.isin(top_suffix_ids[:1])].groupby('message_id').agg({'strongreject_finetuned': 'max'}).sort_values('strongreject_finetuned', ascending=False)
    # filter to the message indices with the highest scores
    univ_per_msg = univ_per_msg[univ_per_msg.strongreject_finetuned > 0.75].index.tolist()
    non_univ_per_msg = df[df.suffix_id.isin(top_suffix_ids[len(top_suffix_ids) // 10:])].groupby('message_id').agg({'strongreject_finetuned': 'max'}).sort_values('strongreject_finetuned', ascending=False)
    non_univ_per_msg = non_univ_per_msg[non_univ_per_msg.strongreject_finetuned > 0.75].index.tolist()

    ## what messages are in the first one and not the second one?
    hard_message_ids = [m for m in univ_per_msg if m not in non_univ_per_msg]
    random.shuffle(hard_message_ids)
    
    df['is_hard_message'] = df.message_id.apply(lambda x: x in hard_message_ids)
    
    ## filter to the message slices, if given:
    if msg_slices is not None:
        chosen_message_ids = df.message_id.unique()
        random.seed(42)
        random.shuffle(chosen_message_ids)
        chosen_message_ids = chosen_message_ids[msg_slices]
        df = df[df.message_id.isin(chosen_message_ids)]

    return df, top_suffix_ids, suffix_ids_to_avg, targets, hard_message_ids
