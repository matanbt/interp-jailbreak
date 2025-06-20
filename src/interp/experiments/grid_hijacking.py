import itertools
import random
from typing import Tuple
import numpy as np
from scipy.stats import spearmanr
from src.interp.utils import load_model
from src.evaluate.utils import load_data
from tqdm.contrib.itertools import product as tproduct
from src.interp.dominance_tools import get_dominance_scores, get_model_hidden_states
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import typer

REPRODUCIBLE_SEED = 42


def grid_hijacking(
    model_name: str="google/gemma-2-2b-it",
    n_messages: int = 30,
    n_suffixes: int = 400,
    suffix_interval_diff: float = 0.05,
    inspected_dom_scores: Tuple[str] = ('Y@attn', 'A', 'Y@resid', 'Y@dcmp_resid', '(X@W_VO)@attn', 'norm(X)', 'norm(Y)'),
    dst_slc_name: str = 'chat[-1]',
    src_slc_names: Tuple[str] = ('instr', 'adv', 'chat', 'input'),
):
    # Load model & data
    model = load_model(model_name)
    data_df = load_data(model_name=model_name)

    # Samples messages and suffixes
    random.seed(REPRODUCIBLE_SEED)
    
    message_ids = random.sample(
        data_df.message_id.unique().tolist(), 
        min(n_messages, len(data_df.message_id.unique()))
        )
    
    # sample suiffixes from intervals of univ_score
    suffix_ids = []
    intervals = (
        [0, 0.005, 0.01] # include highly non-universal suffixes
        + np.arange(suffix_interval_diff, data_df.univ_score.max(), suffix_interval_diff).tolist()
    )
    for i in range(len(intervals) - 1):
        interval = (intervals[i], intervals[i + 1])
        suffixes_in_interval =  data_df[data_df.univ_score.between(*interval)].suffix_id.unique().tolist()
        suffix_ids.extend(random.sample(
        suffixes_in_interval, min(n_suffixes // len(intervals), len(suffixes_in_interval))
        ))

    print(f"#Messages: {len(message_ids)}, Message IDs: {message_ids}")
    print(f"Suffix IDs: {suffix_ids}")
    print(f"#Suffixes: {len(suffix_ids)} with {n_suffixes // len(intervals)} per interval.")
    print(f"Suffix scores: {sorted(data_df[data_df.suffix_id.isin(suffix_ids)].univ_score.unique().tolist())}")

    df = []

    for message_id, suffix_id in tproduct(message_ids, suffix_ids):
        row = data_df[(data_df.message_id == message_id) & (data_df.suffix_id == suffix_id)].iloc[0]
        message_str = row.message_str
        suffix_str, suffix_univ, suffix_id, suffix_category = row.suffix_str, row.univ_score, row.suffix_id, row.suffix_category
        response_score, response_category = row.strongreject_finetuned, row.response_category

        ## Get hidden states and dominance scores' tensors:
        hs_dict, _ = get_model_hidden_states(
            model,
            message_str + suffix_str,
            add_dominance_calc=True,
            selected_dom_scores=inspected_dom_scores,
        )

        # extract the dominance scores per measure:
        for dom_score, dom_score_flavor, aggr_all_layers in itertools.product(
            inspected_dom_scores, 
            ['sum', 'sum-top_q'], 
            [True, False]
            ):
            dom_scores = get_dominance_scores(
                model,
                message_str, suffix_str,
                hs_dict=hs_dict,
                dst_slc_name=dst_slc_name,
                src_slc_names=src_slc_names,
                dominance_metric=dom_score,
                dominance_metric_flavor=dom_score_flavor,
                dominance_metric_flavor_q=0.1,  # TODO make configurable
                aggr_all_layers=aggr_all_layers,
            )
            n_layers = model.cfg.n_layers if not aggr_all_layers else 1
            for src, layer in itertools.product(dom_scores.keys(), range(n_layers)):
                df.append({
                    # prompt:
                    'message_id': message_id,
                    'suffix_id': suffix_id,
                    'suffix_category': suffix_category,
                    'suffix_univ': suffix_univ,
                    'response_score': response_score,
                    'response_category': response_category,

                    # dom score definition:
                    'dom_score_name': dom_score,
                    'dom_score_flavor_name': dom_score_flavor,

                    # location:
                    'src': src,
                    'layer': layer if not aggr_all_layers else -1,  # -1 means aggregated all layers

                    # values:
                    'dom_score': dom_scores[src][layer]
                })

    df = pd.DataFrame(df)


    if not os.path.exists("results/"):
        os.makedirs("results/")
    csv_path = f"results/grid_hijacking[{dst_slc_name}]_{model_name.replace('/', '_')}-n=[{n_messages}, {n_suffixes}].csv"
    df.to_csv(f"results/grid_hijacking[{dst_slc_name}]_{model_name.replace('/', '_')}-n=[{n_messages}, {n_suffixes}].csv", index=False)

    make_box_plot(csv_path, save_fig=True)


def make_box_plot(
    csv_path: str,
    layer: int = 20,  # -1 for all layers
    slc_src_name: str = 'adv',
    dom_score_name: str = 'Y@attn',
    dom_score_flavor_name: str = 'sum',
    sample_filter: str = None,
    save_fig: bool = False,
):
    """
    Make a box plot of hijacking strength vs universality score.
    Args:
        csv_path (str): Path to the CSV file containing the data.
        layer (int): Layer number to filter by. Use -1 for all layers.
        slc_src_name (str): Source name to filter by.
        dom_score_name (str): Dominance score name to filter by.
        dom_score_flavor_name (str): Dominance score flavor name to filter by.
        sample_filter (str): Filter for response categories; default is None (= no filter).
        save_fig (bool): Whether to save the figure as a PDF.
    """
    df = pd.read_csv(csv_path)
    df = df[
        (df.layer == layer) &
        (df.src == slc_src_name) &
        (df.dom_score_name == dom_score_name) &
        (df.dom_score_flavor_name == dom_score_flavor_name) &
        ((df.response_category == 'fail[cannot]') if sample_filter == 'fail[cannot]' else True)
    ]
    assert not df.empty, f"No data found for layer={layer}, src={slc_src_name}, dom_score_name={dom_score_name}, dom_score_flavor_name={dom_score_flavor_name}"

    ## Aggregate dom scores to hijacking strength:
    max_univ_score = df.suffix_univ.max()
    interval_diff = 0.10
    bins = [0, 0.01, 0.05] + np.arange(0.10, max_univ_score + interval_diff, interval_diff).tolist()
    labels = [f"[{bins[i]:.2f}, {bins[i+1]:.2f}]" for i in range(len(bins)-1)]
    agg_df = df.groupby('suffix_id').agg({
        'suffix_univ': 'first',
        'dom_score': 'mean'
    }).reset_index()
    agg_df["univ_bin"] = pd.cut(agg_df["suffix_univ"], bins=bins, labels=labels, include_lowest=True)
    agg_df.sort_values(by="univ_bin", inplace=True)

    # print correlation:
    spearman_corr, _ = spearmanr(agg_df['suffix_univ'], agg_df['dom_score'])
    print(f"Corr: {spearman_corr:.3f} between suffix_univ and dom_score")

    ## Compute means per bin to get the trendline:
    trend_df = agg_df.groupby("univ_bin", observed=True)["dom_score"].median().reset_index()

    # Make boxplot:
    fig = px.box(
        agg_df, x="univ_bin", y="dom_score", points="outliers",
        labels={
            "univ_bin": "Universality Score",
            "dom_score": "Hijacking Strength"
        },
        width=500, height=350,
        color_discrete_sequence=["cornflowerblue"]
    )

    # Add trendline over medians
    fig.add_trace(go.Scatter(
        x=trend_df["univ_bin"],
        y=trend_df["dom_score"],
        mode="lines+markers",
        line=dict(color="crimson", width=3),
        marker=dict(size=6),
        name="Median Trend"
    ))

    # Layout tweaks
    fig.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
    )

    # save to pdf:
    if save_fig:
        fig.write_image(
            f"results/{os.path.basename(csv_path).replace('.csv', '')}_boxplot__"
            f"layer={layer}_src={slc_src_name}_dom={dom_score_name}_{dom_score_flavor_name}.pdf",
        )

    return fig

## setup for CLI:
app = typer.Typer()

@app.command()
def grid_hijacking_cli(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="Model name to load."),
):
    print(f"Running hijacking analysis for model: {model_name}")

    grid_hijacking(
        model_name=model_name,
    )

if __name__ == "__main__":
    app()