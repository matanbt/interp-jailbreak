import itertools
import random
import numpy as np
from src.interp.utils import load_model
from src.evaluate.utils import load_data


model_name = "google/gemma-2-2b-it"
# model_name = "qwen/qwen2.5-1.5b-instruct"
# model_name = "meta-llama/llama-3-8b-instruct"


data_df = load_data(model_name=model_name)
model = load_model(model_name)

random.seed(42)
message_ids = random.sample(data_df.message_id.unique().tolist(), 10)
suffix_ids = []
# intervals = [0, 0.005, 0.01, 0.025] + np.arange(0.05, data_df.univ_score.max(), 0.05).tolist()
# for i in range(len(intervals) - 1):
#     interval = (intervals[i], intervals[i + 1])
#     suffixes_in_interval =  data_df[data_df.univ_score.between(*interval)].suffix_id.unique().tolist()
#     suffix_ids.extend(random.sample(
#        suffixes_in_interval, min(20, len(suffixes_in_interval))  # TODO number should be configurable
#     ))

suffix_ids = data_df.suffix_id.unique().tolist()

print(f"{len(message_ids)=}, {len(suffix_ids)=}")

from tqdm.contrib.itertools import product
import src.interp.dominance_tools as dominance_tools

df = []

for message_id, suffix_id in product(message_ids, suffix_ids):
    row = data_df[(data_df.message_id == message_id) & (data_df.suffix_id == suffix_id)].iloc[0]
    message_str = row.message_str
    suffix_str, suffix_univ, suffix_id = row.suffix_str, row.univ_score, row.suffix_id
    response_score, response_category = row.strongreject_finetuned, row.category

    dom_scores = dominance_tools.get_dom_scores(
        model,
        message_str, suffix_str,
        hs_dict=TODO,
        dst_slc_name = 'chat[-1]',
        hijacking_metric ='Y@attn',  # TODO iterate on configurable list (incl. attetnion and Y@dir)
        hijacking_metric_flavor = 'sum',  # TODO iterate on multiple (also consider variance? attn tracker inspired?)
    )
    for src, layer in itertools.product(['instr', 'adv'], range(model.cfg.n_layers)):
        df.append({
            'message_id': message_id,
            'suffix_id': suffix_id,
            # 'message_str': message_str,
            # 'suffix_str': suffix_str,
            'suffix_univ': suffix_univ,
            'response_score': response_score,
            'response_category': response_category,

            # specific:
            'src': src,
            'layer': layer,
            'dom_score': dom_scores[src][layer].item()
        })
import pandas as pd
df = pd.DataFrame(df)

import os
if not os.path.exists("results/"):
    os.makedirs("results/")
df.to_csv(f"results/grid_hijacking_{model_name.replace('/', '_')}-all.csv", index=False)