# Universal Jailbreak Suffixes Are Strong Attention Hijackers 🥷
The official repository for the paper [“Universal Jailbreak Suffixes Are Strong Attention Hijackers”](https://arxiv.org/abs/2506.12880), analyzing the underlying mechanism of suffix-based LLM jailbreaks, and exploiting it to enhance and mitigate existing jailbreaks.

<div align="center">
<img width="650" src="docs/fig1.png">
</div>


## Setup
The project requires Python `3.1.0` and the installation of `requirements.txt`, preferably in an isolated environment, as it includes a [fork of TransformerLens](https://github.com/matanbt/TransformerLens) (othewise, this could override existing installation of TransformerLens).

## Usage

Common entry points are demonstrated in the [demo notebook](./demo.ipynb). There, we reproduce the initial GCG universality analysis (§3), calculation of the *dominance score* (Eq. 3, §5) over different prompts, which can be aggregated per GCG suffix, leading to their *hijacking strength* (§6).

## Additional Usage
Coming soon...

<!-- ## Usage (Causal Experiments; §2)
Coming soon... 

## Usage (Hijacking Strength vs Universality)
Coming soon...


## Usage (Enhance GCG)
Coming soon...

## Usage (Mitigate GCG)
Coming soon... -->

## Additional Research Artifacts
Some of our additional research artifacts (which are used in this code) may be useful for further/other types of analyses.
- **The GCG suffixes dataset** for further exploring GCG suffixes.
    - GCG suffixes crafted on Gemma-2, Qwen-2.5 and Llama-3.1, their generated response when appended to harmful instructions (from AdvBench, StrongReject's custom), their evaluation and charecterization.
    - In HuggingFace: https://huggingface.co/datasets/MatanBT/gcg-evaluated-data
- **TransformerLens fork** for further exploring the dominance score.
    - Adds to TransformerLens support of fine-grained attention hooks to enable the calculation of dominance score.
    - GitHub Repo: https://github.com/matanbt/TransformerLens


## Acknowledgements
Some code snippets are loosely inspired by the following codebases:
- [GraySwanAI/nanoGCG](https://github.com/GraySwanAI/nanoGCG/tree/main])
- [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)

## Cite
If you find this work, or any of the research artifacts, useful, please cite our paper as:
```
@article{bentov2025universaljailbreaksuffixesstrong,
      title={{U}niversal {J}ailbreak {S}uffixes {A}re {S}trong {A}ttention {H}ijackers}, 
      author={Matan Ben-Tov and Mor Geva and Mahmood Sharif},
      year={2025},
      eprint={2506.12880},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.12880}, 
}
```

