# interp-jailbreak
The official repository for the paper “Universal Jailbreak Suffixes Are Strong Attention Hijackers”.

<div align="center">
<img width="600" src="docs/fig1.png">
</div>


## Setup
The project requires Python `3.1.0` and on and the installation of `requirements.txt`, preferably in an isolated environment, as it includes a [fork of TransformerLens](..) (this could override existing installation of TransformerLens).

## Usage (Dominance and Hijacking Strength)

The [demo notebook](./demo.ipynb) demosntrate calculating the *dominance score* over different prompt, and comparing the *hijacking strength* with GCG suffixes' universality.

<!-- ## Usage (Evaluate Universality)
Coming soon...  -->

## Usage (Enhance GCG)
Coming soon...

## Usage (Mitigate GCG)
Coming soon...

## Additional Usage
- **TransformerLens fork** for further exploring the dominance score.
    - Adds to TransformerLens support of fine-graind attention hooks to enable the calculation of dominance score.
    - GitHub Repo: https://github.com/matanbt/TransformerLens
- **The GCG suffixes dataset** for further exploring GCG suffixes.
    - GCG suffixes crafted on Gemma-2, Qwen-2.5 and Llama-3.1, their generated response when appended to harmful instructions (from AdvBench, StrongReject's custom), their evaluation and charecterization.
    - Huggingface Dataset: https://huggingface.co/datasets/MatanBT/gcg-evaluated-data



## Acks:
- Arditi et al
- nanoGCG

