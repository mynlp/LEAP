# LEAP meta-training

MAML-based meta-training that produces edit-adaptable LoRA parameters (`meta_adapter`): the inner loop edits the adapter on single-hop facts, the outer loop optimizes the pre-edit adapter so that edit propagates to multi-hop queries. See [`../evaluation`](../evaluation) to evaluate the resulting adapter.

## Layout

```
configs/                      # model + LoRA + inner/outer-loop hyperparameters
datasets/meta-training.json   # synthetic multi-hop tasks (see Data below)
src/
  main.py     # CLI entry point
  dataset.py  # MetaTrainDataset
  editor.py   # inner-loop LoRA editor
  trainer.py  # MAML inner/outer-loop training step
  utils.py    # logging, seeding
```

## Usage

```bash
cd LEAP/meta-training
python -m src.main --config_path configs/llama3-8b.yaml
```

This writes a timestamped directory under the config's `output_dir` containing `config.yaml`, logs, and the trained `meta_adapter` LoRA weights.

## Data

`datasets/meta-training.json` is a synthetic dataset we constructed from [Wikidata](https://www.wikidata.org/) (CC0) relation chains, using fictional entities for the edited facts; its entities and relations are disjoint from MQuAKE. See the paper's Appendix A for the full construction procedure.
