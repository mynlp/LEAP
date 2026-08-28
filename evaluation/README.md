# LEAP evaluation

Evaluates a meta-trained LoRA adapter (`meta_adapter`, produced by [`../meta-training`](../meta-training)) on MQuAKE multi-hop editing: for each case, the adapter is edited in place with a short inner-loop, scored for edit accuracy / hop-wise accuracy / locality, then reset to its original weights before the next case.

## Layout

```
configs/llama3-8b.yaml   # model + LoRA + inner-loop hyperparameters
datasets/mquake/         # MQuAKE-CF-3k-v2 eval set + locality probes
src/
  main.py                # CLI entry point
  evaluate.py            # per-case edit-then-score loop
  metrics.py             # accuracy / locality scoring
```

## Usage

```bash
cd LEAP/evaluation
pip install -r requirements.txt
python -m src.main \
    --config configs/llama3-8b.yaml \
    --data_path datasets/mquake/MQuAKE-CF-3k-v2-cake.json \
    --locality_probes_path datasets/mquake/MQuAKE-CF-3k-v2-cake.locality_probes.json \
    --meta_lora_path ../meta-training/outputs/<TIMESTAMP>/meta_adapter \
    --metrics_save_dir ./output
```


## Provenance

Parts of `src/main.py`, `src/evaluate.py`, and `src/metrics.py` are adapted from [CaKE](https://github.com/zjunlp/CaKE) (MIT License, Copyright (c) 2025 ZJUNLP); see the notes at the top of each file for details.

The files under `datasets/mquake/` (`MQuAKE-CF-3k-v2-cake.json`, `MQuAKE-T-cake.json`) also originate from CaKE.
