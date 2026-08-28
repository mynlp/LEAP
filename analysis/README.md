# LEAP analysis

Interpretability tooling for a meta-trained LoRA adapter (`meta_adapter`, from [`../meta-training`](../meta-training)). Both entry points edit the adapter in place with a short inner-loop, compare it against a freshly-initialized `random_adapter` edited the same way, and write one compact JSONL row per case — no per-case images or tensor files.

- `src/main_logitlens.py` — after editing, projects each layer's final-token hidden state through the model's own unembedding (logit lens), recording the target entity's logprob per layer for both adapters, plus generation correctness.
- `src/main_activation_patch_lora.py` — after editing both adapters, patches one adapter's cached per-layer activations into the other, one (layer, token position) at a time over the last `--last_n_tokens` positions, to localize the edit's causal effect (`--patch_direction base_to_meta`/`meta_to_base`).

Two post-processing scripts turn a pair of those JSONLs into a combined figure:

- `src/analyze_logitlens_final_token_delta.py` — takes a `--bridge_input`/`--final_input` pair (`--entity_type intermediate`/`final` runs), buckets cases by LoRA/LEAP correctness, and plots the combined 3x2 (Base/Delta/Meta x Bridge/Final) figure with bootstrap 95% CIs.
- `src/analyze_activation_patch_suffix_profile.py` — takes a `--jsonl <a> <b>` pair (one `base_to_meta` run, one `meta_to_base` run), averages `delta_logprob_by_distance` per (layer, distance) cell, and plots the combined 2-panel figure.

## Layout

```
datasets/MQuAKE-CF-3k-v2-first-hop-cake.json  # subset of CaKE's MQuAKE-CF-3k-v2-cake.json:
                                              # 2-hop cases whose edit is restricted to hop 1
src/
  analysis_utils.py
  main_logitlens.py
  main_activation_patch_lora.py
  analyze_logitlens_final_token_delta.py     # post-processing
  analyze_activation_patch_suffix_profile.py # post-processing
```

## Usage

### Logit lens (3x2 Base/Delta/Meta x Bridge/Final figure)

Run twice — once per entity type:

```bash
cd LEAP/analysis
pip install -r requirements.txt

python -m src.main_logitlens \
    --datatype MQuAKE-CF-3k-v2-first-hop \
    --model_type llama3-8b \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --meta_trained_path ../meta-training/outputs/<TIMESTAMP>/meta_adapter \
    --metrics_save_dir ./output/logitlens \
    --entity_type intermediate   # bridge entity

python -m src.main_logitlens \
    --datatype MQuAKE-CF-3k-v2-first-hop \
    --model_type llama3-8b \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --meta_trained_path ../meta-training/outputs/<TIMESTAMP>/meta_adapter \
    --metrics_save_dir ./output/logitlens \
    --entity_type final          # final-answer entity
```

Each run writes a timestamped `logitlens_<TIMESTAMP>/` directory under `--metrics_save_dir`. Then build the combined figure:

```bash
python -m src.analyze_logitlens_final_token_delta \
    --bridge_input output/logitlens/logitlens_<TIMESTAMP_intermediate> \
    --final_input  output/logitlens/logitlens_<TIMESTAMP_final>
```

### Activation patching (2-panel LoRA-to-LEAP / LEAP-to-LoRA figure)

Run twice — once per `--patch_direction`:

```bash
python -m src.main_activation_patch_lora \
    --datatype MQuAKE-CF-3k-v2-first-hop \
    --model_type llama3-8b \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --meta_trained_path ../meta-training/outputs/<TIMESTAMP>/meta_adapter \
    --metrics_save_dir ./output/activation_patch \
    --patch_direction base_to_meta

python -m src.main_activation_patch_lora \
    --datatype MQuAKE-CF-3k-v2-first-hop \
    --model_type llama3-8b \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --meta_trained_path ../meta-training/outputs/<TIMESTAMP>/meta_adapter \
    --metrics_save_dir ./output/activation_patch \
    --patch_direction meta_to_base
```

Both invocations write directly into `--metrics_save_dir`, ready for:

```bash
python -m src.analyze_activation_patch_suffix_profile \
    --jsonl output/activation_patch/MQuAKE-CF-3k-v2-first-hop_activation_patch_base_to_meta.jsonl \
            output/activation_patch/MQuAKE-CF-3k-v2-first-hop_activation_patch_meta_to_base.jsonl
```

## Provenance

Parts of this pipeline are adapted from [CaKE](https://github.com/zjunlp/CaKE)'s `Analysis/patched_generation.py` (MIT License, Copyright (c) 2025 ZJUNLP): the TraceDict-based hidden-state capture, the Trace-based activation-patching hook, and the logit-lens projection. `datasets/MQuAKE-CF-3k-v2-first-hop-cake.json` is extracted from CaKE's `MQuAKE-CF-3k-v2-cake.json`.
