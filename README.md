# LEAP

**Learning Edit-Adaptable Parameters for Multi-Hop Knowledge Propagation**

LEAP is a MAML-based meta-training framework that learns edit-adaptable LoRA parameters so that ordinary single-hop knowledge edits also propagate to their multi-hop consequences, without changing the test-time editing procedure itself.

## Structure

- [`meta-training/`](meta-training) — meta-trains the edit-adaptable LoRA adapter (`meta_adapter`).
- [`evaluation/`](evaluation) — evaluates a `meta_adapter` on MQuAKE multi-hop edit propagation.
- [`analysis/`](analysis) — interpretability analysis (logit lens, activation patching) of how LEAP changes edit propagation internally.

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). Parts of `evaluation/` and `analysis/` are adapted from [CaKE](https://github.com/zjunlp/CaKE) (MIT License, Copyright (c) 2025 ZJUNLP) — see `THIRD_PARTY_LICENSES/` and each subdirectory's README for details.
