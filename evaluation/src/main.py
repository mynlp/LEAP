"""Evaluate a LEAP meta-trained LoRA adapter (meta_adapter) on MQuAKE
multi-hop editing: for each case, edit the adapter in place with a short
inner-loop (see src/evaluate.edit_meta_lora), score it, then restore the
adapter's original weights before moving to the next case.

This driver's overall shape (load model, loop over cases, dump
`{tag}_metrics.json`/`{tag}_res.json`) follows CaKE's (https://github.com/zjunlp/CaKE,
MIT License, Copyright (c) 2025 ZJUNLP) `test_cake.py` `__main__` driver.
"""

import argparse
import copy
import json
import os

import torch
import yaml
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluate import edit_meta_lora
from src.metrics import calculate_averages, calculate_locality_average


def set_trainable_adapter(model, adapter_name: str):
    for name, param in model.named_parameters():
        param.requires_grad = adapter_name in name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llama3-8b.yaml", type=str)
    parser.add_argument(
        "--data_path",
        default="datasets/mquake/MQuAKE-CF-3k-v2-cake.json",
        type=str,
    )
    parser.add_argument("--locality_probes_path", default=None, type=str)
    parser.add_argument(
        "--meta_lora_path",
        default=None,
        type=str,
        help="Path to a meta-training meta_adapter checkpoint; if omitted, a fresh untrained adapter is used",
    )
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=None, type=int)
    parser.add_argument("--metrics_save_dir", default="./output", type=str)
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["model_name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    if args.meta_lora_path is not None:
        model = PeftModel.from_pretrained(
            model, args.meta_lora_path, adapter_name="meta_adapter"
        )
    else:
        lora_config = config["lora"]
        peft_config = LoraConfig(
            r=lora_config["rank"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config, adapter_name="meta_adapter")
    model.set_adapter("meta_adapter")
    original_adapter_state = copy.deepcopy(
        get_peft_model_state_dict(model, adapter_name="meta_adapter")
    )

    with open(args.data_path, "r") as f:
        full_data = json.load(f)
    data = full_data[args.start_idx : args.end_idx]

    locality_probes_by_case_id = None
    if args.locality_probes_path is not None:
        with open(args.locality_probes_path, "r") as f:
            probe_case_ids_by_case_id = json.load(f)
        item_by_case_id = {item["case_id"]: item for item in full_data}
        locality_probes_by_case_id = {
            int(case_id): [
                item_by_case_id[pid] for pid in probe_case_ids if pid in item_by_case_id
            ]
            for case_id, probe_case_ids in probe_case_ids_by_case_id.items()
        }

    set_trainable_adapter(model, "meta_adapter")
    all_metrics = []
    for item in data:
        locality_probes = (
            locality_probes_by_case_id.get(item["case_id"], [])
            if locality_probes_by_case_id is not None
            else None
        )
        model, metrics = edit_meta_lora(
            model, tokenizer, item, config, locality_probes=locality_probes
        )
        set_peft_model_state_dict(
            model, original_adapter_state, adapter_name="meta_adapter"
        )
        print(json.dumps(metrics, indent=4))
        all_metrics.append(metrics)

    res = calculate_averages(all_metrics)
    if locality_probes_by_case_id is not None:
        locality_res = calculate_locality_average(all_metrics)
        if locality_res is not None:
            res["locality"] = locality_res

    os.makedirs(args.metrics_save_dir, exist_ok=True)
    tag = f"meta_LoRA_{args.start_idx}_{args.end_idx}"
    with open(f"{args.metrics_save_dir}/{tag}_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    with open(f"{args.metrics_save_dir}/{tag}_res.json", "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()
