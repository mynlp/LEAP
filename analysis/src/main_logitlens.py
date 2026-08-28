"""Logit-lens analysis of a LEAP meta-trained LoRA adapter: edit a
`meta_adapter` and a fresh `random_adapter` per case with the same
inner-loop, then compare how much each one's final-token hidden state
reveals the intermediate/final entity when projected through the model's own
unembedding (logit lens), layer by layer.

Writes one JSONL row per case (correctness + per-layer entity logprob, per
adapter) for `analyze_logitlens_final_token_delta.py` to plot.
"""

import argparse
import json
from datetime import datetime
import os
from typing import List

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftConfig, PeftModel, TaskType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.analysis_utils import (
    KnowledgeEditor,
    _get_input_device,
    _iter_trace_roots,
    get_adapter_weights_only,
    get_hidden_states_prompt_only,
    normalize_answer_text,
    resolve_trace_layers,
    set_adapter_weights,
    set_trainable_adapter,
)


def get_intermediate_entity_from_item(item):
    rw = item["requested_rewrite"][0]
    tnew = rw.get("target_new", "")
    if isinstance(tnew, dict) and "str" in tnew:
        return tnew["str"]
    return str(tnew)


def check_answer_in_generation(generated_text: str, answers: List[str]):
    pred = generated_text.lower()
    return any(answer.lower() in pred for answer in answers)


@torch.no_grad()
def generate_from_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
):
    was_training = model.training
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    if was_training:
        model.train()
    return generated_text.strip()


def get_answer_candidates(edit_item):
    cands = list(edit_item["new_answer_alias"])
    cands.append(edit_item["new_answer"])
    # Deduplicate
    seen = set()
    out = []
    for x in cands:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _get_module_device(module, fallback_device):
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return fallback_device


def _get_lm_head_module(model):
    candidates = [
        ("lm_head",),
        ("embed_out",),
        ("model", "lm_head"),
        ("base_model", "lm_head"),
        ("base_model", "model", "lm_head"),
    ]

    for root in _iter_trace_roots(model):
        for chain in candidates:
            obj = root
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, torch.nn.Module):
                return obj

    for root in _iter_trace_roots(model):
        for name, module in root.named_modules():
            if name.endswith("lm_head") or name.endswith("embed_out"):
                return module

    raise ValueError("lm_head / embed_out not found.")


def _get_final_norm_module(model):
    candidates = [
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
        ("model", "decoder", "final_layer_norm"),
        ("decoder", "final_layer_norm"),
        ("transformer", "norm"),
        ("model", "final_layernorm"),
        ("model", "final_layer_norm"),
    ]

    for root in _iter_trace_roots(model):
        for chain in candidates:
            obj = root
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, torch.nn.Module):
                return obj

    return None


@torch.no_grad()
def project_hidden_with_logit_lens(
    model,
    hidden: torch.Tensor,
):
    lm_head = _get_lm_head_module(model)
    final_norm = _get_final_norm_module(model)

    lm_head_device = _get_module_device(
        lm_head, fallback_device=_get_input_device(model)
    )

    try:
        lm_head_dtype = next(lm_head.parameters()).dtype
    except StopIteration:
        lm_head_dtype = hidden.dtype

    lens_hidden = hidden
    if final_norm is not None:
        norm_device = _get_module_device(final_norm, fallback_device=lm_head_device)
        lens_hidden = lens_hidden.to(device=norm_device, dtype=lm_head_dtype)
        lens_hidden = final_norm(lens_hidden)
    else:
        lens_hidden = lens_hidden.to(device=lm_head_device, dtype=lm_head_dtype)

    if lens_hidden.device != lm_head_device:
        lens_hidden = lens_hidden.to(lm_head_device)

    logits = lm_head(lens_hidden)
    return logits


@torch.no_grad()
def logit_lens_final_token_entity_logprob(
    model,
    tokenizer,
    *,
    source_prompt: str,
    entity_text: str,
):
    model.eval()

    source_states, source_ids_cpu = get_hidden_states_prompt_only(
        model, tokenizer, source_prompt
    )
    _, layers = resolve_trace_layers(model)

    final_idx = int(source_ids_cpu.shape[1]) - 1

    entity_text_norm = normalize_answer_text(entity_text)
    entity_ids = tokenizer(
        entity_text_norm, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    if entity_ids.numel() == 0:
        raise ValueError("entity_text tokenized to an empty sequence.")

    target_token_id = int(entity_ids[0].item())
    target_token = tokenizer.convert_ids_to_tokens([target_token_id])[0]

    logprob_by_layer = []
    for lname in layers:
        hidden = source_states[lname][:, final_idx : final_idx + 1, :]  # [B,1,H]
        logits = (
            project_hidden_with_logit_lens(model, hidden)[0, 0]
            .detach()
            .to("cpu")
            .float()
        )  # [V]
        logprob_by_layer.append(F.log_softmax(logits, dim=-1)[target_token_id].item())

    return {
        "entity_first_token": target_token,
        "logprob_by_layer": logprob_by_layer,
    }


def main():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--base_model_path", required=True, type=str)
    parser.add_argument("--meta_trained_path", required=True, type=str)
    parser.add_argument("--datatype", required=True, type=str)
    parser.add_argument("--metrics_save_dir", default="./output_QA", type=str)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=None, type=int)
    parser.add_argument(
        "--entity_type", default="intermediate", choices=["intermediate", "final"]
    )
    parser.add_argument(
        "--generation_max_new_tokens",
        default=10,
        type=int,
        help="Max new tokens to generate from source_prompt when checking correctness",
    )
    args = parser.parse_args()

    os.makedirs(args.metrics_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.metrics_save_dir = os.path.join(
        args.metrics_save_dir, f"logitlens_{timestamp}"
    )
    os.makedirs(args.metrics_save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(
        base_model, args.meta_trained_path, adapter_name="meta_adapter"
    )

    meta_cfg = PeftConfig.from_pretrained(args.meta_trained_path)
    random_cfg = LoraConfig(
        r=meta_cfg.r,
        lora_alpha=meta_cfg.lora_alpha,
        target_modules=meta_cfg.target_modules,
        lora_dropout=meta_cfg.lora_dropout,
        bias=getattr(meta_cfg, "bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    model.add_adapter(
        adapter_name="random_adapter",
        peft_config=random_cfg,
    )

    model.set_adapter("meta_adapter")
    meta_weights = get_adapter_weights_only(model, "meta_adapter")
    model.set_adapter("random_adapter")
    base_weights = get_adapter_weights_only(model, "random_adapter")

    with open(f"./datasets/{args.datatype}-cake.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data = data[args.start_idx : args.end_idx]

    metrics_path = os.path.join(
        args.metrics_save_dir, f"{args.datatype}_logitlens_token_identity.jsonl"
    )
    generation_summary_path = os.path.join(
        args.metrics_save_dir, f"{args.datatype}_generation_accuracy_summary.json"
    )

    edit_config = {
        "num_epochs": 10,
        "lr": 1e-4,
        "weight_decay": 0.0,
    }
    editor = KnowledgeEditor(edit_config)
    generation_counts = {
        "base": {"correct": 0, "total": 0},
        "meta": {"correct": 0, "total": 0},
    }

    with open(metrics_path, "w", encoding="utf-8") as fw:
        for item in tqdm(data):
            case_id = item.get("case_id")
            source_prompt = item["cloze_question"]

            if args.entity_type == "intermediate":
                entity = get_intermediate_entity_from_item(item)
            else:
                entity = item.get("new_answer", "")
            answer_candidates = get_answer_candidates(item)

            results = {}

            for tag, adapter_name, weights in [
                ("base", "random_adapter", base_weights),
                ("meta", "meta_adapter", meta_weights),
            ]:
                model.set_adapter(adapter_name)
                set_adapter_weights(model, weights)
                set_trainable_adapter(model, adapter_name)

                edited_model = editor.edit(
                    model,
                    tokenizer,
                    [
                        {
                            "prompt": rewrite["prompt"].format(rewrite["subject"]),
                            "target_new": rewrite["target_new"]["str"],
                        }
                        for rewrite in item["requested_rewrite"]
                    ],
                )

                generated_text = generate_from_prompt(
                    edited_model,
                    tokenizer,
                    source_prompt,
                    max_new_tokens=args.generation_max_new_tokens,
                )
                is_correct = check_answer_in_generation(
                    generated_text,
                    answer_candidates,
                )
                generation_counts[tag]["total"] += 1
                generation_counts[tag]["correct"] += int(is_correct)

                ll = logit_lens_final_token_entity_logprob(
                    edited_model,
                    tokenizer,
                    source_prompt=source_prompt,
                    entity_text=entity,
                )

                results[tag] = {
                    "correct": is_correct,
                    "generated_text": generated_text,
                    "entity_first_token": ll["entity_first_token"],
                    "logprob_by_layer": ll["logprob_by_layer"],
                }

            fw.write(
                json.dumps(
                    {
                        "case_id": case_id,
                        "entity": entity,
                        "base": results["base"],
                        "meta": results["meta"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fw.flush()

            model.set_adapter("meta_adapter")
            set_adapter_weights(model, meta_weights)
            set_trainable_adapter(model, "meta_adapter")

    generation_summary = {}
    for tag, counts in generation_counts.items():
        total = counts["total"]
        generation_summary[tag] = {
            **counts,
            "accuracy": counts["correct"] / total if total else None,
        }
    with open(generation_summary_path, "w", encoding="utf-8") as fh:
        json.dump(generation_summary, fh, ensure_ascii=False, indent=2)

    print("Saved:", metrics_path)
    print("Generation summary:", generation_summary_path)


if __name__ == "__main__":
    main()
