"""Activation-patching analysis of a LEAP meta-trained LoRA adapter: edit a
`meta_adapter` and a fresh `random_adapter` per case with the same inner-loop,
then patch one side's per-layer hidden states into the other — one (layer,
token position) at a time over the last `--last_n_tokens` prompt positions —
to see which layer/token carries the edit's causal effect
(--patch_direction selects base_to_meta or meta_to_base).

Writes one JSONL row per case (base/meta scores, delta_logprob by
[layer][distance-from-final-token]) for
`analyze_activation_patch_suffix_profile.py` to plot.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from baukit import Trace
from peft import LoraConfig, PeftConfig, PeftModel, TaskType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.analysis_utils import (
    KnowledgeEditor,
    _get_input_device,
    get_adapter_weights_only,
    get_hidden_states_prompt_only,
    normalize_answer_text,
    resolve_trace_layers,
    set_adapter_weights,
    set_trainable_adapter,
)


def extract_target_new_text(rewrite: Dict[str, Any]):
    target_new = rewrite.get("target_new", "")
    if isinstance(target_new, dict):
        return str(target_new.get("str", ""))
    return str(target_new)


def make_edit_requests(item: Dict[str, Any]):
    return [
        {
            "prompt": rewrite["prompt"].format(rewrite["subject"]),
            "target_new": extract_target_new_text(rewrite),
        }
        for rewrite in item["requested_rewrite"]
    ]


def _gold_logprob_sum_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int,
    ans_len: int,
):
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]

    start = prompt_len - 1
    end = start + ans_len

    target_ans = target[:, start:end]
    lp_ans = logprobs[:, start:end, :].gather(-1, target_ans.unsqueeze(-1)).squeeze(-1)
    return lp_ans.sum(dim=-1)


@torch.no_grad()
def gold_logprob_sum(model, tokenizer, prompt: str, answer_text: str):
    """
    Return the sum of log p(answer | prompt): the log-probability that
    answer_text follows prompt.
    """
    model.eval()
    device = _get_input_device(model)
    answer_text = normalize_answer_text(answer_text)

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ]
    ans_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ]

    input_ids = torch.cat([prompt_ids, ans_ids], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return _gold_logprob_sum_from_logits(
        out.logits, input_ids, prompt_ids.shape[1], ans_ids.shape[1]
    )


def _replace_one_token(output, token_idx: int, clean_h: torch.Tensor):
    """
    output: a layer module's output (Tensor or tuple)
    clean_h: [B, H] (CPU is fine; the device is aligned here)
    """
    if isinstance(output, tuple):
        h = output[0]
        rest = output[1:]
        h2 = h.clone()
        h2[:, token_idx, :] = clean_h.to(h2.device)
        return (h2, *rest)
    else:
        h = output
        h2 = h.clone()
        h2[:, token_idx, :] = clean_h.to(h2.device)
        return h2


@torch.no_grad()
def patched_gold_logprob_sum(
    model,
    tokenizer,
    prompt: str,
    answer_text: str,
    layer_name: str,
    token_idx: int,
    clean_states: torch.Tensor,
    trace_root=None,
):
    """
    Return the gold logprob sum with only layer_name's token_idx position
    replaced by the clean value.
    clean_states: [B, S, H] (e.g. the meta-edited side's layer output)
    """
    model.eval()
    device = _get_input_device(model)

    answer_text = normalize_answer_text(answer_text)

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ]
    ans_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = torch.cat([prompt_ids, ans_ids], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    clean_h_tok = clean_states[:, token_idx, :]  # [B, H] (CPU is fine)

    def hook_fn(out):
        return _replace_one_token(out, token_idx=token_idx, clean_h=clean_h_tok)

    if trace_root is None:
        trace_root, _ = resolve_trace_layers(model)
    with Trace(trace_root, layer=layer_name, edit_output=hook_fn):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        return _gold_logprob_sum_from_logits(
            out.logits, input_ids, prompt_ids.shape[1], ans_ids.shape[1]
        )


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
        "--last_n_tokens",
        default=16,
        type=int,
        help=(
            "Only patch the last N prompt token positions (the combined "
            "figure only ever shows this many positions from the end)."
        ),
    )
    parser.add_argument(
        "--patch_direction",
        default="base_to_meta",
        choices=["base_to_meta", "meta_to_base"],
    )
    parser.add_argument("--edit_num_epochs", default=10, type=int)
    parser.add_argument("--edit_lr", default=1e-4, type=float)
    parser.add_argument("--edit_weight_decay", default=0.0, type=float)
    args = parser.parse_args()

    os.makedirs(args.metrics_save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    meta_adapter_name = "meta_adapter"
    base_adapter_name = "random_adapter"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(
        base_model, args.meta_trained_path, adapter_name=meta_adapter_name
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
        adapter_name=base_adapter_name,
        peft_config=random_cfg,
    )

    model.set_adapter(meta_adapter_name)
    meta_weights = get_adapter_weights_only(model, meta_adapter_name)
    model.set_adapter(base_adapter_name)
    base_weights = get_adapter_weights_only(model, base_adapter_name)

    adapter_specs = {
        "base": {"adapter_name": base_adapter_name, "weights": base_weights},
        "meta": {"adapter_name": meta_adapter_name, "weights": meta_weights},
    }
    if args.patch_direction == "base_to_meta":
        source_tag, target_tag = "base", "meta"
    else:
        source_tag, target_tag = "meta", "base"
    source_spec = adapter_specs[source_tag]
    target_spec = adapter_specs[target_tag]

    with open(f"./datasets/{args.datatype}-cake.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data = data[args.start_idx : args.end_idx]

    metrics_path = os.path.join(
        args.metrics_save_dir,
        f"{args.datatype}_activation_patch_{args.patch_direction}.jsonl",
    )

    edit_config = {
        "num_epochs": args.edit_num_epochs,
        "lr": args.edit_lr,
        "weight_decay": args.edit_weight_decay,
    }
    editor = KnowledgeEditor(edit_config)

    with open(metrics_path, "w", encoding="utf-8") as fw:
        for item in tqdm(data):
            case_id = item.get("case_id")
            source_prompt = item["cloze_question"]
            answer_text = str(item.get("new_answer", ""))
            if not answer_text:
                raise ValueError(f"case_id={case_id}: new_answer is empty.")

            edit_requests = make_edit_requests(item)

            model.set_adapter(source_spec["adapter_name"])
            set_adapter_weights(model, source_spec["weights"])
            set_trainable_adapter(model, source_spec["adapter_name"])
            editor.edit(model, tokenizer, edit_requests)

            source_states, prompt_ids_cpu = get_hidden_states_prompt_only(
                model,
                tokenizer,
                source_prompt,
            )
            trace_root, layers = resolve_trace_layers(model)
            seq_len = int(prompt_ids_cpu.shape[1])
            token_positions = list(range(max(0, seq_len - args.last_n_tokens), seq_len))
            source_score = (
                gold_logprob_sum(model, tokenizer, source_prompt, answer_text)
                .mean()
                .item()
            )

            model.set_adapter(target_spec["adapter_name"])
            set_adapter_weights(model, target_spec["weights"])
            set_trainable_adapter(model, target_spec["adapter_name"])
            editor.edit(model, tokenizer, edit_requests)

            target_score = (
                gold_logprob_sum(model, tokenizer, source_prompt, answer_text)
                .mean()
                .item()
            )

            num_pos = len(token_positions)
            # delta_by_distance[layer][distance]: distance 0 = final prompt token
            delta_by_distance = [[0.0] * num_pos for _ in layers]
            for li, layer_name in enumerate(layers):
                source_layer_states = source_states[layer_name]
                for ti, token_idx in enumerate(token_positions):
                    patched_score = (
                        patched_gold_logprob_sum(
                            model,
                            tokenizer,
                            source_prompt,
                            answer_text,
                            layer_name=layer_name,
                            token_idx=token_idx,
                            clean_states=source_layer_states,
                            trace_root=trace_root,
                        )
                        .mean()
                        .item()
                    )
                    distance = num_pos - 1 - ti
                    delta_by_distance[li][distance] = patched_score - target_score

            base_score = source_score if source_tag == "base" else target_score
            meta_score = source_score if source_tag == "meta" else target_score

            fw.write(
                json.dumps(
                    {
                        "case_id": case_id,
                        "patch_direction": args.patch_direction,
                        "base_score": base_score,
                        "meta_score": meta_score,
                        "delta_logprob_by_distance": delta_by_distance,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fw.flush()

            model.set_adapter(target_spec["adapter_name"])
            set_adapter_weights(model, target_spec["weights"])
            set_trainable_adapter(model, target_spec["adapter_name"])

    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
