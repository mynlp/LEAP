"""Utilities shared by src/main_logitlens.py and
src/main_activation_patch_lora.py: locating a PEFT-wrapped model's decoder
layers/prompt hidden states (used by both the logit-lens and
activation-patching entry points), swapping between the meta/random adapter's
weights, and the LoRA inner-loop editor both of them use to simulate
test-time editing.

get_hidden_states_prompt_only adapts CaKE's (https://github.com/zjunlp/CaKE,
MIT License, Copyright (c) 2025 ZJUNLP) Analysis/patched_generation.py
get_hidden_states.
"""

import logging
from typing import Any, Dict, List

import torch
from baukit import TraceDict


def _get_input_device(model):
    return next(model.parameters()).device


def get_adapter_weights_only(model, adapter_name: str):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if adapter_name in name
    }


def set_adapter_weights(model, weights_dict):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weights_dict:
                param.copy_(weights_dict[name].to(param.device))


def set_trainable_adapter(model, adapter_name: str):
    for name, param in model.named_parameters():
        param.requires_grad = adapter_name in name


def normalize_answer_text(ans: str):
    if len(ans) == 0:
        return ans
    if ans[0].isspace():
        return ans
    return " " + ans


def _iter_trace_roots(model):
    seen = set()
    roots = []

    def add(obj):
        if obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        roots.append(obj)

    add(model)

    if hasattr(model, "get_base_model"):
        try:
            add(model.get_base_model())
        except Exception:
            pass

    for chain in (
        ("model",),
        ("base_model",),
        ("model", "model"),
        ("base_model", "model"),
    ):
        obj = model
        for attr in chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        add(obj)

    return roots


def _get_num_hidden_layers(model):
    for root in _iter_trace_roots(model):
        config = getattr(root, "config", None)
        if config is not None and hasattr(config, "num_hidden_layers"):
            return config.num_hidden_layers
    raise ValueError("No config with num_hidden_layers was found.")


def _find_layer_prefix(trace_root):
    matches = [
        name for name, _ in trace_root.named_modules() if name.endswith("layers.0")
    ]
    if not matches:
        return None
    # If multiple candidates share the same root, prefer the shortest prefix.
    return min(matches, key=len)[: -len(".0")]


def resolve_trace_layers(model):
    """
    Return the root module to pass to Trace/TraceDict, along with the layer
    names valid for that root.
    """
    n_layers = _get_num_hidden_layers(model)

    for trace_root in _iter_trace_roots(model):
        layer_prefix = _find_layer_prefix(trace_root)
        if layer_prefix is None:
            continue
        layer_names = [f"{layer_prefix}.{i}" for i in range(n_layers)]
        return trace_root, layer_names

    raise ValueError(
        "No prefix corresponding to `layers.0` was found. Check named_modules()."
    )


def _extract_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    return output


@torch.no_grad()
def get_hidden_states_prompt_only(model, tokenizer, prompt: str):
    model.eval()
    device = _get_input_device(model)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    attention_mask = torch.ones_like(prompt_ids, device=device)

    trace_root, layer_names = resolve_trace_layers(model)
    with TraceDict(trace_root, layer_names) as trace:
        _ = model(input_ids=prompt_ids, attention_mask=attention_mask)

    hs = {}
    for name in layer_names:
        hs[name] = _extract_hidden(trace[name].output).detach().to("cpu")
    return hs, prompt_ids.detach().to("cpu")


class KnowledgeEditor:
    def __init__(self, config):
        self.inner_loop_config = config
        self._inner_opt = None
        self._inner_opt_param_ids = None

    def edit(
        self,
        model,
        tokenizer,
        edit_requests: List[Dict[str, Any]],
    ):
        if not edit_requests:
            logging.warning("No edit requests provided. Returning original model.")
            return model
        model.train()

        optimizer = self._ensure_inner_opt(model)
        device = model.device

        texts = [r["prompt"] for r in edit_requests]
        targets = [r["target_new"] for r in edit_requests]

        full_prompts = [f"{p} {t}" for p, t in zip(texts, targets)]
        tokens = tokenizer(
            full_prompts,
            return_tensors="pt",
            padding="longest",
        )
        tokens["labels"] = tokens["input_ids"].clone()

        prompts = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
        )
        prompt_ids = prompts["input_ids"]
        num_prompt_toks = [int((p != tokenizer.pad_token_id).sum()) for p in prompt_ids]
        num_pad_tokens = [
            int((t == tokenizer.pad_token_id).sum()) for t in tokens["labels"]
        ]
        for i in range(len(texts)):
            tokens["labels"][i][
                num_pad_tokens[i] : num_pad_tokens[i] + num_prompt_toks[i]
            ] = -100
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = -100
        tokens = tokens.to(device)

        for _ in range(self.inner_loop_config["num_epochs"]):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**tokens)
            loss: torch.Tensor = outputs.loss
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        return model

    def _ensure_inner_opt(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        param_ids = tuple(id(p) for p in params)
        if self._inner_opt is None or self._inner_opt_param_ids != param_ids:
            self._inner_opt = torch.optim.AdamW(
                params,
                lr=self.inner_loop_config["lr"],
                weight_decay=self.inner_loop_config["weight_decay"],
            )
            self._inner_opt_param_ids = param_ids
        else:
            self._inner_opt.state.clear()
        return self._inner_opt
