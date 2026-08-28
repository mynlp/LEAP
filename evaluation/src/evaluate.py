from time import time
from typing import Any, Dict, List

import torch

from src.metrics import compute_edit_quality, compute_locality_quality


def _get_first_cuda_device(model):
    try:
        return next(p.device for p in model.parameters() if p.device.type == "cuda")
    except StopIteration:
        return torch.device("cuda:0")


class KnowledgeEditor:
    def __init__(self, config: Dict[str, Any]):
        self.inner_loop_config = config
        self._inner_opt = None

    def edit(self, model, tokenizer, edit_requests: List[Dict[str, Any]]):
        if not edit_requests:
            return model
        model.train()

        optimizer = self._ensure_inner_opt(model)
        device = _get_first_cuda_device(model)

        texts = [r["prompt"] for r in edit_requests]
        targets = [r["target_new"] for r in edit_requests]

        full_prompts = [f"{p} {t}" for p, t in zip(texts, targets)]
        tokens = tokenizer(full_prompts, return_tensors="pt", padding="longest")
        tokens["labels"] = tokens["input_ids"].clone()
        prompts = tokenizer(texts, return_tensors="pt", padding="longest")
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
        if self._inner_opt is None:
            params = [p for p in model.parameters() if p.requires_grad]
            self._inner_opt = torch.optim.AdamW(
                params,
                lr=self.inner_loop_config["lr"],
                weight_decay=self.inner_loop_config["weight_decay"],
            )
        else:
            self._inner_opt.state.clear()
        return self._inner_opt


def edit_meta_lora(model, tokenizer, edit_item, config, locality_probes=None):
    start = time()
    requests = edit_item["requested_rewrite"]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    editor = KnowledgeEditor(config=config["inner_loop"])
    with torch.enable_grad():
        editor.edit(
            model,
            tokenizer,
            [
                {
                    "prompt": rewrite["prompt"].format(rewrite["subject"]),
                    "target_new": rewrite["target_new"]["str"],
                }
                for rewrite in requests
            ],
        )
    exec_time = time() - start

    device = _get_first_cuda_device(model)
    # This metrics dict's shape (case_id/requested_rewrite/time/post) follows
    # the convention of CaKE's edit_utils.py cake()/edit() functions.
    metrics = {
        "case_id": edit_item["case_id"],
        "requested_rewrite": requests,
        "time": exec_time,
        "post": compute_edit_quality(model, tokenizer, edit_item, device),
    }
    if locality_probes is not None:
        metrics["post"]["locality"] = compute_locality_quality(
            model, tokenizer, locality_probes, device
        )
    tokenizer.padding_side = original_padding_side
    return model, metrics
