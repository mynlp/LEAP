import logging
from typing import List, Dict, Any
from accelerate import Accelerator

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase


class KnowledgeEditor:
    def __init__(self, config):
        self.inner_loop_config = config["inner_loop"]
        self._inner_opt = None

    def edit(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        edit_requests: List[Dict[str, Any]],
        accelerator: Accelerator,
    ):
        logging.info("Executing Inner Loop ...")

        # Check that the edit requests are not empty
        if not edit_requests:
            logging.warning("No edit requests provided. Returning original model.")
            return
        logging.info(f"  Edit Request: {edit_requests}")
        model.train()

        optimizer = self._ensure_inner_opt(model, accelerator)
        device = accelerator.device

        texts = [r["prompt"] for r in edit_requests]
        targets = [r["target_new"] for r in edit_requests]

        # --- Inner training loop ---
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

        for it in range(self.inner_loop_config["num_epochs"]):
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**tokens)
                loss: torch.Tensor = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return

    def _ensure_inner_opt(self, model, accelerator):
        if self._inner_opt is None:
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(
                params,
                lr=self.inner_loop_config["lr"],
                weight_decay=self.inner_loop_config["weight_decay"],
            )
            self._inner_opt = accelerator.prepare(opt)
        else:
            self._inner_opt.state.clear()
        return self._inner_opt
