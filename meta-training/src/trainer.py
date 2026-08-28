import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Union
from transformers import Trainer
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from src.editor import KnowledgeEditor
import logging


class MetaTrainer(Trainer):
    def __init__(
        self,
        editor: KnowledgeEditor,
        tokenizer,
        simple_ft: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.editor = editor
        self._tokenizer = tokenizer
        self.simple_ft = simple_ft

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):
        unwrapped_model = self.accelerator.unwrap_model(model)

        if not self.simple_ft:
            original = copy.deepcopy(
                get_peft_model_state_dict(unwrapped_model, adapter_name="meta_adapter")
            )

            # 2) Inner loop: temporarily adapt on the support set (no gradient flows to the outer loop)
            with torch.enable_grad():
                self.editor.edit(
                    model,
                    self._tokenizer,
                    [
                        {"prompt": input["prompt"], "target_new": input["target_new"]}
                        for input in inputs
                    ],
                    self.accelerator,
                )

        all_prompt_answers = [
            prompt_r["prompt"] + " " + prompt_r["ground_truth"][0]
            if isinstance(prompt_r["ground_truth"], list)
            else prompt_r["prompt"] + " " + prompt_r["ground_truth"]
            for input in inputs
            for prompt_r in input["portability_r"]
        ]
        all_prompts = [
            prompt_r["prompt"]
            for input in inputs
            for prompt_r in input["portability_r"]
        ]

        if self.simple_ft:
            all_prompt_answers = [
                input["prompt"] + " " + input["target_new"] for input in inputs
            ] + all_prompt_answers
            all_prompts = [input["prompt"] for input in inputs] + all_prompts

        all_toks = self._tokenizer(
            all_prompt_answers,
            return_tensors="pt",
            padding="longest",
        ).to(self.accelerator.device)
        all_prompt_toks = self._tokenizer(
            all_prompts,
            return_tensors="pt",
            padding="longest",
        )
        tok_labels = all_toks["input_ids"].clone()
        num_prompt_toks = [
            int((p != self._tokenizer.pad_token_id).sum())
            for p in all_prompt_toks["input_ids"]
        ]
        num_pad_tokens = [
            int((t == self._tokenizer.pad_token_id).sum()) for t in tok_labels
        ]
        for i in range(len(all_prompts)):
            tok_labels[i][
                num_pad_tokens[i] : num_pad_tokens[i] + num_prompt_toks[i]
            ] = -100
        tok_labels[tok_labels == self._tokenizer.pad_token_id] = -100
        tok_labels = tok_labels.to(self.accelerator.device)

        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        all_toks["labels"] = tok_labels
        with self.compute_loss_context_manager():
            outputs = model(
                input_ids=all_toks["input_ids"],
                attention_mask=all_toks["attention_mask"],
                labels=all_toks["labels"],
                use_cache=False,
            )
            loss = outputs.loss
            self.accelerator.backward(loss)
            logging.info(f"Portability Reasoning Loss: {loss.item()}")

        if self.simple_ft:
            return loss.detach()

        set_peft_model_state_dict(
            unwrapped_model, original, adapter_name="meta_adapter"
        )

        return loss.detach()
