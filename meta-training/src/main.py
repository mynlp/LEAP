import argparse
import yaml
import os
import logging
from datetime import datetime
import torch
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import get_peft_model, LoraConfig, TaskType

from src.dataset import (
    MetaTrainDataset,
)
from src.editor import KnowledgeEditor
from src.utils import setup_logging, set_seed
from src.trainer import MetaTrainer


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(config["output_dir"], timestamp)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    setup_logging(logs_dir)
    set_seed(config["seed"])

    model_config = config["model"]
    lora_config = config["lora"]

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name"],
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure the LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"],
        attn_implementation=model_config.get("attn_implementation", "eager"),
        device_map=model_config.get("device_map", "auto"),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=model_config.get("low_cpu_mem_usage", True),
    )
    lora_config = LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config, adapter_name="meta_adapter")
    model.set_adapter("meta_adapter")
    model.print_trainable_parameters()
    logging.info(f"Config: {config}")

    train_dataset = MetaTrainDataset(
        data_path=config["data"]["train_data_path"],
    )

    # Initialize the editor
    editor = KnowledgeEditor(config=config)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["outer_loop"].get("batch_size", 8),
        per_device_eval_batch_size=config["outer_loop"].get("batch_size", 8),
        num_train_epochs=config["outer_loop"].get("num_epochs", 1),
        learning_rate=config["outer_loop"].get("lr", 1e-5),
        weight_decay=config["outer_loop"].get("weight_decay", 0),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", False),
        remove_unused_columns=False,
        lr_scheduler_type="constant_with_warmup",
        ddp_find_unused_parameters=False,
    )

    trainer = MetaTrainer(
        editor=editor,
        tokenizer=tokenizer,
        model=model,
        simple_ft=config.get("simple_ft", False),
        args=args,
        train_dataset=train_dataset,
        data_collator=train_dataset.collate_fn,
    )

    model.train()
    trainer.train()
    if model_config.get("meta_adapter_path") is None:
        model.save_pretrained(output_dir, selected_adapters=["meta_adapter"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    main(args.config_path)
