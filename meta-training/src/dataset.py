from torch.utils.data import Dataset
from typing import List, Dict
import json
import logging


class MetaTrainDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        logging.info(f"Loaded {len(self.data)} data pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_texts = item["cloze_question"]
        facts = item["requested_rewrite"][0]

        return [
            {
                "prompt": fact["prompt"].format(fact["subject"]),
                "target_new": fact["target_new"]["str"],
                "ground_truth": fact["target_true"]["str"],
                "subject": fact["subject"],
                "portability_r": [
                    {
                        "prompt": prompt_text,
                        "ground_truth": item["new_answer"][i],
                    }
                    for i, prompt_text in enumerate(prompt_texts)
                    if fact_idx == 0
                ],
            }
            for fact_idx, fact in enumerate(facts)
        ]

    def collate_fn(self, batch):
        return [item for sublist in batch for item in sublist]
