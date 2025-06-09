from transformers import AutoTokenizer
import torch
from config import MODEL_PATH, MAX_LENGTH_PROMPTS, MAX_LENGTH_PAIRS
from torch.utils.data import DataLoader
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

def tokenize_prompts(example):
  prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False)

  kwargs = {"max_length": MAX_LENGTH_PROMPTS, "truncation": True, "padding": "max_length"}

  tokenized = tokenizer(prompt, **kwargs)

  return {
      "input_ids": tokenized["input_ids"],
      "attention_mask": tokenized["attention_mask"]
  }

def tokenize_pairs(example):
  text_chosen = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
  text_rejected = tokenizer.apply_chat_template(example["rejected"], tokenize=False)

  kwargs = {"max_length": MAX_LENGTH_PAIRS, "truncation": True, "padding": "max_length"}

  tokenized_chosen = tokenizer(text_chosen, **kwargs)
  tokenized_rejected = tokenizer(text_rejected, **kwargs)

  return {
      "input_ids_chosen": tokenized_chosen["input_ids"],
      "attention_mask_chosen": tokenized_chosen["attention_mask"],
      "input_ids_rejected": tokenized_rejected["input_ids"],
      "attention_mask_rejected": tokenized_rejected["attention_mask"],
  }

def process_prompts_dataset(dataset):
  dataset = dataset.map(lambda ex: {"prompt": [{"role": "user", "content": ex["prompt"]}]}, remove_columns=dataset.column_names)
  dataset = dataset.map(tokenize_prompts, remove_columns=["prompt"])
  return dataset

def process_pairs_dataset(dataset):
  dataset = dataset.map(
    lambda ex: {
        "chosen": [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["chosen"]}
        ],
        "rejected": [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["rejected"]}
        ]
    }, remove_columns=dataset.column_names)
  dataset = dataset.map(tokenize_pairs, remove_columns=["chosen", "rejected"])
  return dataset


def get_dataloaders(data_path, batch_size, num_workers):
  format_kwargs = {'type': 'torch', 'format_kwargs' :{'dtype': torch.int32}}
  train_data = process_prompts_dataset(load_dataset(data_path, split="train"))
  val_data = process_prompts_dataset(load_dataset(data_path, split="validation"))

  train_data.set_format(**format_kwargs)
  val_data.set_format(**format_kwargs)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

  return train_loader, val_loader