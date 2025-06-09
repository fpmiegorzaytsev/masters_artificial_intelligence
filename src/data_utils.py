import torch
from transformers import AutoTokenizer

MODEL_PATH = "./model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")

def tokenize(example):
	kwargs = {"max_length": 204, "truncation": True, "padding": "max_length"}
	
	prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False)

	tokenized = tokenizer(prompt, **kwargs)

	return {
			"input_ids": tokenized["input_ids"],
			"attention_mask": tokenized["attention_mask"]
	}

def make_prompts(example):
	return {
			"prompt": [
					{"role": "user", "content": example["prompt"]}
			]
	}

def process(dataset):
	all_columns = list(dataset.features.keys())
	dataset = dataset.map(make_prompts, remove_columns=all_columns)
	dataset = dataset.map(tokenize, remove_columns=["prompt"])
	return dataset