from huggingface_hub import snapshot_download
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
import numpy as np
from trl import RewardTrainer, RewardConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import argparse
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from config import (
	MODEL_PATH,
	SCALAR_REWARD_MODEL_PATH,
	DISTRIBUTION_REWARD_MODEL_PATH,
	DATA_PATH,
	LOG_DIR,
	ALIGNED_OUTPUT_DIR,
	NUM_WORKERS,
	N_SCORES
)

from data_preprocess import get_dataloaders
from training.trainer import train_epoch, evaluate
from training.train_utils import get_reward_from_scalar_model, get_reward_from_distributional_model

REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
snapshot_download(repo_id=REPO, local_dir="./model")

def parse_level(args):
	if args.level == 1:
		reward_model_path = SCALAR_REWARD_MODEL_PATH
		get_reward_fn = get_reward_from_scalar_model
		n_labels = 1
	else:
		reward_model_path = DISTRIBUTION_REWARD_MODEL_PATH
		get_reward_fn = get_reward_from_distributional_model
		n_labels = N_SCORES
	
	return reward_model_path, get_reward_fn, n_labels

def main(args):
	accelerator = Accelerator()
	device = accelerator.device
	
	reward_model_path, get_reward_fn, n_labels = parse_level(args)

	reward_model = AutoModelForSequenceClassification.from_pretrained(
		reward_model_path,
		num_labels = n_labels,
		local_files_only=True,
		torch_dtype=torch.float16
	)

	reward_model.eval()

	model = AutoModelForCausalLM.from_pretrained(
		MODEL_PATH,
		local_files_only=True,
	)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	train_loader, val_loader = get_dataloaders(DATA_PATH, args.batch_size, NUM_WORKERS)
	model, reward_model, optimizer, train_loader, val_loader = accelerator.prepare(model, reward_model, optimizer, train_loader, val_loader)
	output_dir = ALIGNED_OUTPUT_DIR.format(level=args.level)

	if accelerator.is_main_process:
		os.makedirs(LOG_DIR, exist_ok=True)
		run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
		log_dir = os.path.join(LOG_DIR, run_name)
		writer = SummaryWriter(log_dir=log_dir)
	else:
		writer = None

	train_reward_history = []
	val_reward_history = []
	all_train_rewards = []

	best_reward = float('-inf')
	for epoch in range(args.n_epochs):
		train_reward, all_train_rewards = train_epoch(
			model,
			reward_model,
			optimizer,
			train_loader,
			get_reward_fn,
			all_train_rewards,
			accelerator,
			writer,
			epoch,
			args.max_new_tokens
		)
		train_reward_history.append(train_reward)

		val_reward = evaluate(model, reward_model, val_loader, get_reward_fn, accelerator, writer, epoch, args.max_new_tokens)
		val_reward_history.append(val_reward)

		if accelerator.is_main_process:
			print(f"Epoch {epoch + 1}/{args.n_epochs} — Train reward: {train_reward:.4f}, Val reward: {val_reward:.4f}")
			writer.add_scalar("Reward/Train", train_reward, epoch)
			writer.add_scalar("Reward/Val", val_reward, epoch)
			if val_reward > best_reward:
				best_reward = val_reward
				unwrapped = accelerator.unwrap_model(model)
				unwrapped.save_pretrained(output_dir, save_function=accelerator.save)
				print(f"Best model saved with val_reward={val_reward:.4f}")

	if accelerator.is_main_process:
		writer.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=3)
	parser.add_argument('--lr', type=float, default=3e-5)
	parser.add_argument('--n_epochs', type=int, default=10)
	parser.add_argument('--max_new_tokens', type=int, default=32)
	parser.add_argument('--level', type=int, choices=[1, 2], default=2)
	
	args = parser.parse_args()
	main(args)