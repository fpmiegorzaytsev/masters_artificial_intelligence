import torch
from tqdm import tqdm
import numpy as np
from .train_utils import generate, compute_loss

def train_epoch(
  model,
  reward_model,
  optimizer,
  dataloader,
  get_reward_fn,
  all_rewards,
  accelerator,
  writer,
  epoch,
  max_new_tokens
):
	reward_history_epoch = []

	for step, batch in enumerate(tqdm(dataloader, desc="Training epoch", disable=not accelerator.is_main_process)):
		input_ids = batch["input_ids"]
		attention_mask = batch["attention_mask"]

		generated, log_probs = generate(model, input_ids, attention_mask, max_new_tokens)
		reward = get_reward_fn(reward_model, generated)

		reward_history_epoch.extend(reward.detach().cpu().numpy().tolist())
		all_rewards.extend(reward.detach().cpu().numpy().tolist())

		loss = compute_loss(log_probs, reward, torch.tensor(all_rewards, dtype=float, device=log_probs.device))
		optimizer.zero_grad()
		accelerator.backward(loss.mean())
		optimizer.step()

	return np.mean(reward_history_epoch), all_rewards

@torch.no_grad()
def evaluate(model, reward_model, dataloader, get_reward_fn, accelerator, writer, epoch, max_new_tokens):
  reward_history = []
  for batch in tqdm(dataloader, desc="Validation epoch", disable=not accelerator.is_main_process):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    generated, _ = generate(model, input_ids, attention_mask, max_new_tokens)
    reward = get_reward_fn(reward_model, generated)
    reward_history.extend(reward.detach().cpu().numpy().tolist())

  return np.mean(reward_history)