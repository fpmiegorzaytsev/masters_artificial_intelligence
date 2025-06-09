import torch
import numpy as np
from tqdm import tqdm
import os

def pairwise_distribution_loss(logits_p, logits_q):
	p = torch.nn.functional.softmax(logits_p, dim=1)
	q = torch.nn.functional.softmax(logits_q, dim=1)
	joint = p.unsqueeze(2) * q.unsqueeze(1)
	mask = torch.tril(torch.ones(p.shape[1], p.shape[1]), diagonal=-1).to(logits_p.device)
	prob = torch.sum(joint * mask, dim=(1, 2))
	return -torch.log(prob).mean()

def expectation(logits):
	probs = torch.nn.functional.softmax(logits, dim=1)
	values = torch.arange(logits.shape[1], dtype=torch.float, device=logits.device) + 1
	return torch.sum(probs * values, dim=1)

def entropy(logits):
	logsumexp = torch.logsumexp(logits, dim=1)
	probs = torch.softmax(logits, dim=1)
	expected_logit = (probs * logits).sum(dim=1)
	return logsumexp - expected_logit
    
def compute_loss(log_probs, reward, moving_averages):
	baseline = torch.mean(moving_averages)
	advantage = (reward - baseline).detach()
	return -log_probs * advantage

@torch.no_grad()
def get_reward_from_distributional_model(reward_model, generated):
	attention_mask = (generated != reward_model.module.config.eos_token_id)
	logits = reward_model(input_ids=generated, attention_mask=attention_mask).logits
	reward = expectation(logits)
	return reward

@torch.no_grad()
def get_reward_from_scalar_model(reward_model, generated):
	attention_mask = (generated != reward_model.module.config.eos_token_id)
	reward = reward_model(input_ids=generated, attention_mask=attention_mask)
	return reward.logits.squeeze(1)

def generate(model, input_ids, attention_mask=None, max_new_tokens=64):
	eos_token_id = model.module.config.eos_token_id
	pad_token_id = eos_token_id
	device = model.device

	generated = input_ids.detach()

	if attention_mask is None:
		attention_mask = (generated != pad_token_id)

	log_probs_sum = torch.zeros(generated.shape[0], device=device)
	finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=device)

	past_key_values = None

	for _ in range(max_new_tokens):

		if past_key_values is None:
			outputs = model(input_ids=generated, attention_mask=attention_mask)
		else:
			outputs = model(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
				
		logits = outputs.logits[:, -1, :]
		past_key_values = outputs.past_key_values
		
		log_prob = torch.log_softmax(logits, dim=-1)
		distribtuion = torch.distributions.categorical.Categorical(logits=logits)
		next_token = distribtuion.sample().unsqueeze(1).detach()
		
		next_token = torch.where(
			finished.unsqueeze(1),
			torch.full_like(next_token, pad_token_id),
			next_token
		)
		
		finished |= (next_token.squeeze(1) == eos_token_id)

		log_prob = torch.gather(log_prob, dim=1, index=next_token).squeeze(1)
		log_probs_sum = log_probs_sum + log_prob
		
		generated = torch.cat([generated, next_token], dim=1)

		if finished.all():
			break

	return generated, log_probs_sum

