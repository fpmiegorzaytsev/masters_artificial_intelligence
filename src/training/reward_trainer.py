import torch
from trl import RewardTrainer
from .train_utils import pairwise_distribution_loss, expectation, entropy

class DistributionRewardTrainer(RewardTrainer):
  def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    logits_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"]).logits
    logits_rejected = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"]).logits
    loss = pairwise_distribution_loss(logits_chosen, logits_rejected)
    entropy_chosen = entropy(logits_chosen).mean()
    entropy_rejected = entropy(logits_rejected).mean()
    loss = loss * 0.9 + 0.1 * (entropy_chosen + entropy_rejected)
    if return_outputs:
        return loss, {
          "rewards_chosen": expectation(logits_chosen).unsqueeze(1),
          "rewards_rejected": expectation(logits_rejected).unsqueeze(1)
        }
    return loss