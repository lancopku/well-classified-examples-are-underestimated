# This is the encouraging loss we used in the paper
import torch
import torch.nn as nn
from torch.nn import functional as F

class EncouragingLoss(nn.Module):
    def __init__(self, log_end=0.75, reduction='mean'):
        super(EncouragingLoss, self).__init__()
        self.log_end = log_end
        self.reduction = reduction

    def forward(self, input, target):
        lprobs = F.log_softmax(input)  # logp
        probs = torch.exp(lprobs)
        bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=1e-5))  # log(1-p)
        if self.log_end != 1.0:  # end of the log curve in conservative bonus # e.g. 0.5  work for all settings
            log_end = self.log_end
            y_log_end = torch.log(torch.ones_like(probs) - log_end)
            bonus_after_log_end = 1/(log_end - torch.ones_like(probs)) * (probs-log_end) + y_log_end
            bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)
        loss = F.nll_loss(lprobs-bonus, target.view(-1), reduction=self.reduction)
        return loss





