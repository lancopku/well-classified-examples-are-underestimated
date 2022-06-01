# parser.add_argument('--use_el', default=0, type=int,
#                     help='')
# parser.add_argument('--log_end', default=0.75, type=float,
#                     help='')
import torch
import torch.nn as nn
import torch.nn.functional as F
class LabelSmoothingEncouragingLoss(nn.Module):
    """
    Encouraging Loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, log_end=0.75):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingEncouragingLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.log_end=log_end

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        probs = torch.exp(logprobs)
        bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=1e-5))  # likelihood bonus
        if self.log_end != 1.0:  # e.g. 0.75
            log_end = self.log_end
            y_log_end = torch.log(torch.ones_like(probs) - log_end)
            bonus_after_log_end = 1/(log_end - torch.ones_like(probs)) * (probs-log_end) + y_log_end
            bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)
        el_loss =(bonus-logprobs).gather(dim=-1, index=target.unsqueeze(1))
        el_loss = el_loss.squeeze(1)
        smooth_loss = (bonus-logprobs).mean(dim=-1)
        loss = self.confidence * el_loss + self.smoothing * smooth_loss
        return loss.mean()
