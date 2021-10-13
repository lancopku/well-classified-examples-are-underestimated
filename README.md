### This is a repository for the paper of "Well-classified Examples are Underestimated in Classification with Deep Neural Networks"

The implementation and scripts for training and evaluation can be found in corresponding directory for each task.

### Example implementation
```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class EncourageLoss(nn.Module):
    def __init__(self, log_end=0.75, reduction='mean'):
        super(EncourageLoss, self).__init__()
        self.log_end = log_end  # 1 refers to the normal bonus, but 0.75 can easily work in existing optimization systems
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

```
