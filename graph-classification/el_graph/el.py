import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class EncourageLoss(nn.Module):
    def __init__(self, opt, cl_eps=1e-5, reduction='mean'):
        super(EncourageLoss, self).__init__()
        self.opt = opt
        self.cl_eps = cl_eps
        self.epoch = 0
        self.reduction=reduction
        # defer_start  bonus_start bonus_gamma bonus_rho

    def forward(self, lprobs, target, is_train=True):
        mle_loss = F.nll_loss(lprobs, target, reduction=self.reduction,)  # -y* log p
        org_loss = mle_loss
        if is_train and not (
                self.opt.defer_start and self.get_epoch() <= self.opt.defer_start):  # defer encourage loss:
            probs = torch.exp(lprobs)
            bg = self.opt.bonus_gamma

            # if bg > 0:
            #     bonus = -torch.pow(probs, bg)  # power bonus
            #     if self.opt.bonus_start != 0.0:  # 20201023 截断bonus
            #         pb = -torch.pow(self.opt.bonus_start *torch.ones_like(probs), bg)
            #         bonus = torch.where(probs >= self.opt.bonus_start, bonus - pb, torch.zeros_like(bonus))
            # else:
            #     bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps))  # likelihood bonus
            #     if self.opt.bonus_start != 0.0:
            #         pb = torch.log(torch.clamp((torch.ones_like(probs) - self.opt.bonus_start), min=self.cl_eps))
            #         bonus = torch.where(probs >= self.opt.bonus_start, bonus - pb, torch.zeros_like(bonus))
            if bg > 0:  # power bonus
                bonus = -torch.pow(probs, bg)  # power bonus
                if self.opt.bonus_start != 0.0:  # 20201023 截断bonus
                    pb = -torch.pow(self.opt.bonus_start * torch.ones_like(probs), bg)
                    bonus = torch.where(probs >= self.opt.bonus_start, bonus - pb, torch.zeros_like(bonus))
            elif bg == -1:  # log
                bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps))  # likelihood bonus
                if self.opt.bonus_start != 0.0:
                    pb = torch.log(torch.clamp((torch.ones_like(probs) - self.opt.bonus_start), min=self.cl_eps))
                    bonus = torch.where(probs >= self.opt.bonus_start, bonus - pb, torch.zeros_like(bonus))
                if self.opt.log_end != 1.0:  # e.g. 0.9
                    log_end = self.opt.log_end
                    y_log_end = torch.log(torch.clamp((torch.ones_like(probs) - log_end), min=self.cl_eps))
                    bonus_after_log_end = 1 / (log_end - torch.clamp((torch.ones_like(probs)), min=self.cl_eps)) * (
                                probs - log_end) + y_log_end
                    # x:log_end, y  torch.log(torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps))
                    bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)
            elif bg == -2:  # cosine
                bonus = torch.cos(probs * math.pi) - 1

            c_loss = F.nll_loss(
                -bonus* self.opt.bonus_rho,
                target.view(-1),
                reduction=self.reduction
            )  # y*log(1-p)
            all_loss = mle_loss + c_loss
            if self.opt.whole_rho>0:
                all_loss = self.opt.whole_rho*all_loss
        else:
            all_loss = mle_loss
        return all_loss, org_loss

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch
