import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class EncourageLoss(nn.Module):
    def __init__(self, opt, ignore_idx=-100, reduction='mean',soft_target=False, cl_eps=1e-5,):
        super(EncourageLoss, self).__init__()
        self.opt = opt
        self.cl_eps = cl_eps
        self.epoch = 0
        self.ignore_index = ignore_idx
        self.reduction = reduction  # 以后公开的简化版再支持reduction eval('torch.'+self.reduction)(loss)
        self.soft_target = soft_target

    def forward(self, input, target, is_train=True):
        lprobs = F.log_softmax(input,dim=-1)
        probs = torch.exp(lprobs)
        if self.opt.base_loss in ['mse','mae','mse_sigmoid','mae_sigmoid']:
            ys = F.one_hot(target, num_classes=probs.size()[-1])
            ones = torch.ones_like(probs)
            if self.opt.base_loss == 'mse':
                mse_loss = torch.mean(ys*(ones-probs)**2+(ones-ys)*probs**2)  # (y*(1-p)^2 +(1-y)p^2)
                org_loss = mse_loss
                # identical to torch.mean((F.one_hot(target, num_classes=probs.size()[-1])-probs)**2)
            elif self.opt.base_loss == 'mse_sigmoid':
                mse_loss = torch.mean(ys*(ones-torch.sigmoid(input))**2+(ones-ys)*torch.sigmoid(input)**2)  # (y*(1-p)^2 +(1-y)p^2)
                org_loss = mse_loss
            elif self.opt.base_loss == 'mae':
                mae_loss = torch.mean(ys*(ones-probs)+(ones-ys)*probs)  # ( y*(1-p) +(1-y)p)
                org_loss = mae_loss
            elif self.opt.base_loss == 'mae_sigmoid':
                mae_loss = torch.mean(ys*(ones-torch.sigmoid(input))+(ones-ys)*torch.sigmoid(input))  # ( y*(1-p) +(1-y)p)
                org_loss = mae_loss
            nll_loss = torch.zeros_like(org_loss)
        else:
            if self.soft_target:
                ce_loss = torch.sum(-target*lprobs,dim=-1).mean()
            else:
                ce_loss = F.nll_loss(lprobs, target, reduction='mean', )  # -y* log p
            nll_loss = ce_loss
            # print('self.opt.label_smoothing',self.opt.label_smoothing)
            if self.opt.label_smoothing > 0:
                org_loss = (1.0-self.opt.label_smoothing)*ce_loss + self.opt.label_smoothing*(-lprobs).mean(-1).mean()
        # ----- add for encouraging loss  -----
        c_loss = torch.zeros_like(org_loss)
        if is_train and not (
                self.opt.defer_start and self.get_epoch() <= self.opt.defer_start):  # defer encourage loss:
            bg = self.opt.bonus_gamma
            if bg != 0:
                if self.opt.base_loss in ['mse','mae','mse_sigmoid','mae_sigmoid']:
                    # mse: min mean y*(1-p)^2 +(1-y)p^2
                    # mirror mse: min  - ( mean y*p^2 + (1-y)*(1-p)^2)
                    preds=probs if self.opt.base_loss in ['mse', 'mae'] else torch.sigmoid(input)
                    mirror_me = -torch.mean(ys*preds**bg+(ones-ys)*(ones-preds)**bg)
                    c_loss = mirror_me * self.opt.bonus_rho
                else:
                    bonus = None
                    if bg > 0:  # power bonus
                        bonus = -torch.pow(probs, bg)  # power bonus
                        if self.opt.bonus_start != 0.0:  # 20201023 截断bonus
                            pb = -torch.pow(self.opt.bonus_start *torch.ones_like(probs), bg)
                            bonus = torch.where(probs >= self.opt.bonus_start, bonus - pb, torch.zeros_like(bonus))
                    elif bg == -1:  # log
                        bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps))  # likelihood bonus
                        if self.opt.bonus_start != 0.0:
                            pb = torch.log(torch.clamp((torch.ones_like(probs) - self.opt.bonus_start), min=self.cl_eps))
                            bonus = torch.where(probs >= self.opt.bonus_start, bonus - pb, torch.zeros_like(bonus))
                        if self.opt.log_end != 1.0:  # e.g. 0.9
                            log_end = self.opt.log_end
                            y_log_end = torch.log(torch.ones_like(probs) - log_end)
                            bonus_after_log_end = 1/(log_end - torch.ones_like(probs)) * (probs-log_end) + y_log_end
                            bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)
                    elif bg == -2:  # cosine
                        bonus = torch.cos(probs*math.pi) - 1
                    if bonus is not None:
                        # for increasing bonus ratio along the training procedure
                        rho = 1
                        if self.opt.el_schedule is not None:
                            cur_time = float(self.get_epoch()) / self.opt.el_schedule_max_epoch  # t/T
                            if self.opt.el_schedule == 'linear':
                                rho = cur_time
                            elif self.opt.el_schedule == 'exp':
                                if self.opt.el_exp_schedule_ratio > 0:  # default 5 exp((t/T-1)*5)
                                    rho = math.exp((cur_time - 1) * self.opt.el_exp_schedule_ratio)
                                else:  # el_exp_schedule_ratio<0  1- exp((t/T)*-5)
                                    rho = 1 - math.exp(cur_time * self.opt.el_exp_schedule_ratio)
                            elif self.opt.el_schedule == 'cos':  # (cos(pi+t/T*pi)+1)*0.5
                                rho = (math.cos(math.pi * (1 + cur_time)) + 1) * 0.5
                        bonus_rho = self.opt.bonus_rho * rho
                        if self.opt.bonus_abrupt > 0:  # can be 0.75
                            bonus = torch.where(probs > self.opt.bonus_abrupt, torch.zeros_like(bonus), bonus)
                        if self.soft_target:
                            c_loss = torch.sum(-target * (-bonus * bonus_rho), dim=-1).mean()
                        else:
                            c_loss = F.nll_loss(
                                -bonus * bonus_rho,
                                target.view(-1),
                                reduction='mean',
                            )  # y*log(1-p)
                        if self.opt.label_smoothing != 0:
                            smoothing_c_loss = bonus.mean(dim=-1) * bonus_rho
                            # if self.ignore_index !=-100:
                            #     pad_mask = target.view(-1).eq(self.ignore_index)
                            #     smoothing_c_loss.masked_fill_(pad_mask, 0.0)
                            smoothing_c_loss = smoothing_c_loss.mean()
                            c_loss = c_loss * (1 - self.opt.label_smoothing) + self.opt.label_smoothing*smoothing_c_loss
        all_loss = org_loss + c_loss
        return all_loss, org_loss, nll_loss

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch
