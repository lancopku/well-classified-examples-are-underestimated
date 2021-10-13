# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn.functional as F
import math # add for encouraging loss


class ARGS():
    def __init__(
            self,
            bonus_gamma=0,
            bonus_rho=1.0,
            bonus_start=0.0,
            defer_start=0,
            log_end=1,
            el_lb=0,
            el_mask_pad=0,
            el_schedule=None,
            el_schedule_max_epoch=None,
            el_linear_schedule_start=0,
            el_linear_schedule_end=None,
            el_exp_schedule_ratio=5,
            sub_data_rho=1.0,
            bonus_abrupt=-1,
            all_dynamic_rho=0,
            all_dynamic_data='fren'
    ):
        self.bonus_gamma=bonus_gamma
        self.bonus_rho=bonus_rho
        self.bonus_start =bonus_start
        self.defer_start=defer_start
        self.log_end=log_end
        self.el_lb=el_lb
        self.el_mask_pad=el_mask_pad
        self.el_schedule=el_schedule
        self.el_schedule_max_epoch=el_schedule_max_epoch
        self.el_linear_schedule_start=el_linear_schedule_start
        self.el_linear_schedule_end=el_linear_schedule_end
        self.el_exp_schedule_ratio=el_exp_schedule_ratio
        self.sub_data_rho=sub_data_rho
        self.bonus_abrupt=bonus_abrupt
        self.all_dynamic_rho=all_dynamic_rho
        self.all_dynamic_data=all_dynamic_data


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)  # 这个是为了gather时用的
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    # (1-epsilon)*y *-logp +epsilon/N * -logp=((1-epsilon)*y+epsilon/N)*-logp
    #y*log(1-p)->((1-epsilon)*y+epsilon/N)*log(1-p)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

# 施工完成才能叫 encouraging loss
@register_criterion("encourage_loss")
class EncourageLoss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        bonus_gamma=0,
        bonus_rho=1.0,
        bonus_start=0.0,
        defer_start=0,
        log_end=1,
        el_lb=0,
        el_mask_pad=0,
        el_schedule=None,
        el_schedule_max_epoch=None,
        el_linear_schedule_start=0,
        el_linear_schedule_end=None,
        el_exp_schedule_ratio=5,
            sub_data_rho=1.0,
            bonus_abrupt=-1,
            all_dynamic_rho=0,
            all_dynamic_data='fren'
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        # add for encouraging loss
        self.args = ARGS(bonus_gamma,bonus_rho,bonus_start,defer_start,log_end,el_lb,el_mask_pad,
                         el_schedule,el_schedule_max_epoch,el_linear_schedule_start,el_linear_schedule_end,
                         el_exp_schedule_ratio,sub_data_rho,bonus_abrupt,all_dynamic_rho,all_dynamic_data=all_dynamic_data)
        self.set_epoch(1)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # add for encouraging loss
        parser.add_argument('--bonus_gamma', type=int, default=0)
        parser.add_argument('--bonus_rho', type=float, default=1.0)
        parser.add_argument('--bonus_start', type=float, default=0.0)
        parser.add_argument('--defer_start', type=int, default=0)
        parser.add_argument('--log_end', type=float, default=1.0)
        # 2021 3.29 i think label smoothing could also be applied to the encouraging loss
        parser.add_argument('--el_lb', type=float, default=0)
        parser.add_argument('--el_mask_pad', type=int, default=0) # should be set to 1
        # 2021 3.31 add schedule
        parser.add_argument('--el_schedule', type=str, choices=[None,'linear','exp','cos'],default=None)
        parser.add_argument('--el_schedule_max_epoch', type=int,
                            default=None)
        parser.add_argument('--el_linear_schedule_start', type=int,default=0)
        parser.add_argument('--el_linear_schedule_end', type=int,default=None)
        parser.add_argument('--el_exp_schedule_ratio', type=int,default=5)
        parser.add_argument('--sub_data_rho', type=float, default=1.0, help='')
        parser.add_argument('--bonus_abrupt', type=float, default=-1, help='')
        parser.add_argument('--all_dynamic_rho', type=int, default=0, help='')
        parser.add_argument('--all_dynamic_data', type=str, default='fren', help='')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        # ----- add for encouraging loss  -----
        bonus = None
        if self.args.bonus_gamma != 0:
            probs = torch.exp(lprobs)
        if self.args.bonus_gamma > 0:
            bonus = -torch.pow(probs, self.args.bonus_gamma)  # power bonus
            if self.args.bonus_start != 0.0:  # 20201023 截断bonus
                pb = -torch.pow(self.args.bonus_start * torch.ones_like(probs), self.args.bonus_gamma)
                bonus = torch.where(probs >= self.args.bonus_start, bonus - pb, torch.zeros_like(bonus))
        elif self.args.bonus_gamma == -1:  # log
            if self.args.sub_data_rho<1:
                bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs*self.args.sub_data_rho), min=1e-5))  # likelihood bonus(sub less data)
            else:
                bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=1e-5))  # likelihood bonus
            if self.args.bonus_start != 0.0:
                pb = torch.log(torch.clamp((torch.ones_like(probs) - self.args.bonus_start), min=1e-5))
                bonus = torch.where(probs >= self.args.bonus_start, bonus - pb, torch.zeros_like(bonus))
            if self.args.log_end != 1.0:  # e.g. 0.9
                log_end = self.args.log_end
                y_log_end = torch.log(torch.ones_like(probs) - log_end)
                bonus_after_log_end = 1/(log_end - torch.ones_like(probs)) * (probs-log_end) + y_log_end
                # x:log_end, y  torch.log(torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps))
                bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)
        elif self.args.bonus_gamma == -2:  # cosine
            bonus = torch.cos(probs*math.pi) - 1

        if bonus is not None:
            # for increasing bonus ratio along the training procedure
            rho = 1
            if self.args.el_schedule is not None:
                cur_time = float(self.get_epoch()) /self.args.el_schedule_max_epoch # t/T
                if self.args.el_schedule == 'linear':
                    rho = cur_time
                elif self.args.el_schedule == 'exp':
                    if self.args.el_exp_schedule_ratio > 0:  # default 5 exp((t/T-1)*5)
                        rho = math.exp((cur_time-1)*self.args.el_exp_schedule_ratio)
                    else:  # el_exp_schedule_ratio<0  1- exp((t/T)*-5)
                        rho = 1 - math.exp(cur_time * self.args.el_exp_schedule_ratio)
                elif self.args.el_schedule == 'cos':  # (cos(pi+t/T*pi)+1)*0.5
                    rho = (math.cos(math.pi*(1+cur_time))+1)*0.5
            bonus_rho = self.args.bonus_rho*rho
            if self.args.bonus_abrupt > 0:  # can be 0.75
                bonus = torch.where(probs > self.args.bonus_abrupt, torch.zeros_like(bonus), bonus)
            c_loss = F.nll_loss(
                -bonus * bonus_rho,
                target.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx if self.args.el_mask_pad and self.padding_idx is not None else -100
            )  # y*log(1-p)

            if self.args.el_lb != 0:
                smoothing_c_loss = bonus.sum(dim=-1) * rho
                if self.args.el_mask_pad and self.padding_idx is not None:
                    pad_mask = target.view(-1).eq(self.padding_idx)
                    smoothing_c_loss.masked_fill_(pad_mask, 0.0)
                smoothing_c_loss = smoothing_c_loss.sum()
                c_loss = c_loss*(1-self.args.el_lb) + (self.args.el_lb/lprobs.size(-1))*smoothing_c_loss
        else:
            c_loss = 0
        loss = loss + c_loss

        fren_le05=[0.91, 0.84, 0.81, 0.8, 0.79, 0.79, 0.79, 0.78, 0.78, 0.78, 0.78, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77,
         0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77,
         0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77,
         0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77]
        fren_le075=[0.89, 0.79, 0.76, 0.74, 0.73, 0.72, 0.72, 0.72, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.7, 0.7, 0.7, 0.7,
         0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
         0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        fren_le09=[0.88, 0.74, 0.7, 0.67, 0.66, 0.65, 0.65, 0.64, 0.64, 0.64, 0.64, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63,
         0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62,
         0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.61, 0.62, 0.61, 0.62, 0.62, 0.62,
         0.62, 0.62, 0.62, 0.61, 0.61, 0.62, 0.61]
        fren_le1=[0.79, 0.5, 0.39, 0.33, 0.31, 0.29, 0.28, 0.28, 0.27, 0.27, 0.27, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.25,
         0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24,
         0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.23, 0.23, 0.23, 0.23,
         0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23]

        deen_le05=[0.94, 0.86, 0.84, 0.85, 0.84, 0.82, 0.83, 0.82, 0.81, 0.8, 0.8, 0.79, 0.78, 0.77, 0.78, 0.78, 0.77, 0.77, 0.77,
         0.77, 0.77, 0.77, 0.77, 0.77, 0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.75, 0.75, 0.76, 0.76, 0.75, 0.75, 0.75,
         0.75, 0.75, 0.75, 0.75, 0.74, 0.75, 0.75, 0.75, 0.74, 0.75, 0.74, 0.75, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74,
         0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.73, 0.74, 0.74, 0.74, 0.74, 0.74, 0.73, 0.74, 0.74, 0.74, 0.73]
        deen_le075=[0.94, 0.85, 0.8, 0.81, 0.8, 0.79, 0.78, 0.78, 0.76, 0.75, 0.73, 0.73, 0.71, 0.71, 0.72, 0.7, 0.7, 0.7, 0.7,
         0.7, 0.7, 0.69, 0.69, 0.69, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.67, 0.68, 0.67, 0.67,
         0.67, 0.67, 0.67, 0.67, 0.66, 0.67, 0.67, 0.67, 0.66, 0.66, 0.66, 0.66, 0.67, 0.66, 0.66, 0.66, 0.66, 0.66,
         0.66, 0.66, 0.65, 0.65, 0.66, 0.66, 0.65, 0.66, 0.65, 0.66, 0.65, 0.65, 0.65, 0.65, 0.65, 0.66, 0.65]
        deen_le1=[0.88, 0.75, 0.65, 0.63, 0.59, 0.56, 0.51, 0.45, 0.42, 0.39, 0.36, 0.34, 0.32, 0.3, 0.29, 0.28, 0.27, 0.26,
         0.25, 0.25, 0.25, 0.24, 0.24, 0.24, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.22, 0.23, 0.22, 0.22,
         0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21,
         0.21, 0.21, 0.21, 0.2, 0.2, 0.21, 0.21, 0.2, 0.2, 0.21, 0.2, 0.2, 0.2, 0.21, 0.2, 0.2, 0.2, 0.2, 0.2]


        # ----- end encouraging loss  -----
        if self.args.all_dynamic_rho:
            mid_loss = loss
            mid_nll_loss=nll_loss
            if self.args.all_dynamic_data =='fren':
                if self.args.log_end ==0.5:
                    loss = fren_le05[self.get_epoch()-1]*mid_loss
                    nll_loss = fren_le05[self.get_epoch()-1]*mid_nll_loss
                if self.args.log_end ==0.75:
                    loss = fren_le075[self.get_epoch()-1]*mid_loss
                    nll_loss = fren_le075[self.get_epoch()-1]*mid_nll_loss
                if self.args.log_end ==0.9:
                    loss = fren_le09[self.get_epoch()-1]*mid_loss
                    nll_loss = fren_le09[self.get_epoch()-1]*mid_nll_loss
                if self.args.log_end ==1:
                    loss = fren_le1[self.get_epoch()-1]*mid_loss
                    nll_loss = fren_le1[self.get_epoch()-1]*mid_nll_loss
            elif self.args.all_dynamic_data =='deen':
                if self.args.log_end ==0.5:
                    loss = deen_le05[self.get_epoch()-1]*mid_loss
                    nll_loss = deen_le05[self.get_epoch()-1]*mid_nll_loss
                if self.args.log_end ==0.75:
                    loss = deen_le075[self.get_epoch()-1]*mid_loss
                    nll_loss = deen_le075[self.get_epoch()-1]*mid_nll_loss
                if self.args.log_end ==1:
                    loss = deen_le1[self.get_epoch()-1]*mid_loss
                    nll_loss = deen_le1[self.get_epoch()-1]*mid_nll_loss
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def set_epoch(self,epoch):
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch