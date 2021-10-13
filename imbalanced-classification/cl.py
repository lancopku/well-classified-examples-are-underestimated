import torch
import torch.nn as nn
from torch.nn import functional as F

class CourageCriterion(nn.Module):
    def __init__(self, opt, num_class, weight=None, ignore_idx=-100,cl_eps=1e-5):
        super(CourageCriterion, self).__init__()
        self.opt = opt
        self.num_class = num_class
        self.ignore_idx = ignore_idx
        self.weight = weight
        self.cl_eps = cl_eps
        self.base_loss=None
        self.courage_items = {'start_encourage': 'none',
                              'epoch_i': 1,
                              'courage_sum': torch.zeros(num_class, dtype=torch.float, requires_grad=False).cuda(),
                              'y_sum': torch.zeros(num_class, dtype=torch.float, requires_grad=False).cuda(),
                              'class_correct': torch.zeros(num_class, requires_grad=False, dtype=torch.float).cuda(),
                              'class_total': torch.zeros(num_class, requires_grad=False, dtype=torch.float).cuda(),
                              'courage': torch.zeros(num_class, dtype=torch.float, requires_grad=False).cuda(),
                              }
    def if_uniform_courage(self):
        if self.opt.uniform_courage:
            self.weight = torch.ones_like(self.weight)
    def set_piece_weight(self, cls_num_list):
        if self.opt.el_part != 1.0:
            num_cls_cuda = torch.from_numpy(cls_num_list).type_as(self.weight)
            many_bool = num_cls_cuda > 100
            medium_bool = ((num_cls_cuda >= 20).int() * (num_cls_cuda <= 100).int()).bool()
            few_bool = num_cls_cuda < 20
            if self.opt.el_part < 1.0:
                _, indices = torch.sort(num_cls_cuda, descending=True)
                # 如果 el_part 0.8 的话，前20% 的常见类权重置零
                selected_indices = indices[:int((1 - self.opt.el_part) * self.num_class)]
                self.weight[selected_indices] = 0.0
            else:
                if self.opt.el_part == 7:  # 110+1 set weights of few to 0
                    self.weight[few_bool] = 0.0
                elif self.opt.el_part == 4:  # 011 +1  set weights of many to 0
                    self.weight[many_bool] = 0.0
                elif self.opt.el_part == 2:  # 001 +1 set weights of many and medium to 0
                    self.weight[many_bool] = 0.0
                    self.weight[medium_bool] = 0.0
    def set_epoch(self, epoch_i):
        self.courage_items['epoch_i'] = epoch_i + 1  # 从1 开始计数
    def get_epoch(self):
        return  self.courage_items['epoch_i']
    def report_courage(self):
        """report courage """
        c = {}
        courage = self.courage_items['courage'].masked_select(self.courage_items['courage'] != 1)
        courage = courage.masked_select(courage != 0)
        if courage is not None and courage.size()[0] > 0:
            c['max'] = torch.max(courage).item()
            c['min'] = torch.min(courage).item()
            c['mean'] = torch.mean(courage).item()
            c['median'] = torch.median(courage).item()
        else:
            c['max'] = 0
            c['min'] = 0
            c['mean'] = 0
            c['median'] = 0
        return c

    def set_mode(self):
        if self.courage_items['epoch_i'] >= self.opt.start_encourage_epoch:
            self.courage_items['start_encourage'] = 'start'  # start encouraging and  update courage every step
        elif self.courage_items['epoch_i'] == self.opt.start_encourage_epoch - 1:
            self.courage_items['start_encourage'] = 'stat'
            # do CE training and statistic courage over the last epoch before the start
        else:
            self.courage_items['start_encourage'] = 'none'  # only do CE training

    def maybe_restat_courage(self):
        if self.courage_items['start_encourage'] == 'stat' or (self.courage_items['start_encourage'] == 'start' and self.opt.always_stat):
            self.courage_items['y_sum'].masked_fill_(self.courage_items['y_sum'] == 0, 1)
            self.courage_items['courage'] = (
                    self.courage_items['courage_sum'] / self.courage_items['y_sum']).detach()
            self.courage_items['y_sum'] = torch.zeros(self.num_class, dtype=torch.float, requires_grad=False).cuda()
            self.courage_items['courage_sum'] = torch.zeros(self.num_class, dtype=torch.float,
                                                            requires_grad=False).cuda()

    def forward(self, x, target, is_train=True):
        lprobs = F.log_softmax(x) # bsz,num_Classes
        if self.base_loss is None:
            mle_loss = F.nll_loss(lprobs, target, reduction='mean', ignore_index=self.ignore_idx)  # -y* log p
        else:
            # actural base loss  focal ldam ce etc.
            mle_loss = self.base_loss(x, target)
        org_loss = mle_loss
        if self.opt.el_start ==-1:
            self.opt.el_start = self.opt.defer_start
        if is_train and not (self.opt.el_start and self.get_epoch() <= self.opt.el_start):  # 2020/7/12 写在这里
            # 2020/9/27 16:55  这里把el start 从defer start 中区分开
            # 2020/9/29 14:06 epoch从1 开始计数的
            probs = torch.exp(lprobs)
            if self.opt.courage_by_weight:
                # bg =self.opt.bg if 'bg' in self.opt else 0 # 6/5 发现之前都是原版
                bg = self.opt.bonus_gamma
                # if bg > 0: # 2020/6/4 17:51 发现之前>1 的bg1 也是log
                #     bonus = -torch.pow(probs, bg)
                #     if self.opt.bonus_start != 0.0: # 20201023 截断bonus
                #         pb = -torch.pow(self.opt.bonus_start *torch.ones_like(probs), bg)
                #         bonus = torch.where(probs >= self.opt.bonus_start, bonus-pb, torch.zeros_like(bonus))
                # else:
                #     bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps))
                #     if self.opt.bonus_start != 0.0:
                #         pb = torch.log(torch.clamp((torch.ones_like(probs) - self.opt.bonus_start), min=self.cl_eps))
                #         bonus = torch.where(probs >= self.opt.bonus_start, bonus-pb, torch.zeros_like(bonus))
                # 以下是2021 2.2 0：31 改的
                if bg > 0:  # power bonus
                    bonus = -torch.pow(probs, bg)  # power bonus
                    if self.opt.bonus_start != 0.0:  # 20201023 截断bonus
                        pb = -torch.pow(self.opt.bonus_start * torch.ones_like(probs), bg)
                        bonus = torch.where(probs >= self.opt.bonus_start, bonus - pb, torch.zeros_like(bonus))
                elif bg == -1:  # log
                    bonus = torch.log(
                        torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps))  # likelihood bonus
                    if self.opt.bonus_start != 0.0:
                        pb = torch.log(torch.clamp((torch.ones_like(probs) - self.opt.bonus_start), min=self.cl_eps))
                        bonus = torch.where(probs >= self.opt.bonus_start, bonus - pb, torch.zeros_like(bonus))
                    if self.opt.log_end != 1.0:  # e.g. 0.9
                        log_end = self.opt.log_end  # le
                        # 2021 1.31 15：22 发现这里我把两个clamp，min 都写错了，第一个clamp个寂寞，第二个倒是斜率1e6
                        # y_log_end = torch.log(torch.clamp((torch.ones_like(probs) - log_end), min=self.cl_eps))  # log(1-le)
                        # # 斜率 1/(le-1)
                        # bonus_after_log_end = 1 / (torch.clamp(log_end - (torch.ones_like(probs)), min=self.cl_eps)) * (
                        #             probs - log_end) + y_log_end
                        y_log_end = torch.log(torch.ones_like(probs) - log_end)  # log(1-le)
                        # 斜率 1/(le-1)
                        bonus_after_log_end = 1 / (log_end - torch.ones_like(probs)) * (
                                probs - log_end) + y_log_end
                        # x:log_end, y  torch.log(torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps))
                        bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)
                cw = self.opt.cw if 'cw' in self.opt else 'max'  # 17:47 才改过来 想先把测试的测了，然后赶紧跑起来
                if cw == 'max':
                    weight_courage = self.weight / torch.max(self.weight) #2020 6.11 1;54除最大值明显不靠谱
                else:  # cw=
                    weight_courage = self.weight  # 2020 6.11 1;54除最大值明显不靠谱
                self.courage_items['courage'] =weight_courage
                c_loss = F.nll_loss(
                    -bonus * weight_courage * self.opt.bonus_rho,
                    target.view(-1),
                    reduction='mean',
                    ignore_index=self.ignore_idx,
                )  # y*log(1-p)
                # print('c loss',c_loss,'courage',weight_courage)
            else:
                c_loss = self.courage_loss(probs=probs, ys=target, cl_eps=self.cl_eps, ignore_idx=self.ignore_idx)
            lambda_mle, lambda_encourage = self.get_loss_coeffienct()
            all_loss = 2 * lambda_mle * mle_loss + 2 * lambda_encourage * c_loss
            if self.opt.report_class_acc:
                self.stat_class_acc(probs, target)
        else:
            all_loss = mle_loss
        return all_loss, org_loss

    def courage_loss(self, probs, ys, cl_eps=1e-5, ignore_idx=None):
        """

        :param probs: bsz*seq-Len,num-class
        :param ys: bsz*seq_len
        :param cl_eps:
        :param ignore_idx:
        :param courage_items : from trainer
        :return:
        """
        c_loss = torch.zeros([1]).type_as(probs)
        num_class = probs.size(-1)
        if self.courage_items['start_encourage'] != 'none':
            # outputs is logits: bsz tgt_len num_class
            # probs = F.softmax(score_view, dim=-1)  # tgt_len*bsz, num_class bsz*seq_len, num_class
            # 3. calculate courage for current step
            with torch.no_grad():
                y = F.one_hot(ys.view(-1), num_classes=num_class)  # tgt_len*bsz,num_class -> bsz*seq_len, num_class
                # courage_func in ['(1-p)^cl_gamma', '(0.5-p)^cl_gamma', '-p^cl_gamma']
                if self.opt.courage_func == '_p^cl_gamma':
                    step_courage = -torch.pow(probs, self.opt.cl_gamma) * y.float()
                    # tgt_len*bsz, num_class -> bsz*seq_len, num_class
                elif self.opt.courage_func == '(cb-p)^cl_gamma':  # cl_gamma = 3 cb=0.5
                    cb = self.opt.courage_bound
                    if cb>0 and cb<1:
                        step_courage = torch.pow(self.opt.courage_bound * torch.ones_like(probs) - probs,
                                                 self.opt.cl_gamma) * y.float()
                    elif cb == 2:  # micro avg in a batch
                        bound = torch.mean(probs*y)
                        step_courage = torch.pow(bound * torch.ones_like(probs) - probs,
                                             self.opt.cl_gamma) * y.float()
                    elif cb in [3, 4]:  # macro avg over vocab in a batch
                        bound = torch.mean(probs * y,dim=0) #vocab
                        if cb == 3 :  # each class has its own bound
                            bound = bound.masked_fill(bound == 0, 0.5 ) # fill
                            step_courage = torch.pow(bound.unsqueeze(0).expand_as(probs) - probs,
                                                     self.opt.cl_gamma) * y.float()
                        else: # macro avg
                            step_courage = torch.pow(torch.mean(bound) * torch.ones_like(probs) - probs,
                                                 self.opt.cl_gamma) * y.float()

                else:  # '(1-p)^cl_gamma'
                    # step_courage = (1-p)^cl_gamma * y # bsz*tgt_Len, num_class
                    step_courage = torch.pow(torch.ones_like(probs) - probs,
                                             self.opt.cl_gamma) * y.float()  # bsz*tgt_len, num_class
                sum_step_courage = torch.sum(step_courage, dim=0)  # num_class
                sum_step_y = torch.sum(y, dim=0)  # num_class
                if self.courage_items['start_encourage'] == 'stat' or (
                        self.courage_items['start_encourage'] == 'start' and self.opt.always_stat):
                    # for stat courage
                    self.courage_items['courage_sum'] = self.courage_items['courage_sum'] + sum_step_courage
                    self.courage_items['y_sum'] = self.courage_items['y_sum'] + sum_step_y.float()
            if self.courage_items['start_encourage'] == 'start':
                with torch.no_grad():
                    if not self.opt.always_stat:
                        cr = self.opt.cr
                        sum_step_y.masked_fill_(sum_step_y == 0, 1)  # avoid div zero
                        # 因为没有出现的class courage 加的 是0， 按道理应该是选择不改变。这是个 bug 啊，之前sgm上的实验按道理也要重跑，如果不是as
                        # 因为0。5-p接近0 所以成功吗。
                        norm_step_courage = sum_step_courage / sum_step_y.float()  # num_class
                        self.courage_items['courage'] = torch.where(norm_step_courage != 0, (1 - cr) * self.courage_items[
                            'courage'] + cr * norm_step_courage, self.courage_items['courage'])
                bg =self.opt.bg if 'bg' in self.opt else 0
                if bg>0:
                    bonus = -torch.pow(probs,bg)
                else:
                    bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=cl_eps))
                # bonus = torch.log(torch.ones_like(probs) - probs + cl_eps)  # log(1-p)
                # self.courage_items['courage'].detach_
                if not self.opt.norm_courage:
                    final_courage = self.courage_items['courage']
                else:
                    final_courage = self.courage_items['courage'] / torch.mean(self.courage_items['courage'])
                c_loss = F.nll_loss(
                    -bonus * final_courage * self.opt.bonus_rho,
                    ys.view(-1),
                    reduction='mean',
                    ignore_index=ignore_idx,
                )  # y*log(1-p)
        return c_loss

    def get_loss_coeffienct(self):
        opt =self.opt
        epoch_i = self.courage_items['epoch_i']
        # a simple linear warmup
        if opt.encourage_warmup_epoch == 0:  # no warmup
            return 0.5, 0.5
        else:
            assert opt.encourage_warmup_epoch > 0, "Encourage warm-up steps is supposed to greater than zero!"
            lambda_encourage = min(0.5, 0.5 * ((epoch_i/ opt.encourage_warmup_epoch)**opt.warmup_gamma))
            return 1 - lambda_encourage, lambda_encourage

    def reset_class_acc(self):
        """reset class acc at the begging of each epoch"""
        self.courage_items['class_correct'] = torch.zeros(self.num_class, requires_grad=False, dtype=torch.float).cuda()
        self.courage_items['class_total'] = torch.zeros(self.num_class, requires_grad=False, dtype=torch.float).cuda()

    def stat_class_acc(self, probs, ys):
        _, pred = torch.max(probs, 1)
        pred_hot = F.one_hot(pred, num_classes=self.num_class)  # bsz*tgt_len, num_class
        targets_hot = F.one_hot(ys, num_classes=self.num_class)
        step_class_correct = torch.sum(pred_hot.float() * pred_hot.eq(targets_hot).float(),
                                       dim=0)  # tgt_len,bsz, num_class
        step_class_total = torch.sum(targets_hot, dim=0).float()
        self.courage_items['class_correct'] = self.courage_items['class_correct'] + step_class_correct
        self.courage_items['class_total'] = self.courage_items['class_total'] + step_class_total

    def report_class_acc(self):
        """ report class avg acc during training """
        class_total = self.courage_items['class_total'].masked_fill(self.courage_items['class_total'] == 0, 1)
        if self.padding_idx is not None:
            self.courage_items['class_correct'][self.NULL_IDX] = 0.0
            class_avg = torch.sum(self.courage_items['class_correct'] / class_total).item() / (self.num_class - 1)
        else:
            class_avg = torch.sum(self.courage_items['class_correct'] / class_total).item() / self.num_class
        return class_avg




