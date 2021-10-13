import torch
from torch.nn import functional as F
from .focal_loss import sigmoid_focal_loss


def sigmoid_mixed_loss(
    inputs: torch.Tensor,  # bsz, num_classes
    targets: torch.Tensor,
    alpha: float = -1,  # alpha = 0.5
    gamma: float = 2,
    reduction: str = "none",  # "sum"
    beta: float = 1.0,  # 1.0 means only  to encourage  foreground classification, 0.0 means
) -> torch.Tensor:
    """
    sigmoid mixed loss which use focal loss in low likelihood area and cross entropy in high likelihood area.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        beta:
    Returns:
        Loss tensor with the reduction option applied.
    """
    if alpha >= 0:  # alpha = 0.5
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    else:
        alpha_t = torch.ones_like(targets)
    pd = torch.zeros_like(inputs)  # 因为后面还有sigmoid激活函数所以这里是以0为界
    pd_fl = sigmoid_focal_loss(inputs=pd, targets=targets, alpha=alpha, gamma=gamma, reduction='none')
    pd_ce = F.binary_cross_entropy_with_logits(pd, targets, reduction='none')
    pd_ce = pd_ce * alpha_t
    bias = pd_ce - pd_fl
    if beta == 1:
        loss = torch.where((inputs > 0) * (targets == 1),  # tail +high
                           F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * alpha_t - bias,
                           sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction='none'),
                           )
    elif beta == 2:  # HFL-head FL-tail compared with FL to illustate  head+high
        loss = torch.where((inputs > 0) * (targets == 0),
                           F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * alpha_t - bias,
                           sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction='none'),
                           )
    elif beta == 3:  # CE-head HFL-tail 和CE 对比说明 low rare
        loss = torch.where((inputs > 0),
                           F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * alpha_t - bias,
                           sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction='none'),
                           )
        loss = torch.where(targets == 1,
                           loss,
                           F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')* alpha_t,)
    elif beta == 4:  # CE-tail HFL-head 和CE 对比说明 low rare
        loss = torch.where((inputs > 0),
                           F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * alpha_t - bias,
                           sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction='none'),
                           )
        loss = torch.where(targets == 0,
                           loss,
                           F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')* alpha_t,)
    else:  # beta==0.0  # 才发现当时写的时候是记得的，只是之后忘了，反正mixed loss 的beta是正常的
        loss = torch.where(inputs <= 0,  # high
                           sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction='none'),
                           F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') * alpha_t - bias)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


sigmoid_mixed_loss_jit = torch.jit.script(
    sigmoid_mixed_loss
)  # type: torch.jit.ScriptModule

def sigmoid_encourage_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,  # alpha = 0.5
    gamma: float = 2,
    reduction: str = "none",  # "sum"
    base_loss: str = 'mle',
    add_loss: str = 'ell',
    log_end: float = 1.0,
    power: float = 2.0,
    beta: float = 1.0,  # 1.0 means only  to encourage  foreground classification # 2020/9/26 21:44 发现这里beta0只是另一个极端，beta0.5 才是都鼓励
    ita: float = 0.0,  # gamma for bonus
    el_start:int = 1,   # 0908这里计算的cur step是和log一致的
    cur_step:int = 1,
) -> torch.Tensor:
    """
    sigmoid mixed loss which use focal loss in low likelihood area and cross entropy in high likelihood area.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        base_loss:
        add_loss:
        log_end:
        power:
        beta:
        ita:
        el_start:
        cur_step:
    Returns:
        Loss tensor with the reduction option applied.
    """
    if alpha >= 0:  # alpha = 0.5
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    else:
        alpha_t = torch.ones_like(targets)
    if base_loss == 'mle':
        mle_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        base = mle_loss
    elif base_loss == 'mle_alpha':  # 0829 添加 mle_alpha
        # 论文里alpha 对cross entropy 是 0.75 最好
        mle_alpha_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')*alpha_t
        base = mle_alpha_loss
    else:  # fl
        focal_loss = sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma, reduction='none')
        base = focal_loss
    if add_loss == 'zero' or (cur_step < el_start):
        add = torch.zeros_like(targets)
    elif add_loss == 'elp':  # add is power bonus
        p = torch.sigmoid(inputs)
        # elp_additional_loss = F.nll_loss(pred_sigmoid.pow(power), label, reduction='none') * label * alpha + \
        #                       F.nll_loss((1 - pred_sigmoid).pow(power), 1-label, reduction='none') * (1 - label) * (
        #                                   1 - alpha)
        elp_additional_loss = -p.pow(power) * targets * (targets * beta) - \
                              (1 - p).pow(power)*(1-targets) * ((1 - targets) * (1 - beta))
        add = elp_additional_loss

    elif add_loss == 'ell_regular':
        # one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        ell_additional_loss = - F.binary_cross_entropy(1-p, targets, reduction='none')  # ylog(1-p)+ (1-y)logp
        if log_end != 1.0:         # log end 应当是个vector ，对于target是1 的时候，是LE, 否则1-LE
            LE = log_end*targets+(1-log_end)*(1-targets)
            y_log_end = - F.binary_cross_entropy(1-LE, targets, reduction='none')  # ylog(1-log_end)+ (1-y)log(1-log_end) =log(1-log_end)
            bonus_after_log_end = 1/(log_end-1) * (p_t - log_end) + y_log_end
            ell_additional_loss = torch.where(p_t > log_end, bonus_after_log_end, ell_additional_loss)
        ell_additional_loss = ell_additional_loss * ((1 - beta) * (1 - targets) + beta * targets)  # decide which class receive more bonus
        ell_additional_loss = ell_additional_loss * ((1 - p_t) ** ita)
        # 加个 ce的对称可能容易点，因为 (1-p)^2
        add = ell_additional_loss
    else:  # ell  # add is log bonus
        # one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)
        p = torch.sigmoid(inputs)
        # 2021.6.1 15：07 发现之前ell 有一个bug,  原来应该事 1-sigmoid(x), 现在是 sigmoid (1-sigmoidx)
        # sigmoid (1-p) 的值域在0.5-0.75 是非常陡的一段区域
        # 1-p的值域是0到1， 完整的区域
        ell_additional_loss = - F.binary_cross_entropy_with_logits(1-p, targets, reduction='none') * (
                (1 - beta) * (1 - targets) + beta * targets)
        add = ell_additional_loss
    loss = base + add
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

sigmoid_encourage_loss_jit = torch.jit.script(
    sigmoid_encourage_loss
)  # type: torch.jit.ScriptModule


def copy_encourage_args(cfg, args):
    # loss_clsz, base_loss, add_loss, power, beta,alpha,gamma
    cfg.loss_clsz = args.loss_clsz
    cfg.base_loss = args.base_loss
    cfg.add_loss = args.add_loss
    cfg.log_end = args.log_end
    cfg.power = args.power
    cfg.beta = args.beta
    cfg.ita= args.ita
    cfg.el_start = args.el_start
    # cfg.alpha = args.alpha
    # cfg.gamma = args.gamma
    return cfg