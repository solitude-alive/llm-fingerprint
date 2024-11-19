import torch
from . import logger
from .matrix_torch import solve_equation


def _logit_add_bias(x, bias, indices):
    """
    Add bias to the logits of the selected indices.

    Parameters:
        x (torch.Tensor): the logits
        bias (torch.Tensor): the bias to add
        indices (list): the indices of the logits to add bias
    """
    x = x.clone()
    for index in indices:
        if isinstance(index, torch.Tensor):
            index = index.item()
        x[index] += bias
    return x


def _recover_prob_from_top_k_o(p_bias, bias, p_ref_bias, p_ref):
    """
    Recover the probability from the top k probability with bias.
    Note: the reference logit should be the same as the original logit.
    When p_bias.size(0) == 1, the residual is small.
    When p_bias.size(0) > 1, the residual is much larger than the original.
    Note:
    This function is not used in the current implementation !!!

    Parameters:
        p_bias (torch.Tensor): the top k values
        bias (torch.Tensor): the bias to add
        p_ref_bias (torch.Tensor): the reference probability after adding bias
        p_ref (torch.Tensor): the reference probability without adding bias
    return:
        prob_recover (torch.Tensor): the recovered probability
    """
    assert 0 <= torch.min(p_bias) and torch.max(p_bias) <= 1, "Invalid p_bias, should be in [0, 1]."
    prob_recover = torch.exp(torch.log(p_bias) - bias - torch.log(p_ref_bias) + torch.log(p_ref))
    return prob_recover


def _recover_prob_from_top_k(p_bias, p_ref_bias, p_ref):
    """
    Recover the probability from the top k probability with bias.
    Note: the reference logit should be added bias.

    Parameters:
        p_bias (torch.Tensor): the top k values
        p_ref_bias (torch.Tensor): the reference probability after adding bias
        p_ref (torch.Tensor): the reference probability without adding bias
    return:
        prob_recover (torch.Tensor): the recovered probability
    """
    assert 0 <= torch.min(p_bias) and torch.max(p_bias) <= 1, "Invalid p_bias, should be in [0, 1]."
    prob_recover = p_bias * p_ref / p_ref_bias
    return prob_recover


def _recover_prob_from_top_1(p_bias, bias, log_softmax=False):
    """
    Recover the logit from the top 1 probability with bias.

    Args:
        p_bias (torch.Tensor): the top 1 probability with bias.
        bias (torch.Tensor): the bias to add.
        log_softmax (bool): whether the input is log_softmax.

    Returns:
        prob_recover (torch.Tensor): the recovered logit.
    """
    if log_softmax:
        log_p = p_bias
    else:
        log_p = torch.log(p_bias)   # log(p_bias)
    prob_recover = 1 / (torch.exp(bias - log_p) - torch.exp(bias) + 1)
    return prob_recover


@torch.no_grad()
def recover_logits(x, replace_inf=False):
    """
    Recover the logits from the probability by Center log ratio transform.
        clr(x_i) =  log((x_i)/g(x)), where g(x) = (x_1 * x_2 * ... * x_n)^(1/n)
        For the numerical stability, we return log(x_i) directly as the g(x) is constant.

    Parameters:
        x (torch.Tensor): the probability
        replace_inf (bool): whether replace the inf value with the minimum abs value of the float
    """
    assert isinstance(x, torch.Tensor)
    assert x.size(1) == 1
    x = x.view(-1)
    log_x = torch.log(x)
    if replace_inf:
        min_log = torch.log(torch.tensor(torch.finfo(torch.float).tiny))
        log_x = torch.where(torch.isinf(log_x), min_log, log_x)
    # mean_log_x = torch.mean(log_x, dim=-1, keepdim=True)
    # return (log_x - mean_log_x).view(-1, 1)
    return log_x.view(-1, 1)


@torch.no_grad()
def recover_from_top_k(x, k):
    """
    Recover the tensor from the top k values and indices.

    Parameters:
        x (torch.Tensor): the logits
        k (int): the number of top k probability to recover
    return:
        p_recover (torch.Tensor): the recovered probability
    """
    assert 0 < k < x.size(0), f"Invalid k: {k}, should be in (0, {x.size(0)})"
    assert x.size(1) == 1

    x = x.view(-1)
    p_recover = torch.zeros_like(x)

    index_sorted = torch.argsort(x, descending=True)
    top_index = index_sorted[0].item()

    ref_prob = torch.softmax(x, dim=0)[top_index]
    p_recover[index_sorted[0]] = ref_prob

    prob = torch.softmax(x, dim=0)

    for i in range(1, x.size(0), k-1):
        lower_bound = i
        upper_bound = min(i + k-1, x.size(0))     # the upper bound of the current group, to avoid out of index

        # get the top k values and indices
        indices = list(index_sorted[lower_bound: upper_bound])

        bias = x[top_index] - x[indices[-1]] + 2.0  # update the bias

        indices.append(torch.tensor(top_index))  # add the top index to the indices

        logit_bias = _logit_add_bias(x, bias, indices)
        prob_bias = torch.softmax(logit_bias, dim=0)
        top_k_values, top_k_indices = torch.topk(prob_bias, k)

        p_ref = ref_prob
        p_ref_bias = top_k_values[0]
        p_bias = top_k_values[1:]

        prob_recover = _recover_prob_from_top_k(p_bias, p_ref_bias, p_ref)

        if len(indices) == 2:
            p_recover[indices[0]] = prob_recover
            continue

        for j in range(len(indices) - 1):
            index = indices[j].item()
            p_recover[index] = prob_recover[j]
            assert torch.allclose(prob_recover[j], prob[index])

    return p_recover.view(-1, 1)


@torch.no_grad()
def recover_from_top_1(x):
    """
    Recover the tensor from the top 1 value.

    Args:
        x (torch.Tensor): the logits

    Returns:
        p_recover (torch.Tensor): the recovered probability

    """
    assert x.size(1) == 1

    x = x.view(-1)
    p_recover = torch.zeros_like(x)
    index_sorted = torch.argsort(x, descending=True)
    top_index = index_sorted[0].item()
    prob = torch.softmax(x, dim=0)

    for i in range(x.size(0)):
        index = index_sorted[i].item()

        bias = x[top_index] - x[index] + 2.0

        logit_bias = _logit_add_bias(x, bias, [index])

        # prob_bias = torch.softmax(logit_bias, dim=0)
        # log_softmax = False

        prob_bias = torch.log_softmax(logit_bias, dim=0)
        log_softmax = True

        prob_recover = _recover_prob_from_top_1(prob_bias[index], bias, log_softmax)
        p_recover[index] = prob_recover
        if torch.isnan(prob_recover):   # break if the probability is nan
            break
        assert torch.allclose(prob_recover, prob[index])

    return p_recover.view(-1, 1)


def _process_logit(w, logit, name=None):
    """
    Process the logit and calculate the residual

    Args:
        w (torch.Tensor): the weight matrix
        logit (torch.Tensor): the logit from the model output
        name (str): logit name from ["Oracle", "softmax", "top-k"]

    Returns:
        residual (float): the residual about the equation of "w * x = logit"
    """
    assert isinstance(logit, torch.Tensor)
    assert logit.size(1) == 1
    residual = solve_equation(w, logit)
    residual = residual.to("cpu").item()
    logger.log(f"Solve equation for {name} successfully, residual: {residual}")
    return residual


def _linear_utils(weight_aug, logit_recover, name, threshold=1):
    res = _process_logit(weight_aug, logit_recover, name=f"{name} with augmentation")
    flag = False  # whether the augmentation is valid
    if res > 1 * threshold:
        weight_aug = torch.cat([weight_aug, logit_recover], dim=1)
    elif res < 1 * threshold:
        logger.log(f"= Size of weight_{name} is {weight_aug.size()}")
    else:
        logger.log(f"Residual is {res}, skip the augmentation.")
        flag = True
    return weight_aug, flag

# ===========================================
# =================== API ===================
# ===========================================


def process_logits(w, logits, *, type_logits: list = None):
    """
    Process the logits and calculate the residuals and rank

    Parameters:
        w (torch.Tensor): the weight matrix
        logits (tuple): the logits from the model output
        type_logits (list): the type of logits calculate for ["all", "softmax", "top-k", "top-1"]
    Returns:
        res (float): the average residual about the equation of "w * x = logit"
        rank (int): the rank of the weight matrix
    """
    assert isinstance(logits, tuple)

    if type_logits is None:
        type_logits = ["softmax"]
    type_list = ["all", "softmax", "top-k", "top-1"]
    assert set(type_logits).issubset(set(type_list)), \
        f"Type of logits should be in {type_list}, but got {type_logits}."

    # convert type_logits to flag
    flag_softmax = "softmax" in type_logits
    flag_top_k = "top-k" in type_logits
    flag_top1 = "top-1" in type_logits
    if "all" in type_logits:
        flag_softmax = flag_top_k = flag_top1 = True

    w_aug = torch.cat([w, torch.ones((w.size(0), 1)).to(w.device)], dim=1).detach()

    w_aug_prob = w.detach()
    w_aug_topk = w.detach()
    w_aug_top = w.detach()

    residuals = []
    residuals_softmax = []
    residuals_top_k = []
    residuals_top1 = []

    for logit in logits:
        logit = logit.view(-1, 1)

        # for oracle logit
        residual = _process_logit(w, logit, name="Oracle")
        residuals.append(residual)

        if flag_softmax:
            # for softmax
            prob = torch.softmax(logit, dim=0)
            logit_softmax_recover = recover_logits(prob)
            if torch.isinf(logit_softmax_recover).any():
                continue
            w_aug_prob, flag = _linear_utils(w_aug_prob, logit_softmax_recover, "soft", 1)
            if flag:
                continue

            residual_softmax = _process_logit(w_aug, logit_softmax_recover, name="softmax")
            residuals_softmax.append(residual_softmax)

        if flag_top_k:
            # for top-k
            prob_top_k = recover_from_top_k(logit, 5)
            logit_top_k_recover = recover_logits(prob_top_k)
            if torch.isinf(logit_top_k_recover).any():
                continue

            w_aug_topk, flag = _linear_utils(w_aug_topk, logit_top_k_recover, "topk", 1)
            if flag:
                continue

            residual_top_k = _process_logit(w_aug, logit_top_k_recover, name="top-k")
            residuals_top_k.append(residual_top_k)

        if flag_top1:
            # for top-1
            prob_top1_recover = recover_from_top_1(logit)
            logit_top1_recover = recover_logits(prob_top1_recover)
            if torch.isinf(logit_top1_recover).any() or torch.isnan(logit_top1_recover).any():
                continue

            w_aug_top, flag = _linear_utils(w_aug_top, logit_top1_recover, "top", 1)
            if flag:
                continue

            residual_top1 = _process_logit(w_aug, logit_top1_recover, name="top-1")
            residuals_top1.append(residual_top1)

    res = sum(residuals) / len(residuals)

    if flag_softmax:
        res_softmax = sum(residuals_softmax) / len(residuals_softmax)
        logger.log(f"=for softmax: {res_softmax} in {len(residuals_softmax)} samples.\n"
                   f"==Rank of the softmax: {w_aug_prob.size(1)}==")
    if flag_top_k:
        res_top_k = sum(residuals_top_k) / len(residuals_top_k)
        logger.log(f"=for top-k: {res_top_k} in {len(residuals_top_k)} samples.\n"
                   f"==Rank of the top-k: {w_aug_topk.size(1)}==")
    if flag_top_k:
        res_top1 = sum(residuals_top1) / len(residuals_top1)
        logger.log(f"=for top-1: {res_top1} in {len(residuals_top1)} samples.\n"
                   f"==Rank of the top-1: {w_aug_top.size(1)}==")

    rank = w_aug_prob.size(1)

    return res, rank
