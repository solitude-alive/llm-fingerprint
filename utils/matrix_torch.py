# This file is used for the matrix calculation, using torch

import torch


@torch.no_grad()
def get_rank(singular_value):
    candidate = torch.zeros(singular_value.size(0)-1)
    for i in range(singular_value.size(0) - 1):
        candidate[i] = torch.log(singular_value[i] / singular_value[i+1])
    return torch.argmax(candidate) + 1


@torch.no_grad()
def get_singular_augmented_matrix(a, b, descending=True):
    u, s, v = torch.svd(torch.cat((a.cpu(), b.cpu()), dim=1), compute_uv=False)
    singular = torch.sort(s, descending=descending).values
    return singular


@torch.no_grad()
def calculate_rank(weight, logits):
    assert isinstance(weight, torch.Tensor), "Weight need to be a torch.Tensor."

    if isinstance(logits, tuple):
        b = torch.cat(logits, dim=0).permute(1, 0)  # convert to tensor
    else:
        b = logits
    assert isinstance(b, torch.Tensor), f"Expect b is a torch.Tensor, but got {type(b)}."

    singular = get_singular_augmented_matrix(weight, b)

    rank = get_rank(singular)

    return rank


@torch.no_grad()
def solve_equation(a, b):
    """
    Solve equation a @ x = b

    Parameters:
        a (torch.Tensor): matrix a
        b (torch.Tensor): matrix b

    return:
        residual: || b - a @ b^{hat} ||
    """
    assert isinstance(a, torch.Tensor), f"Expect a is a torch.Tensor, but got {type(a)}."
    assert isinstance(b, torch.Tensor), f"Expect b is a torch.Tensor, but got {type(b)}."
    solution, residual, rank, _ = torch.linalg.lstsq(a, b, driver="gels")
    del solution, rank, _
    torch.cuda.empty_cache()
    return residual
