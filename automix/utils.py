import torch


def count_parameters(model, trainable_only=True):

    if trainable_only:
        if len(list(model.parameters())) > 0:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            params = 0
    else:
        if len(list(model.parameters())) > 0:
            params = sum(p.numel() for p in model.parameters())
        else:
            params = 0

    return params


def center_crop(x: torch.Tensor, length: int):
    start = (x.shape[-1] - length) // 2
    stop = start + length
    return x[..., start:stop]


def causal_crop(x: torch.Tensor, length: int):
    stop = x.shape[-1] - 1
    start = stop - length
    return x[..., start:stop]


def restore_from_0to1(x: torch.Tensor, min_val: float, max_val: float):
    """Restore tensor back to the original range assuming they have been normalized on (0,1)

    Args:
        x (torch.Tensor): Tensor with normalized values on (0,1).
        min_val (float): Minimum value in the original range.
        max_val (float): Maximum value in the original range.

    Returns:
        y (torch.Tensor): Tensor with denormalized values on (min_val, max_val).
    """
    return (x * (max_val - min_val)) + min_val


def scale_to_0to1(x: torch.Tensor, min_val: float, max_val: float):
    """Scale tensor to 0 to 1 given the supplied min-max range.

    Args:
        x (torch.Tensor): Tensor with normalized values on (0,1).
        min_val (float): Minimum value in the original range.
        max_val (float): Maximum value in the original range.

    Returns:
        y (torch.Tensor): Tensor with normalized values on (0, 1).
    """
    return (x - min_val) / (max_val - min_val)
