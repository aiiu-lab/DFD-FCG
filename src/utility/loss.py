import torch


def focal_loss(input, target, gamma=2, weight=None):
    # Code based on: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    assert len(input.shape) == 2
    assert len(target.shape) == 1
    # input.shape = N,C
    # target.shape = N
    target = target.unsqueeze(1)

    logpt = torch.log_softmax(input, dim=-1)
    logpt = torch.clamp(logpt, max=-0.01, min=-5)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = torch.tensor(logpt.data.exp())

    if (not weight is None):
        # target.shape = N, 1
        weight = torch.tensor(
            weight,
            device=input.device,
            dtype=input.dtype
        )
        at = weight.gather(0, target.data.view(-1))
    else:
        at = 1

    logpt = logpt * at

    loss = -1 * (1 - pt)**gamma * logpt

    return loss
