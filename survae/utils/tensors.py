import torch
import torchvision.utils as vutils


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def mean_except_batch(x, num_dims=1):
    '''
    Averages all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_mean: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).mean(-1)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims=2):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)
    

def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def checkerboard_split(x, flip=False):
    b, c, h, w = x.shape
    mask = checkerboard_mask(b, c, h, w, x.device, flip=flip)        
    x1 = torch.masked_select(x, mask).view(b, c, h, w // 2)
    x2 = torch.masked_select(x, mask == False).view(b, c, h, w // 2)
    return x1, x2


def checkerboard_mask(b, c, h, w, device, flip=False):
    x, y = torch.arange(h, dtype=torch.int32, device=device), torch.arange(w, dtype=torch.int32, device=device)
    xx, yy = torch.meshgrid(x, y)
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.int32).view(1, 1, h, w) > 0
    if flip:
        mask = mask == False

    return mask


def checkerboard_inverse(z1, z2, flip=False):
    b, c, h, w = z1.shape
    mask = checkerboard_mask(b, c, h, w * 2, z1.device, flip=flip).to(z1.dtype)
    x = torch.repeat_interleave(z1, 2, dim=3) * mask + \
        torch.repeat_interleave(z2, 2, dim=3) * (1.0 - mask)
    return x
