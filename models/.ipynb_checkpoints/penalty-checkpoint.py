import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DivergenceLoss(torch.nn.Module):
    def __init__(self, loss_fun):
        """
        Divergence Loss constructor.

        Args:
            loss_fun (torch.nn.Module): PyTorch loss function.
        """
        super(DivergenceLoss, self).__init__()
        self.loss = loss_fun

    def forward(self, preds):
        """
        Forward pass of the Divergence Loss module.

        Args:
            preds (torch.Tensor): Predicted tensor of shape (bs, 2, H, W).

        Returns:
            torch.Tensor: Loss value.
        """
        u = preds[:, :1]
        v = preds[:, -1:]
        u_x = field_grad(u, 0)
        v_y = field_grad(v, 1)
        div = v_y + u_x
        return self.loss(div, div.detach() * 0)

def field_grad(f, dim):
    """
    Compute the gradient of a tensor along a specified dimension.

    Args:
        f (torch.Tensor): Tensor with the last two dimensions as spatial dimensions.
        dim (int): Dimension along which to compute the gradient.

    Returns:
        torch.Tensor: Gradient tensor.
    """
    dx = 1
    dim += 1
    N = len(f.shape)
    out = torch.zeros(f.shape).to(device)
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    # 2nd order interior
    slice1[-dim] = slice(1, -1)
    slice2[-dim] = slice(None, -2)
    slice3[-dim] = slice(1, -1)
    slice4[-dim] = slice(2, None)
    out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2 * dx)

    # 2nd order edges
    slice1[-dim] = 0
    slice2[-dim] = 0
    slice3[-dim] = 1
    slice4[-dim] = 2
    a = -1.5 / dx
    b = 2. / dx
    c = -0.5 / dx
    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    slice1[-dim] = -1
    slice2[-dim] = -3
    slice3[-dim] = -2
    slice4[-dim] = -1
    a = 0.5 / dx
    b = -2. / dx
    c = 1.5 / dx

    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    return out
