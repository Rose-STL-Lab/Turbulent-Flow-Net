import torch


def charb(x, alpha, eps):
    '''charbonnier Loss function'''
    return torch.mean(torch.pow(x.pow(2) + eps, 1. / alpha))


def AAE(input_flow, target_flow):
    '''Average Angular Error:
    Provides a relative measure of performance
    that avoids the divide by zero.
    Calculates the angle between input and target vectors
    augmented with an extra dimension where the associated
    scalar value for that dimension is one.
    '''

    num = 1 + torch.sum(input_flow * target_flow, 1)
    denom = torch.sum(1 + input_flow ** 2, 1)
    denom_gt = torch.sum(1 + target_flow ** 2, 1)
    return torch.acos(num / torch.sqrt(denom * denom_gt)).mean()


class CharbonnierLoss(torch.nn.Module):
    '''From Back to Basics: 
    Unsupervised Learning of Optical Flow
    via Brightness Constancy and Motion Smoothness'''

    def __init__(self, alpha, eps):
        super(CharbonnierLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        return charb(input - target, self.alpha, self.eps)


class MagnitudeLoss(torch.nn.Module):

    def __init__(self, loss):
        super(MagnitudeLoss, self).__init__()
        self.loss = loss

    def forward(self, w):
        return self.loss(w, w.detach() * 0)


class SmoothnessLoss(torch.nn.Module):
    '''From Back to Basics: 
    Unsupervised Learning of Optical Flow
    via Brightness Constancy and Motion Smoothness'''

    def __init__(self, loss, delta=1):
        super(SmoothnessLoss, self).__init__()
        self.loss = loss
        self.delta = delta

    def forward(self, w):
        ldudx = self.loss((w[:, 0, 1:, :] - w[:, 0, :-1, :]) /
                          self.delta, w[:, 0, 1:, :].detach() * 0)
        ldudy = self.loss((w[:, 0, :, 1:] - w[:, 0, :, :-1]) /
                          self.delta, w[:, 0, :, 1:].detach() * 0)
        ldvdx = self.loss((w[:, 1, 1:, :] - w[:, 1, :-1, :]) /
                          self.delta, w[:, 1, 1:, :].detach() * 0)
        ldvdy = self.loss((w[:, 1, :, 1:] - w[:, 1, :, :-1]) /
                          self.delta, w[:, 1, :, 1:].detach() * 0)
        return ldudx + ldudy + ldvdx + ldvdy


class DivergenceLoss(torch.nn.Module):

    def __init__(self, loss, delta=1):
        super(DivergenceLoss, self).__init__()
        self.delta = delta
        self.loss = loss

    def forward(self, w):
        dudx = (w[:, 0, 1:] - w[:, 0, :-1]) / self.delta
        dvdy = (w[:, 1, 1:] - w[:, 1, :-1]) / self.delta
        return self.loss(dudx + dvdy, dudx.detach() * 0)


class WeightedSpatialMSELoss(torch.nn.Module):
    def __init__(self):
        super(WeightedSpatialMSELoss, self).__init__()
        self.loss = torch.nn.MSELoss(reduce=False, size_average=False)

    def forward(self, input, target, weights=1):
        return self.loss(input, target).mean(3).mean(2).mean(1) * weights
