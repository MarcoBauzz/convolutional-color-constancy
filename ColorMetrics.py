from __future__ import print_function
import numpy as np
import torch
import math

def recovery_error(estimated_illuminant, expected_illuminant, angle_type='degrees', reduction='mean'):
    num = torch.sum(expected_illuminant*estimated_illuminant, axis=1)
    den1 = torch.sqrt(torch.sum(expected_illuminant**2, axis=1))
    den2 = torch.sqrt(torch.sum(estimated_illuminant**2, axis=1))
    ang = torch.acos(torch.clamp(num/(den1*den2), -1.0, 1.0))

    if reduction == 'mean':
        ang = torch.mean(ang)
    else:
        ang = ang

    err = 180*ang/math.pi

    return err

def reproduction_error(estimated_illuminant, expected_illuminant, angle_type='degrees', reduction='mean'):
    num = torch.sum((expected_illuminant/estimated_illuminant)*1.0, axis=1)
    den = torch.sqrt(torch.sum((expected_illuminant/estimated_illuminant)**2, axis=1))*torch.sqrt(3)
    ang = torch.acos(torch.clamp(num/den, -1.0, 1.0))

    if reduction == 'mean':
        ang = torch.mean(ang)
    else:
        ang = ang

    err = 180*ang/math.pi

    return err


class RecoveryLoss(torch.nn.Module):
    """ RecoveryLoss
    
    Args:
         []
    """

    def __init__(self, angle_type='degrees', reduction='mean'):
        super(RecoveryLoss, self).__init__()
        self.reduction = reduction

        if angle_type == 'radians':
            self.multiplier = 1.0
        else:
            self.multiplier = 180.0/math.pi

    def forward(self, estimated_illuminant, expected_illuminant):
        # import ipdb; ipdb.set_trace()
        num = torch.sum(expected_illuminant*estimated_illuminant, axis=1)
        den1 = torch.sqrt(torch.sum(expected_illuminant**2, axis=1))
        den2 = torch.sqrt(torch.sum(estimated_illuminant**2, axis=1))
        ang = torch.acos(torch.clamp(num/(den1*den2), -.9999999, .9999999))
        # Note: in -1 and +1 the derivative of acos goes to infinite: d/dx (acos(x)) = -1/sqrt(1-x^2)
        # import torch
        # x = torch.tensor([-1.5, -1.0, -.9999999, -0.5, 0.0, 0.5, .9999999, 1.0, 1.5], requires_grad=True)
        # y = torch.acos(x)
        # z = y.mean()
        # z.backward()
        # x.grad

        if self.reduction == 'mean':
            mea = torch.mean(ang)
        else:
            mea = ang

        mea = mea*self.multiplier

        return mea


if __name__ == "__main__":
    exp = torch.randn(1,3)
    # est = torch.randn(1,3)
    est = exp+.2#.00001#+.2
    # est = exp+2

    print("Estimated:")
    print(est)

    print("Expected:")
    print(exp)

    print("Recovery angular error:")
    print(recovery_error(est, exp))

    nest = est / torch.sqrt(torch.sum(est**2))
    nexp = exp / torch.sqrt(torch.sum(exp**2))
    print("Recovery angular error (normalized vectors):")
    print(recovery_error(nest, nexp))
