import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from torch.nn.common_types import _size_any_t
from torch.nn.modules import utils

class AdaptiveLPPool2d(nn.Module):
    def __init__(self, norm_type: float, stride: Optional[_size_any_t] = None, ceil_mode: bool = False) -> None:
        super(AdaptiveLPPool2d, self).__init__()

        self.norm_type = norm_type
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        kernel_size = input.shape[2:4]
        return F.lp_pool2d(input, float(self.norm_type), kernel_size, self.stride, self.ceil_mode)

class AdaLearnLPP3(nn.Module):
    # AdaptiveLearnableLPPool2d
    def __init__(self, norm_type: float, pow_protection: str, stride: Optional[_size_any_t] = None, ceil_mode: bool = False, kernel_size: int = 0) -> None:
        super(AdaLearnLPP3, self).__init__()

        self.kernel_size = kernel_size

        self.norm_type = torch.nn.Parameter(torch.tensor(norm_type))
        self.stride = stride
        self.ceil_mode = ceil_mode

        if pow_protection == 'abs':
            self.pow_protection = torch.abs
        elif pow_protection == 'relu':
            self.pow_protection = F.relu
        elif pow_protection == 'posrelu':
            self.pow_protection = PosReLU()
        elif pow_protection == 'onerelu':
            self.pow_protection = PosReLU(offset=1.0)
        elif pow_protection == 'oneabs':
            self.pow_protection = lambda a : torch.abs(a-1)+1
        else:
            self.pow_protection = lambda a : a

    # Power function. Notes on pytorch-specific behavior:
    # New behavior (pytorch 1.7.0)
    # if base is negative and exp is real, result is nan
    # if base is zero and exp is negative, result is Inf, and gradient is -Inf
    # if base is zero and exp is >0 and <1, result is zero, but gradient is Inf
    #
    # base     | exponent | result   | gradient |
    #          |          |          |          |
    # > 0 real | >1 real  | ok       | ok       |
    # > 0 real | 1        | ok       | ok (1)   |
    # > 0 real | >0 & <1  | ok       | ok       |
    # > 0 real | 0        | ok (1)   | ok (0)   |
    # > 0 real | >-1 & <0 | ok       | ok       |
    # > 0 real | -1       | ok       | ok       |
    # > 0 real | <-1 real | ok       | ok       |
    #          |          |          |          |
    # 0        | >1 real  | ok (0)   | ok (0)   |
    # 0        | 1        | ok (0)   | ok (1)   |
    # 0        | >0 & <1  | ok (0)   | inf      | !!!
    # 0        | 0        | ok (1)   | ok (0)   |
    # 0        | >-1 & <0 | inf      | -inf     | !!!
    # 0        | -1       | inf      | -inf     | !!!
    # 0        | <-1 real | inf      | -inf     | !!!
    # 0        | <-1 int  | inf      | -inf     | !!!
    #          |          |          |          |
    # < 0 real | >1 int   | ok       | ok       |
    # < 0 real | >1 real  | nan      | nan      | !!!
    # < 0 real | 1        | ok       | ok (1)   |
    # < 0 real | >0 & <1  | nan      | nan      | !!!
    # < 0 real | 0        | ok (1)   | ok (0)   |
    # < 0 real | >-1 & <0 | nan      | nan      | !!!
    # < 0 real | -1       | ok       | ok       |
    # < 0 real | <-1 real | nan      | nan      | !!!
    # < 0 real | <-1 int  | ok       | ok       |
    #
    # Old behavior
    # if base is negative float, and exp is between -1 and 1, result is nan
    # if base is negative int, and exp is between -1 and 1, result is 1
    #
    # v1 = torch.tensor([0.], requires_grad=True); v2 = torch.pow(v1, -2.); print(v2.item()); v2.backward(); print(v1.grad.item())

    def forward(self, input: Tensor) -> Tensor:
        if self.kernel_size > 0:
            kernel_size = [self.kernel_size, self.kernel_size]
        else:
            kernel_size = input.shape[2:4]

        kw, kh = utils._pair(kernel_size)

        exp = self.pow_protection(self.norm_type)

        if self.stride is not None:
            out = F.avg_pool2d(torch.pow(input, exp), kernel_size, self.stride, 0, self.ceil_mode)
        else:
            out = F.avg_pool2d(torch.pow(input, exp), kernel_size, padding=0, ceil_mode=self.ceil_mode)

        return torch.pow((torch.sign(out) * F.relu(torch.abs(out))).mul(kw * kh), 1.0 / exp)


class AdaLearnLPP2(nn.Module):
    # AdaptiveLearnableLPPool2d
    def __init__(self, norm_type: float, pow_func: str, stride: Optional[_size_any_t] = None, ceil_mode: bool = False, kernel_size: int = 0) -> None:
        super(AdaLearnLPP2, self).__init__()

        self.kernel_size = kernel_size

        self.norm_type = torch.nn.Parameter(torch.tensor(norm_type))
        self.stride = stride
        self.ceil_mode = ceil_mode

        self.pow = getattr(self, pow_func)

    def spow(self, base, exp):
        # Simple power
        output = base**exp
        return output

    def cpow(self, base, exp):
        # Constrained power (exp >= 0)
        output = base**torch.abs(exp)
        return output

    def gt1pow(self, base, exp):
        # Constrained power (exp >= 1)
        output = base**(1+torch.abs(exp-1))
        return output

    def relu1pow(self, base, exp):
        # Constrained power (exp >= 1) using posrelu
        output = base**(1+F.relu(exp-1))
        return output

    def ppow(self, base, exp):
        # Protected power (handle negative base)

        isneg = base < 0
        base[isneg] = -base[isneg] # prevents having negative values as a base

        output = base**exp
        output[isneg] = -output[isneg]
        return output

    def ppow2(self, base, exp):
        # Protected power 2

        isneg = base < 0
        base[isneg] = -base[isneg] # prevents having negative values as a base
        base = base + .000000001 # prevents having 0 as a base

        # # Broadcasting avoids in-place power function (which is not differentiable)
        # tot_ch = base.shape[1]
        # exp = torch.nn.functional.pad(exp[None], ((ch, tot_ch-ch-1)), mode='constant', value=1)[None,:,None,None]

        output = base**exp
        output[isneg] = -output[isneg]
        return output

    def forward(self, input: Tensor) -> Tensor:
        if self.kernel_size > 0:
            kernel_size = [self.kernel_size, self.kernel_size]
        else:
            kernel_size = input.shape[2:4]

        kw, kh = utils._pair(kernel_size)

        if self.stride is not None:
            out = F.avg_pool2d(self.pow(input, self.norm_type), kernel_size, self.stride, 0, self.ceil_mode)
        else:
            out = F.avg_pool2d(self.pow(input, self.norm_type), kernel_size, padding=0, ceil_mode=self.ceil_mode)

        return self.pow((torch.sign(out) * F.relu(torch.abs(out))).mul(kw * kh), 1.0 / self.norm_type)

# Positive ReLU
class PosReLU(nn.Module):
    def __init__(self, offset: float = 0.0000001) -> None:
        super(PosReLU, self).__init__()
        self.offset = offset

    def forward(self, input: Tensor) -> Tensor:
        out = F.relu(input-self.offset)+self.offset
        return out

# NormRGB
class NormRGB(nn.Module):
    def __init__(self) -> None:
        super(NormRGB, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        # input.shape >> torch.Size([32, 3, 1, 1])
        som = input.pow(2).sum(axis=1, keepdims=True).sqrt()
        return input/som

class Power(nn.Module):
    # Power
    def __init__(self, exponent: float, pow_func: str) -> None:
        super(Power, self).__init__()

        self.exponent = torch.nn.Parameter(torch.tensor(exponent))
        self.pow = getattr(self, pow_func)

    def spow(self, base, exp):
        # Simple power
        output = base**exp
        return output

    def cpow(self, base, exp):
        # Constrained power (exp >= 0)
        output = base**torch.abs(exp)
        return output

    def ppow(self, base, exp):
        # Protected power (handle negative base)

        isneg = base < 0
        base[isneg] = -base[isneg] # prevents having negative values as a base

        output = base**exp
        output[isneg] = -output[isneg]
        return output

    def ppow2(self, base, exp):
        # Protected power 2

        isneg = base < 0
        base[isneg] = -base[isneg] # prevents having negative values as a base
        base = base + .000000001 # prevents having 0 as a base

        # # Broadcasting avoids in-place power function (which is not differentiable)
        # tot_ch = base.shape[1]
        # exp = torch.nn.functional.pad(exp[None], ((ch, tot_ch-ch-1)), mode='constant', value=1)[None,:,None,None]

        output = base**exp
        output[isneg] = -output[isneg]
        return output

    # Power function. Notes on pytorch-specific behavior:
    # New behavior
    # if base is negative and exp is fractional, result is nan
    # if base is zero and exp is negative, result is inf
    # Old behavior
    # if base is negative float, and exp is between -1 and 1, result is nan
    # if base is negative int, and exp is between -1 and 1, result is 1

    def forward(self, input: Tensor) -> Tensor:
        out = self.pow(input, self.exponent)
        return out
