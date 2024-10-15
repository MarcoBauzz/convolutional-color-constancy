import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as FF

import math
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import os
import argparse

from torch import Tensor
from PIL import Image

from ColorMetrics import RecoveryLoss, recovery_error
from ColorDatasets import ColorCheckerDataset
from ColorModules import *

import warnings

# import random
# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch Convolutional Edge Based Color Constancy')
parser.add_argument('--dir-train', type=str, default='/data/Datasets/ColorConstancy/ColorChecker/Hemrit/masked_long800/', metavar='D',
                    help='training directory')
parser.add_argument('--dir-exp', type=str, default='./Experiments/', metavar='D',
                    help='saving directory')
parser.add_argument('--name-exp', type=str, default='expTMP', metavar='N',
                    help='experiment name (e.g. exp001)')
parser.add_argument('--dir-gt', type=str, default='/data/Datasets/ColorConstancy/ColorChecker/GT/', metavar='D',
                    help='gt directory')
parser.add_argument('--gt', type=str, default='GT_HemritRec.txt', metavar='GT',
                    help='gt file')
parser.add_argument('--mode', type=str, default='train', metavar='M',
                    help='train')
parser.add_argument('--device', type=str, default='cuda', metavar='D',
                    help='device (cuda or cpu. default: cuda)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--image-size', type=int, default=800, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--starting-epoch', type=str, default=0, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.01 for degree)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--wdecay', type=float, default=0.0, metavar='W',
                    help='weight decay (default: 0.0)')
parser.add_argument('--W', type=int, default=3, metavar='W',
                    help='filtersize (default 3)')
parser.add_argument('--W-last', type=int, default=1, metavar='W',
                    help='filtersize for the last convolutional layer (default 1)')
parser.add_argument('--dilation', type=int, default=1, metavar='D',
                    help='dilation for inner conv only (default 1)')
parser.add_argument('--njet', type=int, default=1, metavar='N',
                    help='derivative order (default 1)')
parser.add_argument('--nl', type=str, default='prelu', metavar='F',
                    help='non-linearities for internal convs (''relu'', ''prelu'', ''posrelu'', ''abs'', ''none'')')
parser.add_argument('--inner-size', type=int, default=9, metavar='S',
                    help='size of the inner convolution (default 9)')
parser.add_argument('--intermediate-blocks', type=int, default=1, metavar='N',
                    help='number of intermediate filter blocks (0 or 1)')
parser.add_argument('--njet-protection1', type=str, default='prelu', metavar='P',
                    help='protection used before computing pow (''relu'', ''prelu'', ''posrelu'', ''abs'', ''none'')')
parser.add_argument('--njet-protection2', type=str, default='posrelu', metavar='P',
                    help='protection used before computing ^1/pow (''relu'', ''prelu'', ''posrelu'', ''abs'', ''none'')')
parser.add_argument('--mink-norm', type=float, default=1.000001, metavar='P',
                    help='minkowski norm (P) (default 1. Use 0 to skip, use -1 for max pooling)')
parser.add_argument('--mink-protection', type=str, default='abs', metavar='P',
                    help='protection used before computing the LPpool (''relu'', ''prelu'', ''posrelu'', ''abs'', ''none'')')
parser.add_argument('--mink-protection-pow', type=str, default='oneabs', metavar='P',
                    help='protection used inside the LPpool for the exponent (''relu'', ''posrelu'', ''onerelu'', ''oneabs'', ''abs'', ''none'')')
# parser.add_argument('--mink-pow-func', type=str, default='relu1pow', metavar='F',
                    # help='power function used for the LPpool (''spow'', ''cpow'', ''ppow'', ''ppow2'', ''gt1pow'', ''relu1pow'')')
parser.add_argument('--sigma', type=float, default=-1.0, metavar='S',
                    help='sigma (default: 0.0)')
parser.add_argument('--checkpoint-interval', type=int, default=25, metavar='N',
                    help='how many')
parser.add_argument('--data-augmentation', default=False, action='store_true')
parser.add_argument('--validation-transform', type=str, default='isotropic', metavar='T',
                    help='rescaling operation used on validation and test data. relies on --image-size (''none'', ''isotropic'', ''anisotropic'')')
# parser.add_argument('--unaltered-test', default=False, action='store_true')
parser.add_argument('--save-init', default=False, action='store_true')
parser.add_argument('--save-output', default=False, action='store_true',
                    help='save prediction (and error) output')
parser.add_argument('--pool-size', type=int, default=1, metavar='P',
                    help='filter size for (max) pooling layers. default 1 corresponds to doing nothing.')
parser.add_argument('--train-sets', type=int, nargs='+', default=[2,3], metavar='SS',
                    help='identifiers for the ColorChecker training set. default [2,3]')
parser.add_argument('--valid-sets', type=int, nargs='+', default=[2,3], metavar='SS',
                    help='identifiers for the ColorChecker training set. default [2,3]')
parser.add_argument('--test-sets', type=int, nargs='+', default=[1], metavar='SS',
                    help='identifiers for the ColorChecker training set. default [2,3]')
parser.add_argument('--loss', type=str, default='L1', metavar='T',
                    help='loss function used to guide gradient backpropagation (''Angular'', ''L1'', ''L2'')')

args = parser.parse_args()

# General purpose implementation
class ConvolutionalEB(nn.Module):
    def __init__(self, njet=0, sigma=None, mink_norm=1.0, W=3, N=1):
        super(ConvolutionalEB, self).__init__()

        # Traditional parameters from Low-Level Color Constancy
        #  njet: derivative filter order
        #  sigma: gaussian standard deviation
        #  mink_norm: minkoski norm (p)

        # Additional parameters
        #  W: filter size
        #  N: number of intermediate filters (only 0 or 1)

        # Additional parameters inherited from args (for debug only)
        #  inner_size (TODO bring as input parameter)
        #  dilation
        #  mink_protection_pow
        #  njet_protection1
        #  nl
        #  njet_protection2
        #  mink_protection
        #  pool_size (TODO bring as input parameter)
        #  W_last (TODO bring as input parameter)

        # Constants
        break_off_sigma = 3.

        # Parameters check:
        # inner_size can be customized only if there is an intermediate filter (N==1), otherwise it is reverted to the output of conv1, i.e. 9.
        # inner_size should not be < 9 since it would destroy information from conv1 in case of 2nd-order Grey Edge
        if N<1 and not args.inner_size==9:
            args.inner_size = 9
            warnings.warn('args.inner_size reverted to 9, since no intermediate convolution was requested (N<1).', stacklevel=2)
        if args.inner_size < 9 and njet==2:
            warnings.warn('args.inner_size < 9 destroys information from conv1 in case of 2nd-order Grey Edge. Use at your own risk.', stacklevel=2)
        # W should match the formula for break_off_sigma=3 if sigma is specified. If it does not, the actual filter gets truncated (or centered)
        # W should be an odd number
        if not sigma is None and sigma >= 0:
            W_necessary = (np.floor(break_off_sigma*sigma+0.5)*2+1).astype(int).item()
            if W <= 0:
                W = W_necessary
            else:
                if W < W_necessary:
                    warnings.warn('The requested filter size (W) is too small to accomodate the requested sigma. A truncated version will be used. Alternatively, use W <= 0 to define it automatically.', stacklevel=2)
                elif W > W_necessary:
                    warnings.warn('The requested filter size (W) is larger than need for the requested sigma. Its content will be centered. Alternatively, use W <= 0 to define it automatically.', stacklevel=2)
        if (W % 2) == 0:
            warnings.warn('The requested filter size (W) should be an odd number.')
        # The intermediate non-linearity is ignored if there are no intermediate filters
        if N<1 and not args.nl=='none':
            args.nl = 'none'
            warnings.warn('args.nl reverted to ''none'', since no intermediate convolution was requested (N<1).', stacklevel=2)

        # Behavior:
        # If sigma is not specified (or < 0):
        #     Initialize a network whose ARCHITECTURE is equivalent to Low-Level Color Constancy.
        #     The FILTERS are randomly initialized according do Xavier et al.
        #     Default filter size if W is not specified (or <=0) is 3.
        # Else, if sigma is specified:
        #     Initialize a network whose ARCHITECTURE and FILTERS are equivalent to Low-Level Color Constancy.
        #     If W is not specified (or <= 0):
        #          W is computed automatically such that break-off sigma = 3
        #     Else, if W is specified:
        #          The filters are truncated (or centered) to respect the required filter size.

        # Initialize convolutional filters
        # Conv2D-1
        if sigma >= 0:
            padding_mode = 'replicate'
        else:
            padding_mode = 'zeros'
        self.conv1 = nn.Conv2d(3, 9, W, stride=1, padding=math.floor(((W-1)/2)+0.5), padding_mode=padding_mode)
        # Conv2D (intermediate)
        if N == 0:
            self.convI = nn.Identity()
        elif N == 1:
            self.convI = nn.Conv2d(9, args.inner_size, W, stride=1, dilation=args.dilation, padding=math.floor(((W-1)/2)+0.5))
        else:
            error('Unsupported number of intermediate filters (only 0 or 1)')
        # Conv2D-2
        self.conv2 = nn.Conv2d(args.inner_size, 3, args.W_last, stride=1, padding=math.floor(((args.W_last-1)/2)+0.5))
        self.conv2.bias.data.fill_(1.)

        # Change filters initialization if sigma is specified
        if not sigma is None and sigma >= 0:
            # .weight shape: # (out_channels, in_channels, kernel_size[0], kernel_size[1])
            # .bias shape: # (out_channels)

            # Midpoint for filters of size W. Floor used only to handle even-sized filters.
            midW = np.floor((W-1)/2.).astype(int)
            midWlast = np.floor((args.W_last-1)/2.).astype(int)

            if sigma == 0:
                # Grey World, White Patch, Shades of Grey

                self.conv1.weight.data.fill_(0.)
                for ii in range(3):
                    self.conv1.weight.data[ii, ii, midW, midW] = 1.
                self.conv1.bias.data.fill_(0.)

                if isinstance(self.convI, nn.Conv2d):
                    self.convI.weight.data.fill_(0.)
                    for ii in range(3):
                        self.convI.weight.data[ii, ii, midW, midW] = 1.
                    self.convI.bias.data.fill_(0.)

                self.conv2.weight.data.fill_(0.)
                for ii in range(3):
                    self.conv2.weight.data[ii, ii, midWlast, midWlast] = 1.
                self.conv2.bias.data.fill_(0.)

            else:
                half_filter_size = torch.floor(torch.tensor(break_off_sigma*sigma+0.5))
                x = torch.arange(-half_filter_size, half_filter_size+1)
                Gauss = 1/(torch.sqrt(2 * torch.tensor(np.pi)) * sigma)* torch.exp((x**2)/(-2 * sigma * sigma))

                if njet == 0:
                    # General Grey World

                    # initialize conv1 as 3 independent gaussian filters with specified sigma ---------------
                    G0 = Gauss/torch.sum(Gauss)
                    G00 = G0[...,None]*G0[None,...] # gd00 = filter2(G', G, 'full'); filter2(G', filter2(G, f_ggw(:,:,ii)));
                    # Crop or center G00
                    G00 = FF.center_crop(G00, [W,W])
                    self.conv1.weight.data.fill_(0.)

                    self.conv1.weight.data[0, 0, :, :] = G00 # gDer(input_data(:,:,ii),sigma,0,0);
                    self.conv1.weight.data[1, 1, :, :] = G00 # "
                    self.conv1.weight.data[2, 2, :, :] = G00 # "
                    self.conv1.bias.data.fill_(0.)

                    # initialize convI as identity if it is a convolutional filter ------------------------------------------
                    if isinstance(self.convI, nn.Conv2d):
                        self.convI.weight.data.fill_(0.)
                        for ii in range(3):
                            self.convI.weight.data[ii, ii, midW, midW] = 1.
                        self.convI.bias.data.fill_(0.)

                    # initialize conv2 as ... ----------------------------------------------------------
                    self.conv2.weight.data.fill_(0.)
                    self.conv2.weight.data[0, 0, midWlast, midWlast] = 1.
                    self.conv2.weight.data[1, 1, midWlast, midWlast] = 1.
                    self.conv2.weight.data[2, 2, midWlast, midWlast] = 1.
                    self.conv2.bias.data.fill_(0.)

                elif njet == 1:
                    # 1st-order Grey Edge

                    # initialize conv1 as six independent gaussian-derivative filters with specified sigma ---------------
                    G0 = Gauss/torch.sum(Gauss)
                    G1 = -(x/sigma**2)*Gauss
                    G1 = G1/(torch.sum(torch.sum(x*G1)));
                    G10 = G0[...,None]*G1[None,...] # gd10 = filter2(G0', G1, 'full'); # filter2(G0', filter2(G1, f_ge1(:,:,ii)));
                    G01 = G1[...,None]*G0[None,...] # gd01 = -filter2(G1', G0, 'full'); # filter2(G1', filter2(G0, f_ge1(:,:,ii)));
                    # Crop or center G10 G01
                    G10 = FF.center_crop(G10, [W,W])
                    G01 = FF.center_crop(G01, [W,W])
                    self.conv1.weight.data.fill_(0.)
                    self.conv1.weight.data[0, 0, :, :] = G10 # Rx=gDer(R,sigma,1,0);
                    self.conv1.weight.data[1, 1, :, :] = G10 # Gx=gDer(G,sigma,1,0);
                    self.conv1.weight.data[2, 2, :, :] = G10 # Bx=gDer(B,sigma,1,0);
                    self.conv1.weight.data[3, 0, :, :] = G01 # Ry=gDer(R,sigma,0,1);
                    self.conv1.weight.data[4, 1, :, :] = G01 # Gy=gDer(G,sigma,0,1);
                    self.conv1.weight.data[5, 2, :, :] = G01 # By=gDer(B,sigma,0,1);
                    self.conv1.bias.data.fill_(0.)

                    # initialize convI as identity if it is a convolutional filter ------------------------------------------
                    if isinstance(self.convI, nn.Conv2d):
                        self.convI.weight.data.fill_(0.)
                        for ii in range(6):
                            self.convI.weight.data[ii, ii, midW, midW] = 1.
                        self.convI.bias.data.fill_(0.)

                    # initialize conv2 as ... ----------------------------------------------------------
                    self.conv2.weight.data.fill_(0.)
                    # Rx.^2+Ry.^2
                    self.conv2.weight.data[0, 0, midWlast, midWlast] = 1.
                    self.conv2.weight.data[0, 3, midWlast, midWlast] = 1.
                    # Gx.^2+Gy.^2
                    self.conv2.weight.data[1, 1, midWlast, midWlast] = 1.
                    self.conv2.weight.data[1, 4, midWlast, midWlast] = 1.
                    # Bx.^2+By.^2
                    self.conv2.weight.data[2, 2, midWlast, midWlast] = 1.
                    self.conv2.weight.data[2, 5, midWlast, midWlast] = 1.
                    self.conv2.bias.data.fill_(0.)

                elif njet == 2:
                    # 2nd-order Grey Edge

                    # initialize conv1 as nine independent gaussian-derivative filters with specified sigma ---------------
                    G0 = Gauss/torch.sum(Gauss)
                    G1 = -(x/sigma**2)*Gauss
                    G1 = G1/(torch.sum(torch.sum(x*G1)))
                    G2 = (x**2/sigma**4-1/sigma**2)*Gauss
                    G2 = G2-torch.sum(G2)/len(x)
                    G2 = G2/torch.sum(0.5*x*x*G2)
                    G11 = G1[...,None]*G1[None,...] # gd11 = -filter2(G1', G1, 'full'); # filter2(G1', filter2(G1, f_ge2(:,:,ii)));
                    G02 = G2[...,None]*G0[None,...] # gd02 = filter2(G2', G0, 'full'); # filter2(G2', filter2(G0, f_ge2(:,:,ii)));
                    G20 = G0[...,None]*G2[None,...] # gd20 = filter2(G0', G2, 'full'); # filter2(G0', filter2(G2, f_ge2(:,:,ii)));
                    # Crop or center G11 G02 G20
                    G11 = FF.center_crop(G11, [W,W])
                    G02 = FF.center_crop(G02, [W,W])
                    G20 = FF.center_crop(G20, [W,W])
                    self.conv1.weight.data.fill_(0.)
                    self.conv1.weight.data[0, 0, :, :] = G20 # Rxx=gDer(R,sigma,2,0);
                    self.conv1.weight.data[1, 1, :, :] = G20 # Gxx=gDer(G,sigma,2,0);
                    self.conv1.weight.data[2, 2, :, :] = G20 # Bxx=gDer(B,sigma,2,0);
                    self.conv1.weight.data[3, 0, :, :] = G02 # Ryy=gDer(R,sigma,0,2);
                    self.conv1.weight.data[4, 1, :, :] = G02 # Gyy=gDer(G,sigma,0,2);
                    self.conv1.weight.data[5, 2, :, :] = G02 # Byy=gDer(B,sigma,0,2);
                    self.conv1.weight.data[6, 0, :, :] = G11 # Rxy=gDer(R,sigma,1,1);
                    self.conv1.weight.data[7, 1, :, :] = G11 # Gxy=gDer(G,sigma,1,1);
                    self.conv1.weight.data[8, 2, :, :] = G11 # Bxy=gDer(B,sigma,1,1);
                    self.conv1.bias.data.fill_(0.)

                    # initialize convI as identity if it is a convolutional filter ------------------------------------------
                    if isinstance(self.convI, nn.Conv2d):
                        self.convI.weight.data.fill_(0.)
                        for ii in range(9):
                            self.convI.weight.data[ii, ii, midW, midW] = 1.
                        self.convI.bias.data.fill_(0.)

                    # initialize conv2 as ... ----------------------------------------------------------
                    self.conv2.weight.data.fill_(0.)
                    # Rxx.^2+4*Rxy.^2+Ryy.^2
                    self.conv2.weight.data[0, 0, midWlast, midWlast] = 1.
                    self.conv2.weight.data[0, 6, midWlast, midWlast] = 2.
                    self.conv2.weight.data[0, 3, midWlast, midWlast] = 1.
                    # Gxx.^2+4*Gxy.^2+Gyy.^2
                    self.conv2.weight.data[1, 1, midWlast, midWlast] = 1.
                    self.conv2.weight.data[1, 7, midWlast, midWlast] = 2.
                    self.conv2.weight.data[1, 4, midWlast, midWlast] = 1.
                    # Bxx.^2+4*Bxy.^2+Byy.^2
                    self.conv2.weight.data[2, 2, midWlast, midWlast] = 1.
                    self.conv2.weight.data[2, 8, midWlast, midWlast] = 2.
                    self.conv2.weight.data[2, 5, midWlast, midWlast] = 1.
                    self.conv2.bias.data.fill_(0.)

                else:
                    error('Unsupported njet > 2')

        if args.pool_size > 1:
            self.pool1 = nn.MaxPool2d(args.pool_size, stride=1, padding=math.floor(((args.pool_size-1)/2)+0.5))
        else:
            self.pool1 = nn.Identity()

        if mink_norm > 0:
            self.poolN = AdaLearnLPP3(norm_type=mink_norm, pow_protection=args.mink_protection_pow)
        elif mink_norm == -1:
            self.poolN = nn.AdaptiveMaxPool2d(1)
            # TODO: replace with nn.MaxPool2d to handle local max pooling, for spatially varying estimation
        else:
            self.poolN = nn.Identity()

        if njet < 1:
            self.power = 1
        else:
            self.power = 2

        if args.njet_protection1 == 'abs':
            self.nl1 = torch.abs
        elif args.njet_protection1 == 'relu':
            self.nl1 = F.relu
        elif args.njet_protection1 == 'prelu':
            self.nl1 = nn.PReLU(init=1.0) # <<<<<<<<<< initialized 1.0
        elif args.njet_protection1 == 'posrelu':
            self.nl1 = PosReLU()
        else:
            self.nl1 = lambda a : a

        if args.nl == 'abs':
            self.nlI = torch.abs
        elif args.nl == 'relu':
            self.nlI = F.relu
        elif args.nl == 'prelu':
            self.nlI = nn.PReLU(init=1.0)
        elif args.nl == 'posrelu':
            self.nlI = PosReLU()
        else:
            self.nlI = lambda a : a

        if args.njet_protection2 == 'abs':
            self.nl2 = torch.abs
        elif args.njet_protection2 == 'relu':
            self.nl2 = F.relu
        elif args.njet_protection2 == 'prelu':
            self.nl2 = nn.PReLU(init=0.0)
        elif args.njet_protection2 == 'posrelu':
            self.nl2 = PosReLU(offset=0.00000000000000000001)
        else:
            self.nl2 = lambda a : a

        if args.mink_protection == 'abs':
            self.mink_protection = torch.abs
        elif args.mink_protection == 'relu':
            self.mink_protection = F.relu
        elif args.mink_protection == 'prelu':
            self.mink_protection = nn.PReLU(init=0.0)
        elif args.mink_protection == 'posrelu':
            self.mink_protection = PosReLU()
        else:
            self.mink_protection = lambda a : a

    def forward(self, x):

        x = x*255.0

        x = self.conv1(x)
        x = self.nl1(x)
        x = self.pool1(x)
        x = torch.pow(x, self.power)

        x = self.convI(x)
        x = self.nlI(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nl2(x)
        x = self.pool1(x)
        x = torch.pow(x, 1/self.power)

        x = self.mink_protection(x)
        x = self.poolN(x)

        x = F.normalize(x, p=2, dim=1)

        return x

print(args)
print('Parameters:\n\tnjet = {:d}\n\tmink_norm (p) = {:.2f}\n\tsigma = {:.2f}\n\tfilter size (W) = {:d}\n\tintermediate filter blocks (N) = {:d}'.format(args.njet, args.mink_norm, args.sigma, args.W, args.intermediate_blocks))
if torch.cuda.is_available() and args.device.lower()=='cuda':
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)

os.makedirs(os.path.join(args.dir_exp, args.name_exp), exist_ok=True)

# Model instantiation
ceb = ConvolutionalEB(njet=args.njet, mink_norm=args.mink_norm, sigma=args.sigma, W=args.W, N=args.intermediate_blocks)
if isinstance(args.starting_epoch, str) or args.starting_epoch > 0:
    ceb.load_state_dict(torch.load(os.path.join(args.dir_exp,args.name_exp,'ceb_{}.pth'.format(args.starting_epoch))))
ceb.to(device)

# Loss function
if args.loss.lower() == 'angular':
    criterion = RecoveryLoss(angle_type='degrees', reduction='mean')
if args.loss.lower() == 'l1':
    criterion = nn.L1Loss(reduction='mean')
if args.loss.lower() == 'l2':
    criterion = nn.MSELoss(reduction='mean')

# Optimizer
optimizer = optim.Adam(ceb.parameters(), lr=args.lr, weight_decay=args.wdecay)

if args.data_augmentation:
    print('Data Augmentation ENABLED.')
    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(10, resample=Image.BILINEAR, expand=False),
        torchvision.transforms.RandomAffine(10, translate=(0.1,0.1), scale=None, shear=10, resample=Image.BILINEAR),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.80, 1.25), ratio=(0.75, 1.3333333333333333), interpolation=Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        ])
else:
    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.image_size,args.image_size)),
        torchvision.transforms.ToTensor(),
        ])

if args.validation_transform == 'none':
    transforms_valid = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])
    valid_batch_size = args.batch_size
elif args.validation_transform == 'isotropic':
    # TODO: generalize for other datasets:
    warnings.warn('Isotropic rescaling works only for Color Checker dataset', stacklevel=2)
    min_size = np.round(args.image_size * 533.0/800.0).astype(int).item()
    # min_size = args.image_size

    transforms_valid = torchvision.transforms.Compose([
        torchvision.transforms.Resize(min_size),
        torchvision.transforms.ToTensor(),
        ])
    valid_batch_size = 1
elif args.validation_transform == 'anisotropic':
    transforms_valid = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.image_size,args.image_size)),
        torchvision.transforms.ToTensor(),
        ])
    valid_batch_size = args.batch_size

train_dataset = ColorCheckerDataset(root=args.dir_train,
                              gt=os.path.join(args.dir_gt,args.gt),
                              sets=args.train_sets,
                              extension='.png',
                              transforms=transforms_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

valid_dataset = ColorCheckerDataset(root=args.dir_train,
                              gt=os.path.join(args.dir_gt,args.gt),
                              sets=args.valid_sets,
                              extension='.png',
                              transforms=transforms_valid)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)

test_dataset = ColorCheckerDataset(root=args.dir_train,
                              gt=os.path.join(args.dir_gt,args.gt),
                              sets=args.test_sets,
                              extension='.png',
                              transforms=transforms_valid)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)

best_train = 10000
best_valid = 10000
best_train_epoch = -1
best_valid_epoch = -1

def train(epoch):
    ceb.train()

    loss = 0
    num = 0

    for batch_idx, (img, target) in enumerate(train_loader):

        img = img.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        output = ceb(img)
        output = torch.squeeze(torch.squeeze(output,2),2)

        cur_loss = criterion(output, target)
        cur_loss.backward()
        optimizer.step()

        print('TRAIN\te:{:4d}\tb:{:4d}\tp:{:.2f}  -  Loss: {:.5f}'.format(epoch, batch_idx, ceb.poolN.norm_type.item(), cur_loss))

        cur_num = target.shape[0]
        loss += cur_loss.item()*cur_num
        num += cur_num

    torch.save(ceb.state_dict(), os.path.join(args.dir_exp,args.name_exp,'ceb_last.pth'))

    loss = loss/num

    print('TRAIN\te:{:4d}\tAVG     -  Loss: {:.5f} *'.format(epoch, loss))
    global best_train
    global best_train_epoch
    if loss < best_train:
        best_train = loss
        best_train_epoch = epoch
        print('* BEST ON TRAINING **************************')
        torch.save(ceb.state_dict(), os.path.join(args.dir_exp,args.name_exp,'ceb_best_train.pth'))

    if epoch % args.checkpoint_interval == 0:
        torch.save(ceb.state_dict(), os.path.join(args.dir_exp,args.name_exp,'ceb_{}.pth'.format(epoch)))


def valid(epoch):
    with torch.no_grad():
        ceb.eval()

        err = 0
        num = 0
        for batch_idx, (img, target) in enumerate(valid_loader):
            img = img.to(device)
            target = target.to(device)

            output = ceb(img)
            output = torch.squeeze(torch.squeeze(output,2),2)

            cur_err = recovery_error(output, target)

            cur_num = target.shape[0]
            err += cur_err.item()*cur_num
            num += cur_num

        err = err/num

    print('VALID\te:{:4d}\tAVG     -   Err: {:.5f} #'.format(epoch, err))
    global best_valid
    global best_valid_epoch
    if err < best_valid:
        best_valid = err
        best_valid_epoch = epoch
        print('* BEST ON VALIDATION ########################')
        torch.save(ceb.state_dict(), os.path.join(args.dir_exp,args.name_exp,'ceb_best_valid.pth'))


def test(save_output=False):
    with torch.no_grad():
        ceb.eval()

        errs = np.array([], dtype='single')
        outputs = np.zeros((0,3), dtype='single')

        num = 0
        for batch_idx, (img, target) in enumerate(test_loader):
            img = img.to(device)
            target = target.to(device)

            output = ceb(img)
            output = torch.squeeze(torch.squeeze(output,2),2)

            cur_err = recovery_error(output, target, reduction=None)

            cur_num = target.shape[0]
            errs = np.concatenate((errs, cur_err.cpu().numpy()))
            outputs = np.concatenate((outputs, output.cpu().numpy()))
            num += cur_num

    print('Min\tMean\tMedian\t90-prc\t95-prc\t99-prc\tMax')
    print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t'.format(errs.min(), errs.mean(), np.median(errs), np.percentile(errs,90), np.percentile(errs,95), np.percentile(errs,99), errs.max()))

    if save_output:
        np.savetxt(os.path.join(args.dir_exp,args.name_exp,'output.csv'), outputs, delimiter=',')
        np.savetxt(os.path.join(args.dir_exp,args.name_exp,'errors.csv'), errs, delimiter=',')


def time(save_output=False):
    with torch.no_grad():
        ceb.eval()

        # Dry run
        for img, _ in test_loader:
            img = img.to(device)
            _ = ceb(img)

        times = np.array([], dtype='single')

        num = 0
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        for batch_idx, (img, _) in enumerate(test_loader):
            img = img.to(device)

            t_start.record()
            _ = ceb(img)
            t_end.record()
            torch.cuda.synchronize()
            t = t_start.elapsed_time(t_end)

            cur_num = img.shape[0]
            times = np.concatenate((times, (t,)))
            num += cur_num

    print('Mean\tStd\tMedian')
    print('{:.4e}\t{:.4e}\t{:.4e}'.format(np.mean(times/1000.0)/args.batch_size, np.std(times/1000.0)/args.batch_size, np.median(times/1000.0)/args.batch_size))

    filename = f'time_sigma{args.sigma}_imsize{args.image_size}_device{args.device}_batchsize{args.batch_size}_{args.name_exp}.txt';
    with open(filename, 'w') as text_file:
        text_file.write('{:.4e}\t{:.4e}\t{:.4e}'.format(np.mean(times/1000.0)/args.batch_size, np.std(times/1000.0)/args.batch_size, np.median(times/1000.0)/args.batch_size))

    if save_output:
        np.savetxt(os.path.join(args.dir_exp,args.name_exp,'times.csv'), times, delimiter=',')


if args.mode == 'train':
    # Save initialization (only if starting from scratch)
    if args.save_init and not isinstance(args.starting_epoch, str) and not args.starting_epoch > 0:
        torch.save(ceb.state_dict(), os.path.join(args.dir_exp,args.name_exp,'ceb_0.pth'))

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        valid(epoch)

    if args.epochs > 0:
        print('\nTEST SET (last epoch)')
        test()

    print('\nTEST SET (best on train: epoch {:d})'.format(best_train_epoch))
    ceb.load_state_dict(torch.load(os.path.join(args.dir_exp,args.name_exp,'ceb_best_train.pth')))
    ceb.to(device)
    test()

    print('\nTEST SET (best on valid: epoch {:d})'.format(best_valid_epoch))
    ceb.load_state_dict(torch.load(os.path.join(args.dir_exp,args.name_exp,'ceb_best_valid.pth')))
    ceb.to(device)
    test(save_output = args.save_output)

elif args.mode == 'test':
    test(save_output = args.save_output)

elif args.mode == 'time':
    time(save_output = args.save_output)

elif args.mode == 'debug':
    # Single-image inference
    img_path = '/building1.png'

    img = Image.open(img_path)
    img = torchvision.transforms.ToTensor()(img)
    img = img[None,...]

    img = img.to(device)
    output = ceb(img)

    print(torch.reshape(output.detach().cpu(), (-1,)).numpy())
    