##########################################################
# pytorch-qnn v1.0
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# August 2020
##########################################################

import numpy                   as np
from numpy.random import RandomState
import torch
from torch.autograd import Variable
import torch.nn.functional      as F
import torch.nn                 as nn
from torch.nn.parameter import Parameter
from torch.nn import Module
from .hypercomplex_ops import *
from .helpers import _ntuple, to_1tuple, to_2tuple, to_3tuple, to_4tuple, to_ntuple
import math
import sys


class HyperTransposeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, num_components=8,
                 bias=True, operation='conv2d'):
        super(HyperTransposeConv, self).__init__()
        if operation.endswith('1d'):
            convFunc = nn.ConvTranspose1d
        elif operation.endswith('2d'):
            convFunc = nn.ConvTranspose2d
        elif operation.endswith('3d'):
            convFunc = nn.ConvTranspose3d
        else:
            raise Exception(f'parameter "operation" has to end with [1d, 2d, 3d] but found {operation}')

        n = int(''.join(c for c in operation if c.isdigit()))
        self.num_components = num_components
        if (in_channels % self.num_components) != 0:
            raise Exception(f'number of input channel must be divisible by {self.num_components}; found {in_channels}')
        if (out_channels % self.num_components) != 0:
            raise Exception(f'number of output channel must be divisible by {self.num_components}; found {out_channels}')

        self.in_channels = in_channels // self.num_components
        self.out_channels = out_channels // self.num_components
        self.stride = to_ntuple(n)(stride)
        self.padding = to_ntuple(n)(padding)
        self.output_padding = to_ntuple(n)(output_padding)
        self.groups = groups
        self.dilation = to_ntuple(n)(dilation)
        # self.init_criterion = init_criterion
        # self.weight_init = weight_init
        # self.seed = seed if seed is not None else np.random.randint(0, 1234)
        # self.rng = RandomState(self.seed)
        self.operation = operation

        self.bias = bias
        self.kernel_size = to_ntuple(n)(kernel_size)

        # self.conv_blks = []
        # self.components = ['r', 'i', 'j', 'k', 'e', 'l', 'm', 'n']
        for component in range(self.num_components):  # self.components:
            conv_c = convFunc(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              bias=self.bias, stride=self.stride,
                              kernel_size=self.kernel_size, padding=self.padding,
                              output_padding=self.output_padding)
            setattr(self, f'conv_{component}', conv_c)
            # self.conv_blks.append(conv_c)

        # initialization
        # self.kernel_size = conv_c.kernel_size
        # weights = weight_init(in_features=self.in_channels, out_features=self.out_channels,
        #                       kernel_size=self.kernel_size, modalities=self.modalities)
        # for modality in range(self.modalities):
        #     self.conv_blks[modality].weight.data = weights[modality]

    def forward(self, input):
        check_input(input, self.num_components)
        comp = get_comp_mat(num_components=self.num_components)
        x = [get_c(input, component=idx, num_components=self.num_components) for idx in range(self.num_components)]
        y = []
        for comp_i in comp:
            y_comp = 0
            for idx, ii in enumerate(comp_i):
                itr = np.abs(ii)
                sign = np.sign(ii)
                # y_comp += sign * self.conv_blks[itr](x[idx])
                y_comp += eval(f'sign * self.conv_{itr}(x[idx])')
                # y_comp += sign * self.conv_blks[itr](get_c(input, component=idx, num_components=self.num_components))
            y.append(y_comp)
        out = torch.cat(y, dim=1)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels * self.num_components) \
               + ', out_channels=' + str(self.out_channels * self.num_components) \
               + ', bias=' + str(self.bias) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', out_padding=' + str(self.output_padding) \
               + ', num_components=' + str(self.num_components) \
               + ', operation=' + str(self.operation) + ')'


class HyperConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, num_components=8,
                 bias=True, operation='conv2d'):
        super(HyperConv, self).__init__()
        if operation.endswith('1d'):
            convFunc = nn.Conv1d
        elif operation.endswith('2d'):
            convFunc = nn.Conv2d
        elif operation.endswith('3d'):
            convFunc = nn.Conv3d
        else:
            raise Exception(f'parameter "operation" has to end with [1d, 2d, 3d] but found {operation}')

        n = int(''.join(c for c in operation if c.isdigit()))
        self.num_components = num_components
        if (in_channels % self.num_components) != 0:
            raise Exception(f'number of input channel must be divisible by {self.num_components}; found {in_channels}')
        if (out_channels % self.num_components) != 0:
            raise Exception(f'number of output channel must be divisible by {self.num_components}; found {out_channels}')

        self.in_channels = in_channels // self.num_components
        self.out_channels = out_channels // self.num_components
        self.stride = to_ntuple(n)(stride)
        self.padding = to_ntuple(n)(padding)
        self.groups = groups
        self.dilation = to_ntuple(n)(dilation)
        # self.init_criterion = init_criterion
        # self.weight_init = weight_init
        # self.seed = seed if seed is not None else np.random.randint(0, 1234)
        # self.rng = RandomState(self.seed)
        self.operation = operation

        self.bias = bias
        self.kernel_size = to_ntuple(n)(kernel_size)

        # self.conv_blks = []
        # self.components = ['r', 'i', 'j', 'k', 'e', 'l', 'm', 'n']
        for component in range(self.num_components):  # self.components:
            conv_c = convFunc(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              bias=self.bias, stride=self.stride,
                              kernel_size=self.kernel_size, padding=self.padding, groups=self.groups)
            setattr(self, f'conv_{component}', conv_c)
            # self.conv_blks.append(conv_c)

        # initialization
        # self.kernel_size = conv_c.kernel_size
        # weights = weight_init(in_features=self.in_channels, out_features=self.out_channels,
        #                       kernel_size=self.kernel_size, modalities=self.modalities)
        # for modality in range(self.modalities):
        #     self.conv_blks[modality].weight.data = weights[modality]

    def forward(self, input):
        check_input(input, self.num_components)
        comp = get_comp_mat(num_components=self.num_components)
        x = [get_c(input, component=idx, num_components=self.num_components) for idx in range(self.num_components)]
        y = []
        for comp_i in comp:
            y_comp = 0
            for idx, ii in enumerate(comp_i):
                itr = np.abs(ii)
                sign = np.sign(ii)
                # y_comp += sign * self.conv_blks[itr](x[idx])
                y_comp += eval(f'sign * self.conv_{itr}(x[idx])')
                # y_comp += sign * self.conv_blks[itr](get_c(input, component=idx, num_components=self.num_components))
            y.append(y_comp)
        out = torch.cat(y, dim=1)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_channels=' + str(self.in_channels * self.num_components) \
               + ', out_channels=' + str(self.out_channels * self.num_components) \
               + ', bias=' + str(self.bias) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ((', groups=' + str(self.groups)) if self.groups > 1 else '')  \
               + ', num_components=' + str(self.num_components) \
               + ', operation=' + str(self.operation) + ')'


class HyperLinear(nn.Module):
    # def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=0,
    #              dilation=1, groups=1, num_components=8,
    #              bias=True, operation='conv2d'):
    def __init__(self, in_features, out_features, num_components=8, bias=True, ):
        r"""Applies an hypercomplex linear transformation to the incoming data.
            """
        super(HyperLinear, self).__init__()

        self.num_components = num_components
        if (in_features % self.num_components) != 0:
            raise Exception(f'number of input feature must be divisible by {self.num_components}; found {in_features}')
        if (out_features % self.num_components) != 0:
            raise Exception(f'number of output feature must be divisible by {self.num_components}; found {out_features}')

        self.in_features = in_features // self.num_components
        self.out_features = out_features // self.num_components
        # self.init_criterion = init_criterion
        # self.weight_init = weight_init
        # self.seed = seed if seed is not None else np.random.randint(0, 1234)
        # self.rng = RandomState(self.seed)

        self.bias = bias

        for component in range(self.num_components):  # self.components:
            fc_c = nn.Linear(in_features=self.in_features,
                             out_features=self.out_features,
                             bias=self.bias,
                             )
            setattr(self, f'fc_{component}', fc_c)

        # initialization
        # self.kernel_size = conv_c.kernel_size
        # weights = weight_init(in_features=self.in_features, out_features=self.out_features,
        #                       kernel_size=self.kernel_size, modalities=self.modalities)
        # for modality in range(self.modalities):
        #     self.conv_blks[modality].weight.data = weights[modality]

    def forward(self, input):
        check_input(input, self.num_components)
        comp = get_comp_mat(num_components=self.num_components)
        x = [get_c(input, component=idx, num_components=self.num_components) for idx in range(self.num_components)]
        y = []
        for comp_i in comp:
            y_comp = 0
            for idx, ii in enumerate(comp_i):
                itr = np.abs(ii)
                sign = np.sign(ii)
                # y_comp += sign * self.conv_blks[itr](x[idx])
                y_comp += eval(f'sign * self.fc_{itr}(x[idx])')
                # y_comp += sign * self.conv_blks[itr](get_c(input, component=idx, num_components=self.num_components))
            y.append(y_comp)
        out = torch.cat(y, dim=1)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias) \
               + ', num_components=' + str(self.num_components) + ')'


class HyperConv1D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride, num_components=8,
                 dilation=1, padding=0, groups=1, bias=True):
        super(HyperConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          dilation=dilation, padding=padding,
                                          groups=groups, bias=bias, num_components=num_components,
                                          operation='convolution1d'
                                          )

    def __repr__(self):
        config = super(HyperConv1D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config


class HyperConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride, num_components=8,
                 dilation=1, padding=0, groups=1, bias=True):
        super(HyperConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          dilation=dilation, padding=padding,
                                          groups=groups, bias=bias, num_components=num_components,
                                          operation='convolution2d'
                                          )

    def __repr__(self):
        config = super(HyperConv2D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config


class HyperConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride, num_components=8,
                 dilation=1, padding=0, groups=1, bias=True):
        super(HyperConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          dilation=dilation, padding=padding,
                                          groups=groups, bias=bias, num_components=num_components,
                                          operation='convolution3d'
                                          )

    def __repr__(self):
        config = super(HyperConv3D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config


class ComplexConv1D(HyperConv):

        def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                     dilation=1, padding=0, groups=1, bias=True, ):
            super(ComplexConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                dilation=dilation, padding=padding,
                                                groups=groups, bias=bias,num_components=2, operation='convolution1d',
                                                )

        def __repr__(self):
            config = super(ComplexConv1D, self).__repr__()
            config = config.replace(f', operation={str(self.operation)}', '')
            return config.replace(f', num_components={str(self.num_components)}', '')


class ComplexConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(ComplexConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias,
                                            num_components=2, operation='convolution2d',
                                            )

    def __repr__(self):
        config = super(ComplexConv2D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class ComplexConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(ComplexConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            dilation=dilation, padding=padding,
                                            groups=groups, bias=bias,
                                            num_components=2, operation='convolution3d',
                                            )

    def __repr__(self):
        config = super(ComplexConv3D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class QuaternionConv1D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(QuaternionConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               dilation=dilation, padding=padding,
                                               groups=groups, bias=bias,
                                               num_components=4, operation='convolution1d',
                                               )

    def __repr__(self):
        config = super(QuaternionConv1D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class QuaternionConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(QuaternionConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               dilation=dilation, padding=padding,
                                               groups=groups, bias=bias,
                                               num_components=4, operation='convolution2d',
                                               )

    def __repr__(self):
        config = super(QuaternionConv2D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class QuaternionConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(QuaternionConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                               dilation=dilation, padding=padding,
                                               groups=groups, bias=bias,
                                               num_components=4, operation='convolution3d',
                                               )

    def __repr__(self):
        config = super(QuaternionConv3D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class OctonionConv1D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(OctonionConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                             dilation=dilation, padding=padding,
                                             groups=groups, bias=bias,
                                             num_components=8, operation='convolution1d',
                                             )

    def __repr__(self):
        config = super(OctonionConv1D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class OctonionConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(OctonionConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                             dilation=dilation, padding=padding,
                                             groups=groups, bias=bias,
                                             num_components=8, operation='convolution2d',
                                             )

    def __repr__(self):
        config = super(OctonionConv2D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class OctonionConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(OctonionConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                             dilation=dilation, padding=padding,
                                             groups=groups, bias=bias,
                                             num_components=8, operation='convolution3d',
                                             )

    def __repr__(self):
        config = super(OctonionConv3D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class SedanionConv1D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(SedanionConv1D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                             dilation=dilation, padding=padding,
                                             groups=groups, bias=bias,
                                             num_components=16, operation='convolution1d',
                                             )

    def __repr__(self):
            config = super(SedanionConv1D, self).__repr__()
            config = config.replace(f', operation={str(self.operation)}', '')
            return config.replace(f', num_components={str(self.num_components)}', '')


class SedanionConv2D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(SedanionConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                             dilation=dilation, padding=padding,
                                             groups=groups, bias=bias,
                                             num_components=16, operation='convolution2d',
                                             )

    def __repr__(self):
        config = super(SedanionConv2D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class SedanionConv3D(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride,  # num_components=8,
                 dilation=1, padding=0, groups=1, bias=True, ):
        super(SedanionConv3D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                             dilation=dilation, padding=padding,
                                             groups=groups, bias=bias,
                                             num_components=16, operation='convolution3d',
                                             )

    def __repr__(self):
        config = super(SedanionConv3D, self).__repr__()
        config = config.replace(f', operation={str(self.operation)}', '')
        return config.replace(f', num_components={str(self.num_components)}', '')


class ComplexLinear(HyperLinear):

    def __init__(self, in_features, out_features,  # num_components=8,
                 bias=True, ):
        super(ComplexLinear, self).__init__(in_features, out_features, num_components=2, bias=bias,)

    def __repr__(self):
        config = super(ComplexLinear, self).__repr__()
        config = config.replace(f', num_components={str(self.num_components)}', '')
        return config


class QuaternionLinear(HyperLinear):

    def __init__(self, in_features, out_features,  # num_components=8,
                 bias=True, ):
        super(QuaternionLinear, self).__init__(in_features, out_features, num_components=4, bias=bias,)

    def __repr__(self):
        config = super(QuaternionLinear, self).__repr__()
        config = config.replace(f', num_components={str(self.num_components)}', '')
        return config


class OctonionLinear(HyperLinear):

    def __init__(self, in_features, out_features,  # num_components=8,
                 bias=True,):
        super(OctonionLinear, self).__init__(in_features, out_features, num_components=8, bias=bias,)

    def __repr__(self):
        config = super(OctonionLinear, self).__repr__()
        config = config.replace(f', num_components={str(self.num_components)}', '')
        return config


class SedanionLinear(HyperLinear):

    def __init__(self, in_features, out_features,  # num_components=8,
                 bias=True,):
        super(SedanionLinear, self).__init__(in_features, out_features, num_components=16, bias=bias,)

    def __repr__(self):
        config = super(SedanionLinear, self).__repr__()
        config = config.replace(f', num_components={str(self.num_components)}', '')
        return config
