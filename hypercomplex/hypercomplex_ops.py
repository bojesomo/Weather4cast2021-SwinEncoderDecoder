##########################################################
# pytorch-qnn v1.0
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# August 2020
##########################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState
import sys
import pdb
from scipy.stats import chi
from .hypercomplex_utils import get_comp_mat


def h_normalize(input, channel=1, num_components=8):
    components = [get_c(input, component, num_components) for component in range(num_components)]
    norm = torch.sqrt(torch.stack([component**2 for component in components]).sum() + 0.0001)
    components = [component / norm for component in components]

    return torch.cat(components, dim=channel)


def check_input(input, num_components=8):
    if input.dim() not in {2, 3, 4, 5}:
        raise RuntimeError(
            "This accepts only input of dimension 2 or 3. conv accepts up to 5 dim "
            " input.dim = " + str(input.dim())
        )

    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]

    if nb_hidden % num_components != 0:
        raise RuntimeError(
            f"Tensors must be divisible by {num_components}. {input.size()[1]} = " + str(nb_hidden)
        )


#
# Getters
#
def get_c(input, component=0, num_components=8):
    check_input(input, num_components)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    # components = ['r', 'i', 'j', 'k', 'e', 'l', 'm', 'n']
    index = component  # components.index(component)
    c_hidden = nb_hidden // num_components  # 8
    if input.dim() == 2:
        return input.narrow(1, index * c_hidden, c_hidden)
    if input.dim() == 3:
        return input.narrow(2, index * c_hidden, c_hidden)
    if input.dim() >= 4:
        return input.narrow(1, index * c_hidden, c_hidden)


def get_modulus(input, vector_form=False, num_components=8):
    check_input(input, num_components)
    components = [get_c(input, component, num_components) for component in range(num_components)]
    # r = get_r(input)
    # i = get_i(input)
    # j = get_j(input)
    # k = get_k(input)
    # e = get_e(input)
    # l = get_l(input)
    # m = get_m(input)
    # n = get_n(input)
    if vector_form:
        # return torch.sqrt(r * r + i * i + j * j + k * k + e * e + l * l + m * m + n * n)
        return torch.sqrt(torch.stack([component**2 for component in components]).sum())
    else:
        # return torch.sqrt((r * r + i * i + j * j + k * k + e * e + l * l + m * m + n * n).sum(dim=0))
        return torch.sqrt(torch.stack([component ** 2 for component in components]).sum(dim=0))


def get_normalized(input, eps=0.0001, num_components=8):
    check_input(input, num_components)
    data_modulus = get_modulus(input, num_components=num_components)
    if input.dim() == 2:
        data_modulus_repeated = data_modulus.repeat(1, num_components)  # 8)
    elif input.dim() == 3:
        data_modulus_repeated = data_modulus.repeat(1, 1, num_components)  # 8)
    return input / (data_modulus_repeated.expand_as(input) + eps)


def hypercomplex_exp(input, num_components=8):
    components = [get_c(input, component, num_components) for component in range(num_components)]
    norm_v = torch.sqrt(torch.stack([component ** 2 for component in components]).sum()) + 0.0001
    exp = torch.exp(components[0])

    components[0] = torch.cos(norm_v)
    components[1:] = [(component / norm_v) * torch.sin(norm_v) for component in components[1:]]

    return torch.cat([exp * component for component in components], dim=1)


def make_hypercomplex_mul(weights, n_divs=4, comp_mat=None):
    """
    The constructed 'hamilton' W is a modified version of the hypercomplex representation,
    """
    # return phm(weights, n_divs=n_divs, comp_mat=comp_mat)
    if comp_mat is None:
        comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing

    def sign(ii):
        return np.sign(ii) if np.sign(ii) != 0 else 1

    neg_weights = [-t for t in weights]
    cat_kernels_hypercomplex = torch.cat([torch.cat([weights[np.abs(ii)] if sign(ii) > 0 else neg_weights[np.abs(ii)] for ii in comp_i], dim=1)
                                          for comp_i in comp_mat], dim=0)
    return cat_kernels_hypercomplex


def fast_hypercomplex_mul(weights, n_divs=4, comp_mat=None):
    """
    The constructed 'hamilton' W is a modified version of the hypercomplex representation,
    """
    # return phm(weights, n_divs=n_divs, comp_mat=comp_mat)
    if comp_mat is None:
        comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing


    neg_weights = [-t for t in weights[1:][::-1]]
    weights_new = torch.cat([weights, neg_weights], dim=0)
    kernel = torch.cat([weights_new[comp_i].flatten(1, 2) for comp_i in comp_mat], dim=0)

    return kernel


def hypercomplex_conv(input, weights, bias, stride,
                      padding, groups, dilation):
    """
    Applies a hypercomplex convolution to the incoming data:
    (a, b) (c, d) = (ac -d"b, da + bc")
    d" => d conjugate
    """
    num_components = len(weights)
    comp_mat = get_comp_mat(num_components=num_components)

    # cat_kernel_hypercomplex_i = []
    # for comp_i in comp_mat:
    #     kernel_hypercomplex_i = []
    #     for idx, ii in enumerate(comp_i):
    #         itr = np.abs(ii)
    #         sign = np.sign(ii) if np.sign(ii) != 0 else 1
    #         kernel_hypercomplex_i.append(sign * weights[itr])
    #     cat_kernel_hypercomplex_i.append(torch.cat(kernel_hypercomplex_i, dim=1))
    # # print(cat_kernel_hypercomplex_i.__len__(), [x_.shape for x_ in cat_kernel_hypercomplex_i])
    # cat_kernels_hypercomplex = torch.cat(cat_kernel_hypercomplex_i, dim=0)
    # # print(cat_kernels_hypercomplex.shape)
    cat_kernels_hypercomplex = make_hypercomplex_mul(weights, num_components, comp_mat)

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, cat_kernels_hypercomplex, bias, stride, padding, dilation, groups)


def hypercomplex_transpose_conv(input, weights, bias, stride,
                                padding, output_padding, groups, dilation):
    """
       Applies a hypercomplex trasposed convolution to the incoming data:
       (a, b) (c, d) = (ac -d"b, da + bc")
       d" => d conjugate
       """

    num_components = len(weights)
    comp_mat = get_comp_mat(num_components=num_components)

    cat_kernel_hypercomplex_i = []
    for comp_i in comp_mat:
        kernel_hypercomplex_i = []
        for idx, ii in enumerate(comp_i):
            itr = np.abs(ii)
            sign = np.sign(ii) if np.sign(ii) != 0 else 1
            kernel_hypercomplex_i.append(sign * weights[itr])
        cat_kernel_hypercomplex_i.append(torch.cat(kernel_hypercomplex_i, dim=1))

    cat_kernels_hypercomplex = torch.cat(cat_kernel_hypercomplex_i, dim=0)
    if input.dim() == 3:
        convfunc = F.conv_transpose1d
    elif input.dim() == 4:
        convfunc = F.conv_transpose2d
    elif input.dim() == 5:
        convfunc = F.conv_transpose3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, cat_kernels_hypercomplex, bias, stride, padding, output_padding, groups, dilation)


def hypercomplex_linear(input, weights, bias=True):
    """
    Applies a octonion linear transformation to the incoming data:

    It is important to notice that the forward phase of a QNN is defined
    as W * Inputs (with * equal to the Hamilton product). The constructed
    cat_kernels_8_octonion is a modified version of the octonion representation
    so when we do torch.mm(Input,W) it's equivalent to W * Inputs.

    """
    
    num_components = len(weights)
    comp_mat = get_comp_mat(num_components=num_components)

    cat_kernel_hypercomplex_i = []
    for comp_i in comp_mat:
        kernel_hypercomplex_i = []
        for idx, ii in enumerate(comp_i):
            itr = np.abs(ii)
            sign = np.sign(ii) if np.sign(ii) != 0 else 1
            kernel_hypercomplex_i.append(sign * weights[itr])
        cat_kernel_hypercomplex_i.append(torch.cat(kernel_hypercomplex_i, dim=0))

    cat_kernels_hypercomplex = torch.cat(cat_kernel_hypercomplex_i, dim=1)
    
    if input.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, input, cat_kernels_hypercomplex)
        else:
            return torch.mm(input, cat_kernels_hypercomplex)
    else:
        output = torch.matmul(input, cat_kernels_hypercomplex)
        if bias is not None:
            return output + bias
        else:
            return output


# Custom AUTOGRAD for lower VRAM consumption
class HyperLinearFunction(torch.autograd.Function):

    # @staticmethod
    # def forward(ctx, input, r_weight, i_weight, j_weight, k_weight,
    #             e_weight, l_weight, m_weight, n_weight, bias=None):
    @staticmethod
    def forward(ctx, input, bias, *weights):
        # ctx.save_for_backward(input, r_weight, i_weight, j_weight, k_weight,
        #                       e_weight, l_weight, m_weight, n_weight, bias)
        # ctx.save_for_backward(input, weights, bias)
        # weights = [weight_in for weight_in in weights_in]
        ctx.save_for_backward(input, bias, *weights)
        num_components = len(weights)
        # print(num_components)
        # print(weights)
        # print(bias)
        check_input(input, num_components=num_components)

        comp_mat = get_comp_mat(num_components=num_components)

        cat_kernel_hypercomplex_i = []
        for comp_i in comp_mat:
            kernel_hypercomplex_i = []
            for idx, ii in enumerate(comp_i):
                itr = np.abs(ii)
                sign = np.sign(ii) if np.sign(ii) != 0 else 1
                kernel_hypercomplex_i.append(sign * weights[itr])
            cat_kernel_hypercomplex_i.append(torch.cat(kernel_hypercomplex_i, dim=0))

        cat_kernels_hypercomplex = torch.cat(cat_kernel_hypercomplex_i, dim=1)

        if input.dim() == 2:
            if bias is not None:
                return torch.addmm(bias, input, cat_kernels_hypercomplex)
            else:
                return torch.mm(input, cat_kernels_hypercomplex)
        else:
            output = torch.matmul(input, cat_kernels_hypercomplex)
            if bias is not None:
                return output + bias
            else:
                return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        saved_tensors = ctx.saved_tensors
        input = saved_tensors[0]
        bias = saved_tensors[1]
        weights = saved_tensors[2:]

        num_components = len(weights)
        grad_input = grad_bias = None
        grad_weights = [None for _ in range(num_components)]

        comp_mat = get_comp_mat(num_components=num_components)

        cat_weight_hypercomplex_i = []
        for comp_i in comp_mat:
            weight_hypercomplex_i = []
            for idx, ii in enumerate(comp_i):
                itr = np.abs(ii)
                sign = np.sign(ii) if np.sign(ii) != 0 else 1
                weight_hypercomplex_i.append(sign * weights[itr])
            cat_weight_hypercomplex_i.append(torch.cat(weight_hypercomplex_i, dim=0))

        weight_mat_T = Variable(torch.cat(cat_weight_hypercomplex_i, dim=1).permute(1, 0), requires_grad=False)

        inputs = [get_c(input, component=component, num_components=num_components) for
                  component in range(num_components)]
        cat_input_hypercomplex_i = []
        for comp_i in comp_mat:
            input_hypercomplex_i = []
            for idx, ii in enumerate(comp_i):
                itr = np.abs(ii)
                sign = np.sign(ii) if np.sign(ii) != 0 else 1
                input_hypercomplex_i.append(sign * inputs[itr])
            cat_input_hypercomplex_i.append(torch.cat(input_hypercomplex_i, dim=0))

        input_mat = Variable(torch.cat(cat_input_hypercomplex_i, dim=1), requires_grad=False)

        grad_outputs = [get_c(grad_output, component=component, num_components=num_components) for
                  component in range(num_components)]
        cat_input_hypercomplex_i = []
        for comp_i in comp_mat.T:
            grad_hypercomplex_i = []
            for idx, ii in enumerate(comp_i):
                itr = np.abs(ii)
                sign = np.sign(ii) if np.sign(ii) != 0 else 1
                grad_hypercomplex_i.append(sign * grad_outputs[itr])
            cat_input_hypercomplex_i.append(torch.cat(grad_hypercomplex_i, dim=1))

        grad_mat = torch.cat(cat_input_hypercomplex_i, dim=0)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_mat_T)
        if ctx.needs_input_grad[2]:
            grad_weight = grad_mat.permute(1, 0).mm(input_mat).permute(1, 0)
            unit_size_x = weights[0].size(0)
            unit_size_y = weights[0].size(1)
            grad_weights = [grad_weight.narrow(0, 0, unit_size_x).narrow(1, component * unit_size_y, unit_size_y)
                            for component in range(num_components)]
            # grad_weight_r = grad_weight.narrow(0, 0, unit_size_x).narrow(1, 0, unit_size_y)
            # grad_weight_i = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y, unit_size_y)
            # grad_weight_j = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y * 2, unit_size_y)
            # grad_weight_k = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y * 3, unit_size_y)
            # grad_weight_e = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y * 4, unit_size_y)
            # grad_weight_l = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y * 5, unit_size_y)
            # grad_weight_m = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y * 6, unit_size_y)
            # grad_weight_n = grad_weight.narrow(0, 0, unit_size_x).narrow(1, unit_size_y * 7, unit_size_y)
        if ctx.needs_input_grad[1]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return (grad_input, grad_bias, *grad_weights)


#
# PARAMETERS INITIALIZATION
#
def unitary_init(in_features, out_features, rng, kernel_size=None, criterion='he', num_components=8):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v = [np.random.uniform(-1.0, 1.0, number_of_weights) for component in range(num_components)]

    # Unitary Hypercomplex
    for i in range(0, number_of_weights):
        # norm = np.sqrt(v_r[i] ** 2 + v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2 +
        #                v_e[i] ** 2 + v_l[i] ** 2 + v_m[i] ** 2 + v_n[i] ** 2) + 0.0001
        #
        norm = np.sqrt(sum([v_[i] ** 2 for v_ in v])) + 0.0001
        for idx in range(len(v)):
            v[idx][i] /= norm
        # v_r[i] /= norm
        # v_i[i] /= norm
        # v_j[i] /= norm
        # v_k[i] /= norm
        # v_e[i] /= norm
        # v_l[i] /= norm
        # v_m[i] /= norm
        # v_n[i] /= norm
    weights = [v_.reshape(kernel_shape) for v_ in v]

    return weights


def random_init(in_features, out_features, rng, kernel_size=None, criterion='glorot', num_components=8):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2 * fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    weights = [np.random.uniform(-1.0, 1.0, number_of_weights).reshape(kernel_shape) for component in
               range(num_components)]

    return weights


def hypercomplex_init(in_features, out_features, rng, kernel_size=None, criterion='glorot', num_components=8):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    # rng = RandomState(np.random.randint(1, 1234))

    # Generating randoms and purely imaginary hyper(complex) :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(num_components, loc=0, scale=s, size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v = [np.random.uniform(-1.0, 1.0, number_of_weights) for component in range(num_components-1)]

    # Purely imaginary hyper(complex) unitary
    for i in range(0, number_of_weights):
        norm = np.sqrt(sum(v[j][i]**2 for j in range(num_components-1)) + 0.0001)
        for j in range(num_components-1):
            v[j][i] /= norm
    v = [v_c.reshape(kernel_shape) for v_c in v]

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight = [torch.from_numpy(modulus * np.cos(phase)).type(torch.FloatTensor)]
    weight.extend([torch.from_numpy(modulus * v_c*np.sin(phase)).type(torch.FloatTensor) for v_c in v])

    return weight


def create_dropout_mask(dropout_p, size, rng, as_type, operation='linear'):
    if operation == 'linear':
        mask = rng.binomial(n=1, p=1 - dropout_p, size=size)
        return Variable(torch.from_numpy(mask).type(as_type))
    else:
        raise Exception("create_dropout_mask accepts only 'linear'. Found operation = "
                        + str(operation))


def affect_init(weights, init_func, rng, init_criterion):
    # if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
    #         r_weight.size() != k_weight.size() or r_weight.size() != e_weight.size() or \
    #         r_weight.size() != l_weight.size() or r_weight.size() != m_weight.size() or \
    #         r_weight.size() != n_weight.size():
    #     raise ValueError('The real and imaginary weights '
    #                      'should have the same size . Found: r:'
    #                      + str(r_weight.size()) + ' i:'
    #                      + str(i_weight.size()) + ' j:'
    #                      + str(j_weight.size()) + ' k:'
    #                      + str(k_weight.size()) + ' e:'
    #                      + str(e_weight.size()) + ' l:'
    #                      + str(l_weight.size()) + ' m:'
    #                      + str(m_weight.size()) + ' n:'
    #                      + str(n_weight.size()))
    weights_size = [weight.size() for weight in weights]
    if len(set(weights_size)) != 1:
        raise ValueError(f"The real and imaginary weights should have the same size . Found: {weights_size}")
    # elif r_weight.dim() != 2:
    #     raise Exception('affect_init accepts only matrices. Found dimension = '
    #                     + str(r_weight.dim()))
    elif weights[0].dim() != 2:
        raise Exception('affect_init accepts only matrices. Found dimension = '
                        + str(weights[0].dim()))
    kernel_size = None
    # r, i, j, k, e, l, m, n = init_func(r_weight.size(0), r_weight.size(1), rng, kernel_size, init_criterion)
    weights_ = init_func(weights[0].size(0), weights[0].size(1), rng, kernel_size, init_criterion, len(weights))
    # r, i, j, k, e, l, m, n = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k), \
    #                          torch.from_numpy(e), torch.from_numpy(l), torch.from_numpy(m), torch.from_numpy(n)
    # weights_ = [torch.from_numpy(weight) for weight in weights_]
    # for idx, weight in weights:
    #     weight.data = weights_[idx].type_as(weight.data)
    num_components = len(weights)
    for idx in range(num_components):
        weights[idx].data = weights_[idx].type_as(weights[idx].data)
    # r_weight.data = r.type_as(r_weight.data)
    # i_weight.data = i.type_as(i_weight.data)
    # j_weight.data = j.type_as(j_weight.data)
    # k_weight.data = k.type_as(k_weight.data)
    # e_weight.data = e.type_as(e_weight.data)
    # l_weight.data = l.type_as(l_weight.data)
    # m_weight.data = m.type_as(m_weight.data)
    # n_weight.data = n.type_as(n_weight.data)


def affect_init_conv(weights, kernel_size, init_func, rng,
                     init_criterion):
    weights_size = [weight.size() for weight in weights]
    if len(set(weights_size)) != 1:
        raise ValueError(f"The real and imaginary weights should have the same size . Found: {weights_size}")

    elif 2 >= weights[0].dim():
        raise Exception('affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
                        + str(weights[0].dim()))
    num_components = len(weights)
    weights_ = init_func(weights[0].size(1), weights[0].size(0),
                        rng=rng, kernel_size=kernel_size,
                        criterion=init_criterion, num_components=num_components)
    # weights_ = [torch.from_numpy(weight) for weight in weights_]

    # for idx, weight in weights:
    #     weight.data = weights_[idx].type_as(weight.data)
    for idx in range(num_components):
        weights[idx].data = weights_[idx].type_as(weights[idx].data)


def get_kernel_and_weight_shape_old(operation, in_channels, out_channels, kernel_size):
    if operation == 'convolution1d':
        if type(kernel_size) is not int:
            raise ValueError(
                """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                must be integer in the case. Found kernel_size = """ + str(kernel_size)
            )
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + tuple((ks,))
    else:  # in case it is 2d or 3d.
        if operation == 'convolution2d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size)
        elif operation == 'convolution3d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size, kernel_size)
        elif type(kernel_size) is not int:
            if operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                    must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                )
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                    must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape

def get_kernel_and_weight_shape(operation, in_channels, out_channels, kernel_size, groups):
    if operation == 'convolution1d':
        if type(kernel_size) is not int:
            raise ValueError(
                """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                must be integer in the case. Found kernel_size = """ + str(kernel_size)
            )
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels // groups) + tuple((ks,))
    else:  # in case it is 2d or 3d.
        if operation == 'convolution2d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size)
        elif operation == 'convolution3d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size, kernel_size)
        elif type(kernel_size) is not int:
            if operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                    must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                )
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                    must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels // groups) + (*ks,)
    return ks, w_shape
