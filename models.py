# coding:utf-8
##########################################################
# pytorch v1.0
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# April 2021
##########################################################

import os
import sys
sys.path.extend(os.getcwd())

from torch import optim
import numpy as np
import warnings
from swin_transformer3d import SwinEncoderDecoderTransformer3D
import torch_optimizer as extra_optim
from functools import partial

optimizer_dict = {'adadelta': optim.Adadelta,
                  'adagrad': optim.Adagrad,
                  'adam': optim.Adam,
                  'adamw': optim.AdamW,
                  'swats': extra_optim.SWATS,
                  'sparse_adam': optim.SparseAdam,
                  'adamax': optim.Adamax,
                  'asgd': optim.ASGD,
                  'sgd': optim.SGD,
                  'rprop': optim.Rprop,
                  'rmsprop': optim.RMSprop,
                  'lbfgs': optim.LBFGS}
n_div_dict = {'sedenion': 16,
              'octonion': 8,
              'quaternion': 4,
              'complex': 2,
              'real': 1}


class HyperSwinEncoderDecoder3D(SwinEncoderDecoderTransformer3D):
    def __init__(self, args):
        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        heads_ = 8
        n_multiples_in = int(n_divs * np.ceil(args.len_seq_in / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if args.sf < embed_dim:
            warnings.warn(f"args.sf = {args.sf} < embed_dim used [{embed_dim}]")
        if args.sf > embed_dim:
            embed_dim = int(embed_dim * np.ceil(args.sf / embed_dim))

        super().__init__(depths=tuple([args.nb_layers] * args.stages),
                         num_heads=tuple([8] * args.stages),
                         out_chans=args.len_seq_out,
                         in_chans=args.len_seq_in,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(args.height, args.width),
                         in_depth=args.depth, out_depth=len(args.target_vars),
                         n_divs=n_divs,
                         drop_rate=args.dropout,
                         patch_size=(1, *([args.patch_size] * 2))
                         )
