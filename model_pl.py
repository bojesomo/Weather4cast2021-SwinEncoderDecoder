import numpy as np

from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch
import os

import argparse
from argparse import ArgumentParser
from models import optimizer_dict, HyperSwinEncoderDecoder3D
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import utils.data_utils as data_utils


class Model(pl.LightningModule):
    def __init__(self, args,
                 extra_data: str, depth: int, height: int,
                 width: int, len_seq_in: int, len_seq_out: int, bins_to_predict: int,
                 seq_mode: str, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.args = args

        self.net = HyperSwinEncoderDecoder3D(args)
        self.extra_data = extra_data
        self.depth = depth
        self.height = height
        self.width = width
        self.len_seq_in = len_seq_in
        self.len_seq_out = len_seq_out
        self.bins_to_predict = bins_to_predict
        self.seq_mode = seq_mode

        self.loss_fn = F.mse_loss
        
        self.core_dir = ''
        self.transfer_dir = ''

    def forward(self, x, mask=None):
        return self.net(x)

    def _compute_loss(self, y_hat, y, agg=True, mask=None):
        if mask is not None:
            y_hat = y_hat.flatten()[~mask.flatten()]
            y = y.flatten()[~mask.flatten()]
            
        if agg:
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')
        return loss

    @staticmethod
    def process_batch(batch):
        # in_seq, out_seq, metadata = batch
        return batch

    def training_step(self, batch, batch_idx, phase='train'):

        x, y, metadata = self.process_batch(batch)
        y_hat = self.forward(x)
        loss = self._compute_loss(y_hat, y, mask=metadata['out'].get('masks'))
        self.log(f'{phase}_loss', loss, on_epoch=True)
        return loss

    def create_inference_dirs(self):
        for region_id in ['R1', 'R2', 'R3']:
            os.makedirs(os.path.join(self.core_dir, region_id, 'test'), exist_ok=True)
        for region_id in ['R4', 'R5', 'R6']:
            os.makedirs(os.path.join(self.transfer_dir, region_id, 'test'), exist_ok=True)
        
    def on_validation_epoch_start(self):
        epoch_dir = os.path.join(self.args.versiondir, 'inference', f"epoch={self.current_epoch}")
        self.core_dir = os.path.join(epoch_dir, f'core_{self.current_epoch}')
        self.transfer_dir = os.path.join(epoch_dir, f'transfer_{self.current_epoch}')
        self.create_inference_dirs()

    def on_test_epoch_start(self):
        ckpt_name = str(os.path.basename(self.trainer.resume_from_checkpoint))
        folder_name = ckpt_name.split('.')[0].split('-')[0].split('=')[-1]
        epoch_dir = os.path.join(self.args.versiondir, 'test', f"epoch={folder_name}")
        self.core_dir = os.path.join(epoch_dir, f'core_{folder_name}')
        self.transfer_dir = os.path.join(epoch_dir, f'transfer_{folder_name}')
        self.create_inference_dirs()

    def save_prediction(self, y_hat, metadata, batch_idx, loader_idx):
        y_hat = torch.reshape(y_hat, (-1, self.len_seq_out, len(self.args.target_vars), self.height, self.width))
        y_hat = y_hat.data.cpu().numpy()

        for idx, (region_id, day_in_year) in enumerate(zip(metadata['out']['region_id'],
                                                           metadata['out']['day_in_year'][0])):
            if region_id in ['R1', 'R2', 'R3']:  # 'w4c-core-stage-1'
                save_path = os.path.join(self.core_dir, region_id, 'test', f"{day_in_year}.h5")
            else:  # 'w4c-transfer-learning-stage-1'
                save_path = os.path.join(self.transfer_dir, region_id, 'test', f"{day_in_year}.h5")
            y = data_utils.postprocess_fn(y_hat[idx], self.args.target_vars,
                                          self.args.preprocess['source'])
            data_utils.write_data(y, save_path)

    def validation_step(self, batch, batch_idx, loader_idx, phase='val'):

        x, y, metadata = self.process_batch(batch)
        y_hat = self.forward(x)
        if loader_idx == 0:  # for validation loader only
            loss = self._compute_loss(y_hat, y, mask=metadata['out'].get('masks'))
            self.log(f'{phase}_loss', loss, prog_bar=True)  # , logger=True)
        else:  # for prediction
            self.save_prediction(y_hat, metadata, batch_idx, loader_idx)

    def test_step(self, batch, batch_idx):  # , phase='test'):

        x, y, metadata = self.process_batch(batch)
        y_hat = self.forward(x)
        self.save_prediction(y_hat, metadata, batch_idx, loader_idx=0)

    def configure_optimizers(self):
        other_args = {}
        print(self.args)
        if self.args.optimizer == 'sgd':
            other_args = {'lr': self.args.lr, 'momentum': self.args.momentum,
                          'weight_decay': self.args.weight_decay, 'nesterov': True}
            optimizer = optimizer_dict[self.args.optimizer](self.net.parameters(), **other_args)
            t_max = math.ceil(self.args.epochs * self.args.train_dims /
                              (self.args.batch_size * len(
                                  self.args.gpus.split(','))))  # self.args.epochs * self.args.train_dims
            scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=0),
                         'interval': 'step',  # or 'epoch'
                         }
        elif self.args.optimizer == 'adam':
            other_args = {'lr': self.args.lr, 'eps': self.args.epsilon,
                          'betas': (self.args.beta_1, self.args.beta_2),
                          'weight_decay': self.args.weight_decay}
            optimizer = optimizer_dict[self.args.optimizer](self.net.parameters(), **other_args)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, cooldown=0, min_lr=1e-7)
        elif self.args.optimizer == 'swats':
            other_args = {'lr': self.args.lr,
                          'weight_decay': self.args.weight_decay,
                          'nesterov': True
                          }
            optimizer = optimizer_dict[self.args.optimizer](self.net.parameters(), **other_args)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, cooldown=0, min_lr=1e-7)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--net_type', default='real', help='type of network',
                            choices=['sedenion', 'real', 'complex', 'quaternion', 'octonion'])
        parser.add_argument('--patch_size', type=int, default=2, help='patch size to use in swin transfer')
        parser.add_argument('--nb_layers', type=int, default=4, help='depth of resnet blocks (default: 1)')
        parser.add_argument('--sf', type=int, default=16 * 1,
                            help='number of feature maps/embedding dimension (default: 16*1)')
        parser.add_argument('--stages', type=int, default=3,
                            help='number of encoder stages (<1 means infer) (default:0)')
        parser.add_argument('--classifier_activation', default='sigmoid',
                            help='hidden layer activation (default: hardtanh)')  # sigmoid?
        parser.add_argument('--modify_activation', type=bool, default=True,
                            help='modify the range of hardtanh activation')
        parser.add_argument('--inplace_activation', type=bool, default=True, help='inplace activation')
        parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')

        parser.add_argument('--optimizer', default='adam', help='optimizer to train with',
                            choices=['sgd', 'adam', 'swats'])
        parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum term for sgd')
        parser.add_argument('--beta_1', default=0.9, type=float, help='beta_1 term for adam')
        parser.add_argument('--beta_2', default=0.999, type=float, help='beta_2 term for adam')
        parser.add_argument('--epsilon', default=1e-8, type=float, help='epsilon term for adam')
        parser.add_argument('--weight_decay', default=1e-6, type=float,
                            help='weight decay for regularization (default: 1e-6)')

        return parser

