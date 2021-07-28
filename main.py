import pathlib
import sys
import os

module_dir = str(pathlib.Path(os.getcwd()))
sys.path.append(module_dir)

import re
import argparse
import warnings

import numpy as np
import pandas as pd
import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import torch

from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import CSVLogger

import config as cf
from model_pl import Model
from models import HyperSwinEncoderDecoder3D

from utils.w4c_dataloader import create_dataset
from train_utils import model_summary 


def get_held_out_params(params):
    held_out_params = params
    # print(params.keys())
    old_path = held_out_params['data_path']
    paths = re.split('/|\\\\', old_path)
    paths[-2] += '-heldout'
    new_path = f'{os.sep}'.join(paths)
    held_out_params['data_path'] = new_path
    return held_out_params


class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """

    def __init__(self, params, training_params, args):
        super().__init__()
        self.params = params
        self.training_params = training_params
        self.args = args
        self.train_dims = None
        self.precision = args.precision if hasattr(args, 'precision') else 16
        self.populate_mask = args.populate_mask if hasattr(args, 'populate_mask') else False

        self.train = self.val = self.predict = self.held_out = None
        self.all_regions = self.core_regions = None

    def setup(self):
        if self.params['use_all_region']:
            train_datasets = []
            val_datasets = []
            predict_datasets = []
            held_out_datasets = []
            core_regions = ['R1', 'R2', 'R3']
            all_regions = [f"R{i + 1}" for i in range(6)]
            if self.args.competition == 'ieee-bd':
                core_regions.extend(['R7', 'R8'])
                all_regions.extend([f"R{i + 1}" for i in range(6, 11)])
            self.all_regions = all_regions
            self.core_regions = core_regions
            for region_id in core_regions:
                params_i = cf.get_params(region_id=region_id, competition=self.args.competition,
                                         collapse_time=self.args.collapse_time, use_static=self.args.use_static,
                                         use_all_variables=self.args.use_all_variables,
                                         use_cloud_type=self.args.use_cloud_type, use_time_slot=self.args.use_time_slot)
                train_datasets.append(create_dataset('training', params_i['data_params'], precision=self.precision,
                                                     populate_mask=self.populate_mask))
                val_datasets.append(create_dataset('validation', params_i['data_params'], precision=self.precision,
                                                   populate_mask=self.populate_mask))

            for region_id in all_regions:
                params_i = cf.get_params(region_id=region_id, competition=self.args.competition,
                                         collapse_time=self.args.collapse_time, use_static=self.args.use_static,
                                         use_all_variables=self.args.use_all_variables,
                                         use_cloud_type=self.args.use_cloud_type, use_time_slot=self.args.use_time_slot)
                predict_datasets.append(create_dataset('test', params_i['data_params'], precision=self.precision,
                                                       populate_mask=self.populate_mask))
                if self.args.held_out:  # using held-out data
                    held_out_params = get_held_out_params(params_i['data_params'])
                    held_out_datasets.append(create_dataset('test', held_out_params, precision=self.precision,
                                                            populate_mask=self.populate_mask))

            self.train = ConcatDataset(train_datasets)
            self.val = ConcatDataset(val_datasets)
            self.predict = ConcatDataset(predict_datasets)
            if self.args.held_out:  # using held-out data
                self.held_out = ConcatDataset(held_out_datasets)
        else:
            self.train = create_dataset('training', self.params, precision=self.precision,
                                        populate_mask=self.populate_mask)
            self.val = create_dataset('validation', self.params, precision=self.precision,
                                      populate_mask=self.populate_mask)
            self.predict = create_dataset('test', self.params, precision=self.precision,
                                          populate_mask=self.populate_mask)
            if self.args.held_out:  # using held-out data
                held_out_params = get_held_out_params(self.params)
                self.held_out = create_dataset('test', held_out_params, precision=self.precision,
                                               populate_mask=self.populate_mask)


        self.train_dims = self.train.__len__()

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(dataset,
                        batch_size=self.training_params['batch_size'], num_workers=self.training_params['n_workers'],
                        shuffle=shuffle, pin_memory=pin)
        return dl

    def train_dataloader(self):
        ds = self.train  # create_dataset('training', self.params)
        return self.__load_dataloader(ds, shuffle=True, pin=True)

    def val_dataloader(self):
        val_loader = self.__load_dataloader(self.val, shuffle=False, pin=True)
        predict_loader = self.__load_dataloader(self.predict, shuffle=False, pin=True)
        return [val_loader, predict_loader]

    def test_dataloader(self):
        if self.args.held_out:  # using held-out data
            predict_loader = self.__load_dataloader(self.held_out, shuffle=False, pin=True)
        else:
            predict_loader = self.__load_dataloader(self.predict, shuffle=False, pin=True)
        return predict_loader


def print_training(params):
    """ print pre-training info """

    print(f'Extra variables: {params["extra_data"]} | spatial_dim: {params["spatial_dim"]} ',
          f'| collapse_time: {params["collapse_time"]} | in channels depth: {params["depth"]} | len_seq_in: {params["len_seq_in"]}')


def modify_options(options, n_params):
    filename = '_'.join(
        [f"{item}" for item in ('ALL' if options.use_all_region else options.region, options.net_type, 'swinencoder3d',
                                int(n_params))])
    options.filename = options.name or filename  # to account for resuming from a previous state

    options.versiondir = os.path.join(options.log_dir, options.filename, options.time_code)
    os.makedirs(options.versiondir, exist_ok=True)
    readme_file = os.path.join(options.versiondir, 'options.csv')
    args_dict = vars(argparse.Namespace(**{'modelname': options.filename, 'num_params': n_params}, **vars(options)))
    args_df = pd.DataFrame([args_dict])
    if os.path.exists(readme_file):
        args_df.to_csv(readme_file, mode='a', index=False, header=False)
    else:
        args_df.to_csv(readme_file, mode='a', index=False)

    return options


def save_options(options, n_params):
    options.versiondir = os.path.join(options.log_dir, options.filename, options.time_code)
    os.makedirs(options.versiondir, exist_ok=True)
    readme_file = os.path.join(options.versiondir, 'options.csv')
    args_dict = vars(argparse.Namespace(**{'modelname': options.filename, 'num_params': n_params}, **vars(options)))
    args_df = pd.DataFrame([args_dict])
    if os.path.exists(readme_file):
        args_df.to_csv(readme_file, mode='a', index=False, header=False)
    else:
        args_df.to_csv(readme_file, mode='a', index=False)
    return options


def get_trainer(options):
    """ get the trainer, modify here it's options:
        - save_top_k
        - max_epochs
     """
    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # should be found in logs
        patience=20,
        strict=False,  # will act as disabled if monitor not found
        verbose=False,
        mode='min'
    )

    logger = CSVLogger(save_dir=options.log_dir,
                       name=options.filename,
                       version=options.time_code,
                       ) 

    resume_from_checkpoint = None
    if options.name and options.time_code:
        checkpoint_dir = os.path.join(options.versiondir, 'checkpoints')
        if options.initial_epoch == -1:
            checkpoint_name = 'last.ckpt'
        else:
            format_str = f"epoch={options.initial_epoch:02g}"
            checkpoint_names = os.listdir(checkpoint_dir)
            checkpoint_name = checkpoint_names[[t.startswith(format_str) for t in checkpoint_names].index(True)]
        resume_from_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)


    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3,
                                          save_last=True, verbose=False,
                                          filename='{epoch:02d}-{val_loss:.6f}')

    callbacks = [lr_monitor, checkpoint_callback, early_stop_callback]

    trainer = pl.Trainer(gpus=options.gpus,
                         max_epochs=options.epochs,
                         progress_bar_refresh_rate=10,
                         deterministic=True,
                         gradient_clip_val=1,  # to clip gradient value and prevent exploding gradient
                         gradient_clip_algorithm='value',
                         default_root_dir=os.path.dirname(options.log_dir),
                         callbacks=callbacks,
                         profiler='simple',
                         sync_batchnorm=True,
                         num_sanity_val_steps=0,
                         # accelerator='ddp',
                         logger=logger,
                         resume_from_checkpoint=resume_from_checkpoint,
                         num_nodes=1,
                         precision=options.precision if hasattr(options, 'precision') else 16,
                         )

    return trainer


def train(region_id, mode, options=None):
    """ main training/evaluation method
    """

    # some needed stuffs
    warnings.filterwarnings("ignore")

    params = cf.get_params(region_id=region_id, competition=options.competition, collapse_time=options.collapse_time,
                           use_static=options.use_static, use_all_variables=options.use_all_variables,
                           use_cloud_type=options.use_cloud_type, use_time_slot=options.use_time_slot)
    # print(params['data_params'])
    params['data_params']['use_all_region'] = options.use_all_region
    options = argparse.Namespace(**{**vars(options), **params['model_params'], **params['data_params']}) \
        if options else argparse.Namespace(**params)

    pl.seed_everything(options.manual_seed, workers=True)
    torch.manual_seed(options.manual_seed)
    torch.cuda.manual_seed_all(options.manual_seed)

    training_params = {'batch_size': options.batch_size,
                       'n_workers': options.workers  # 8
                       }

    # ------------
    # Data and model params
    # ------------
    data = DataModule(params['data_params'], training_params, options)
    data.setup()

    # add other depending args
    options.train_dims = data.train_dims
    options.core_regions = data.core_regions
    options.all_regions = data.all_regions

    # let's load model for printing structure
    model = HyperSwinEncoderDecoder3D(options)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x_all = torch.rand(1, params['data_params']['len_seq_in'], params['data_params']['depth'], 256, 256)
    _ = model_summary(model, x_all, print_summary=True, max_depth=1)
    del model, x_all
    
    # ------------
    # trainer
    # ------------
    options = modify_options(options, n_params)
    trainer = get_trainer(options)
    print_training(params['data_params'])

    # ------
    # Model
    # -----
    checkpoint_path = trainer.resume_from_checkpoint
    if checkpoint_path is not None:
        model = Model.load_from_checkpoint(checkpoint_path)
    else:
        model = Model(options, **params['data_params'])



    print(options)
    # ------------
    # train & final validation
    # ------------
    if mode == 'train':
        print("-----------------")
        print("-- TRAIN MODE ---")
        print("-----------------")
        trainer.fit(model, datamodule=data)
    # elif mode == 'val':
    #     print("-----------------")
    #     print("-- Validation only for metric collection---")
    #     print("-----------------")
    #     trainer.validate(model, datamodule=data)
    else:
        print("-----------------")
        print("--- TEST MODE ---")
        print("-----------------")
        trainer.test(model, datamodule=data)
    

def set_parser(parent_parser):
    """ set custom parser """
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("-g", "--gpus", type=str, required=False, default='0',
                        help="specify a gpu ID. 0 as default")
    parser.add_argument("-r", "--region", type=str, required=False, default='R1',
                        help="region_id to load data from. R1 as default")
    parser.add_argument("-a", "--use_all_region", type=bool, required=False, default=True,
                        help="use all region")
    parser.add_argument("-m", "--mode", type=str, required=False, default='test',
                        help="choose mode: train (default) / test")
    parser.add_argument("-ho", "--held-out", type=bool, required=False, default=False,
                        help="are we using held-out dataset for the 'test'")
    
    return parser


def add_main_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--competition', default='stage-1', help='competition name', choices=['stage-1', 'ieee-bd'])
    parser.add_argument('--collapse-time', type=bool, default=False, help='collapse time axis')
    parser.add_argument('--use_static', type=bool, default=False, help='use static variable (Default: True)')
    parser.add_argument('--use_time_slot', type=bool, default=False, help='use time slots (Default: True)')
    parser.add_argument('--use_cloud_type', type=bool, default=False,
                        help='use cloud type variables. [Only when all variables are used] (Default: False)')
    parser.add_argument('--use_all_variables', type=bool, default=False,
                        help='use available variables for the variable types used (Default: False)')
    parser.add_argument('--populate_mask', type=bool, default=True, help='use mask to work only on unmasked data')

    parser.add_argument('--precision', type=int, default=32, help='precision to use for training', choices=[16, 32])
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    
    parser.add_argument('--manual-seed', default=0, type=int, help='manual global seed')
    parser.add_argument('--log-dir', default='logs', help='base directory to save logs')
    parser.add_argument('--model-dir', default='', help='base directory to save logs')
    parser.add_argument('--name', default='ALL_real_swinencoder3d_688080',
                        help='identifier for model if already exist')
    parser.add_argument('--time-code', default='20210630T224355',
                        help='identifier for model if already exist')
    parser.add_argument('--initial-epoch', type=int, default=58,
                        help='number of epochs done (-1 == last)')
    parser.add_argument('--memory_efficient', type=bool, default=True, help='memory_efficient')

    return parser


def get_time_code():
    time_now = [f"{'0' if len(x) < 2 else ''}{x}" for x in np.array(time.localtime(), dtype=str)][:6]
    if os.path.exists('t.npy'):
        time_before = np.load('t.npy')   # .astype(np.int)
        if abs(int(''.join(time_before)) - int(''.join(time_now))) < 70:
            time_now = time_before
        else:
            np.save('t.npy', time_now)
    else:
        np.save('t.npy', time_now)
    time_now = ''.join(time_now[:3]) + 'T' + ''.join(time_now[3:])
    return time_now


def main():
    parser = argparse.ArgumentParser(description="Weather4Cast Arguments")
    parser = set_parser(parser)
    parser = add_main_args(parser)
    parser = Model.add_model_specific_args(parser)
    options = parser.parse_args()

    options.region = options.region.upper()
    options.workers = 6  
    
    time_code = get_time_code()
    options.time_code = options.time_code or time_code  # to account for resuming from a previous state

    train(options.region, options.mode, options)


if __name__ == "__main__":
    main()
    """ examples of usage:

    - a.1) train from scratch
    python main.py --gpus 0 --region R1

    - a.2) fine tune a model from a checkpoint
    python main.py --gpu_id 1 --region R1 --mode train --name ALL_real_swinencoder3d_688080 --time-code 20210630T224355 --initial-epoch 60
    
    - b.1) evaluate an untrained model (with random weights)
    python main.py --gpus 0 --region R1 --mode test

    - b.2) evaluate a trained model from a checkpoint
    python main.py --gpu_id 1 --region R1 --mode test --name ALL_real_swinencoder3d_688080 --time-code 20210630T224355 --initial-epoch 60
    
    """