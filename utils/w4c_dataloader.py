# Author: Pedro Herruzo
# Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
# IARAI licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from torch.utils.data import Dataset
import utils.data_utils as data_utils
from utils.context_variables import get_static
import os

class NWCSAF(Dataset):
    
    def __init__(self, data_split, products, input_vars, target_vars,
                 spatial_dim, collapse_time=True, 
                 len_seq_in=4, len_seq_out=32, bins_to_predict=32, day_bins=96,
                 region_id=None,  preprocess=None,
                 crop_in=None, crop_out=None,
                 extra_data='', crop_static=None, static_paths=None,
                 data_path='', control_params=None,
                 train_splits='splits.csv', 
                 test_splits='test_split.json', 
                 black_list_path='blacklist.json', precision=16, populate_mask=False, **kwargs):
        self.precision = {16: np.float16, 32: np.float32}[precision]
        self.channel_dim = 1  # specifies the dimension to concat multiple channels/variables

        # data dimensions
        self.spatial_dim = spatial_dim
        self.collapse_time = collapse_time
        self.len_seq_in = len_seq_in
        self.len_seq_out = len_seq_out
        self.bins_to_predict = bins_to_predict
        self.day_bins = day_bins
        self.day_strings = ['{}{}{}{}00'.format('0'*bool(i<10), i, '0'*bool(j<10), j) for i in np.arange(0, 24, 1) for j in np.arange(0, 60, 15)]
        
        # type of data & processing variables
        self.products = products
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.region_id = region_id
        self.preprocess = preprocess
        self.populate_mask = populate_mask
        self.crop_in, self.crop_out = crop_in, crop_out
        self.control_params = control_params
        
        # load extra variables if any 
        self.extra_data, self.static_tensor, self.static_desc = [], [], []
        if extra_data != '':
            self.extra_data = extra_data.split('-')
            self.static_tensor, self.static_desc = get_static(self.extra_data, self.len_seq_in, static_paths, 
                                                              crop=crop_static, channel_dim=self.channel_dim)
        
        # data splits to load (training/validation/test)
        self.data_path = data_path + f'/{data_split}'
        self.data_split = data_split
        self.day_paths, self.test_splits = data_utils.read_splits(train_splits, test_splits)
        
    
        # prepare all elements to load - batch idx will use the object 'self.idx'
        if self.data_split != 'test':
            self.day_paths = self.day_paths[self.day_paths.split == self.data_split].reset_index()
            # self.idxs = data_utils.get_triple_idxs_w_blacklist(self.day_paths['id_date'].values, self.bins_to_predict,
            #                                                    self.day_bins, self.len_seq_in,
            #                                                    black_list_path=black_list_path)
            self.idxs = data_utils.get_double_idxs_w_blacklist(self.day_paths['id_date'].values, self.bins_to_predict,
                                                               self.day_bins, self.len_seq_in,
                                                               black_list_path=black_list_path)
        else:
            test_dates = self.day_paths[self.day_paths.split==self.data_split].reset_index()
            # self.idxs = data_utils.get_test_triplets(test_dates['id_date'].sort_values().values,
            #                                          self.test_splits,
            #                                          self.bins_to_predict)
            self.idxs = data_utils.get_test_doubles(test_dates['id_date'].sort_values().values,
                                                    self.test_splits,
                                                    self.bins_to_predict)
            self.day_paths = self.day_paths[self.day_paths.split.isin(['test', 'test-next'])].reset_index()
            
            

    def __len__(self):
        """ total number of samples (sequences of in:4-out:1 in our case) to train """
        return len(self.idxs)
    
    def load_in_seq(self, day_id, in_start_id):  # , lead_time):
        """ load the input sequence """
        
        # 1. load nwcsaf products & metadata
        in_seq, in_info = data_utils.get_sequence_netcdf4(self.len_seq_in, in_start_id, day_id,
                                                          self.products, self.data_path, self.input_vars,
                                                          # self.target_vars,
                                                          crop=self.crop_in, preprocess=self.preprocess['source'],
                                                          day_bins=self.day_bins,
                                                          sorted_dates=self.day_paths.id_date.sort_values().values,
                                                          populate_mask=self.populate_mask)
        
        # 2. Load extra features
        if len(self.static_tensor) != 0:  # 2.1 static features
            in_seq = np.concatenate((in_seq, self.static_tensor), axis=self.channel_dim)
            in_info['channels'] += self.static_desc

        # # 3. Load lead time to predict and normalize it
        # data = np.ones(shape=(self.len_seq_in, 1, self.spatial_dim[0], self.spatial_dim[1]))
        # data[...] = (lead_time+1)/self.bins_to_predict
        # in_seq = np.concatenate((in_seq, data), axis=self.channel_dim)
        # in_info['channels'] += ['lead_time']

        # 3. Load time_slot
        if self.control_params['use_time_slot']:
            data = np.stack([np.ones(shape=(1, *self.spatial_dim)) * ((i + in_start_id) % self.day_bins) for i in
                             range(self.len_seq_in)]) / self.day_bins
            in_seq = np.concatenate((in_seq, data), axis=self.channel_dim)
            in_info['channels'] += ['time_slot']


        
        # 4. Collapse time if needed and set the appropriate data type for learning
        if self.collapse_time:
            in_seq = data_utils.time_2_channels(in_seq, *self.spatial_dim)
        
        in_seq = in_seq.astype(self.precision)  # np.float16)  # np.float32)
        
        return in_seq, in_info        

    def load_in_out(self, day_id, in_start_id):  #, lead_time):
        """ load input/output data """
        
        # load input sequence
        in_seq, in_info = self.load_in_seq(day_id, in_start_id)  # , lead_time)

        # load ground truth
        if self.data_split != 'test':
            target_time = in_start_id + self.len_seq_in  # + lead_time
            # out, masks, channels = data_utils.get_products_netcdf4(day_id, self.day_strings[target_time],
            #                                                self.products, self.data_path, self.target_vars,
            #                                                self.crop_out, self.preprocess['source'])

            out, out_info = data_utils.get_sequence_netcdf4(self.len_seq_out, target_time, day_id,
                                                            self.products, self.data_path, self.target_vars,
                                                            crop=self.crop_out, preprocess=self.preprocess['source'],
                                                            day_bins=self.day_bins,
                                                            sorted_dates=self.day_paths.id_date.sort_values().values,
                                                            populate_mask=self.populate_mask)
            if self.collapse_time:
                out = data_utils.time_2_channels(out, *self.spatial_dim)

            metadata = {'in': in_info, 
                        'out': {'day_in_year': [day_id],  # 'lead_time': [lead_time],
                                'time_bins': [target_time], 'region_id': self.region_id}}
                                # 'masks': out_info['masks']}}
            if self.populate_mask:
                metadata['out']['masks'] = out_info['masks']
        else:
            out = np.asarray([]) # we don't have the ground truth for the test split
            metadata = {'in': in_info,
                        'out': {'day_in_year': [day_id],  # 'lead_time': [lead_time]
                                'region_id': self.region_id}}
        out = out.astype(self.precision)  # np.float16)  # np.float32)
        
        return in_seq, out, metadata

    def __getitem__(self, idx):
        """ load 1 sequence (1 sample) """
        # day_id, in_start_id, lead_time = self.idxs[idx]
        # return self.load_in_out(day_id, in_start_id, lead_time)
        day_id, in_start_id = self.idxs[idx]
        if 'heldout' in self.data_path:
            day_id += 1000
        return self.load_in_out(day_id, in_start_id)
    
    def get_date(self, id_day):
        """ get date from day_in_year id """
        return str(self.day_paths[self.day_paths.id_date==id_day]['date'].values[0])
    
    def geti(self, idx=0):
        """ this function allows you to get 1 sample for debugging
            Note that the batch dimension is missing, so it is added
            
            example: 
                ds = create_dataset(data_split, params)
                in_seq, out, metadata = ds.geti(0)
        """
        in_seq, out, metadata = self.__getitem__(idx)
        in_seq = np.expand_dims(in_seq, axis=0)
        out = np.expand_dims(out, axis=0)
        metadata = np.expand_dims(metadata, axis=0)

        return in_seq, out, metadata
    
def create_dataset(data_split, params, precision=16, populate_mask=False):
    return NWCSAF(data_split, precision=precision, populate_mask=populate_mask, **params)