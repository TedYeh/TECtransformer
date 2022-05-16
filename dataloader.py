from random import shuffle
import torch
import csv
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import numpy as np
import math
import os
from sklearn.preprocessing import StandardScaler
from GIM_TXT_to_csv import read_file, natural_keys, read_omni

class TecDataset(Dataset):
    def __init__(self, path, data_type='dir', mode='train', window_size=4, pred_future=False):
        self.mon_day = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}        
        self.year = path[-5:-1]
        self.path = path
        self.mode = mode
        self.data_type = data_type
        self.pred_future = pred_future
        self.window_size = window_size
        datetimes = []
        if self.data_type == 'dir':
            full_data = []
            file_list = os.listdir(path)
            file_list.sort(key = natural_keys)
            for file_name in file_list:
                one_day_data, datetime = read_file(path+file_name)
                full_data += one_day_data
                datetimes += datetime
                #if mode == 'train':full_data += self.standardize(one_day_data)
                #else:full_data += one_day_data
            self.datetimes = datetimes
            self.tec_data = full_data
        else: 
            self.tec_data, datetime = read_file(path)  
            self.datetimes = datetime     
        
        self._input, self.target, self.tar_date = self.get_data()

    def standardize(self, fulldata):
        tmp = []
        for data in fulldata:
            mean, std = np.mean(data), np.std(data)
            tmp.append((data-mean) / std)
        return tmp
    
    def omni_std(self, data):
        mean, std = np.mean(data), np.std(data)
        tmp = (data-mean) / std
        return tmp

    def get_data(self):
        _input, target, date = [], [], []
        
        for idx in range(len(self.tec_data)):
            if self.pred_future and idx + self.window_size + 4 >= len(self.tec_data):break
            elif idx + self.window_size >= len(self.tec_data):break
            _input.append(self.standardize(self.tec_data[idx:idx+self.window_size]))
            if self.pred_future:target.append(self.tec_data[idx+self.window_size:idx+self.window_size+4])
            else:target.append(self.tec_data[idx+self.window_size])
            date.append(self.datetimes[idx+self.window_size])
            '''
            if self.mode == 'train':
                _input.append(self.tec_data[idx:idx+4])
                target.append(self.tec_data[idx+4])
            else:
                _input.append(self.standardize(self.tec_data[idx:idx+4]))
                target.append(self.tec_data[idx+4])
            '''
        return np.array(_input), target, date

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        latitude_range = np.linspace(87.5, -87.5, 71) #將緯度依2.5度切割為71等分
        lat = torch.tensor([[math.sin(i/90), math.cos(i/90)] for i in latitude_range]) #取sin, cos來表示緯度位置
        input_ori = torch.tensor(self._input[idx], dtype=torch.float)
        _input = torch.cat(tuple(input_ori[:self.window_size]), 1) #將GIM MAP concate在一起
        day = sum([self.mon_day[i] for i in range(1,self.tar_date[idx][1])]) + self.tar_date[idx][2] 
        tar_date = torch.tensor([[math.sin((day)/366), math.cos(day/366), math.sin(self.tar_date[idx][3]/24), \
        math.cos(self.tar_date[idx][3]/24)]for i in range(71)], dtype=torch.float)
        information = torch.cat((lat, tar_date), 1)
        #_input = torch.cat((_input, lat, tar_date), 1)
        #_input = torch.tensor(self._input[idx], dtype=torch.float)
        target = torch.tensor(self.target[idx], dtype=torch.float)        
        return _input, target, self.tar_date[idx], information

if __name__ == '__main__':
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    '''
    tmpdata = TecDataset('txt/valid_2020/', data_type='dir', pred_future=False)
    #tmpdataloader = DataLoader(tmpdata, batch_size = 16, shuffle = False)    
    #tmpdata = TecDataset('txt/2020/CODG0500.20I', data_type='file', mode='test', pred_future=True)
    tmpdataloader = DataLoader(tmpdata, batch_size = 64, shuffle = True) 
    print(len(tmpdataloader))
    for inp_map, tar_map, date, information in tmpdataloader:
        #print(inp_map, tar_map[:,0,:,:].size())
        print(inp_map.size(), tar_map.size(), information.size())
        print(date)
        input()