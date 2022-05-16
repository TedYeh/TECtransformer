import argparse
from train_model import *
from dataloader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from inference import *

def main(
    epoch = 100,
    batch_size = 32,
    train_path = 'txt/2019/',
    path_save_model = 'save_model/',
    path_save_loss = 'save_loss/',
    device='cpu',
    use_model='LSTM',
    pred_future = False
):
    in_dim, out_dim = 73*4+8, 73
    train_dataset = TecDataset(train_path, pred_future=pred_future)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle = True)
    valid_dataset = TecDataset('txt/valid_2020/', data_type='dir', mode='test', pred_future=pred_future)
    valid_dataloader = DataLoader(valid_dataset, batch_size = 16 , shuffle = False)
    #test_dataset = TecDataset('txt/2020/CODG0500.20I', data_type='file', mode='test')
    test_dataset = TecDataset('txt/test_2020/', data_type='dir', mode='test', pred_future=pred_future)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    clean_directory()
    #if "Transformer" in use_model:
    #    best_model = train_transformer(train_dataloader, valid_dataloader, 73*4+8, 73, batch_size, epoch, path_save_model, path_save_loss, device, use_model)
    #else:
    #    best_model = train(train_dataloader, valid_dataloader, 73*4, 73, batch_size, epoch, path_save_model, path_save_loss, device, use_model)  
    
    best_model = train(train_dataloader, valid_dataloader, in_dim, out_dim, batch_size, epoch, path_save_model, path_save_loss, device, use_model)  
    inference(path_save_model, in_dim, out_dim, best_model, test_dataloader, device, use_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_path', type=str, default='txt/2019/')
    parser.add_argument('--path_save_model', type=str, default='save_model/')
    parser.add_argument('--path_save_loss', type=str, default='save_loss/')
    parser.add_argument('--use_model', type=str, default='LSTM')
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    main(epoch=args.epoch, batch_size=args.batch_size, train_path=args.train_path, path_save_model=args.path_save_model, path_save_loss=args.path_save_loss, device=device, use_model=args.use_model)
