from model import TEC_LSTM, TEC_GRU
from torch.utils.data import DataLoader
import torch, random
import torch.nn as nn
from dataloader import TecDataset
import logging
import pandas as pd
from helper import *
import matplotlib.pyplot as plt
from GIM_TXT_to_csv import plot, generate_gif, plot_init
import numpy as np
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def mape(y_pred, y_target):
    mask=y_target!=0
    return np.fabs((y_target[mask]-y_pred[mask])/y_target[mask])

def read_one_day_omni(all_omni, DOY, hour):   
    omni_data = all_omni.iloc[(DOY-1)*24+hour].to_list() 
    #print(omni_data)
    #Kp index,R (Sunspot No.),"Dst-index, nT","ap_index, nT",f10.7_index
    #['Dst index', 'f10.7 index', 'Kp index', 'ap index', 'Sunspot No.']
    data = list(omni_data[3:])
    return [data[2], data[4], data[0], data[3], data[1], np.nan]

def get_error(output, target):
    mse_loss = torch.nn.MSELoss()
    loss = torch.nn.MSELoss(reduction='none')
    taiwan_point = 25*72 + 61
    rmse_map = torch.flatten(torch.sqrt(loss(output[0], target[0]))).cpu().detach().numpy()
    error_map = torch.flatten(torch.sub(output[0], target[0])).cpu().detach().numpy()
    rmse = rmse_map[taiwan_point]
    mape_loss = mape(output.cpu().detach().numpy(), target.cpu().detach().numpy())
    global_rmse, global_mape = torch.sqrt(mse_loss(output, target)), mape_loss.mean()
    local_rmse, local_mape = float(rmse), mape_loss[taiwan_point]
    max_error, min_error = max(error_map), min(error_map)
    return [float("%2.3f"%global_rmse.cpu().detach().numpy()), float("%2.3f"%local_rmse), float("%2.3f"%global_mape), float("%2.3f"%local_mape), float("%2.3f"%max_error), float("%2.3f"%min_error)]

def inference_four_hour(path_save_model, in_dim, out_dim, best_model, dataloader, device, use_model):
    tec_tar, tec_pred, dates, errors, omnis = [], [], [], [], []
    taiwan_point, idx = 25*72 + 61, 0
    DOY_label = []
    omni_path = 'omni/2020hourly.csv'
    all_omni = pd.read_csv(omni_path)
    plot_init()
    device = torch.device(device)
    model = models[use_model](in_dim, out_dim, 1, device).float().to(device)
    model.load_state_dict(torch.load(path_save_model+best_model))
    criterion = torch.nn.MSELoss()
    #x, y = random.randint(0, 73), random.randint(0, 71)
    data_list = list(dataloader)
    model.eval()
    while True:
        if idx >= len(data_list):break
        batch = data_list[idx]
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        b_information = batch[3].to(device)
        b_time = tuple(b.numpy()[0] for b in batch[2])  
        idx_4_h = int(idx)
        for _ in range(4):
            if idx_4_h >= len(data_list):break
            batch = data_list[idx_4_h]
            t_target, information = tuple(b.to(device) for b in [batch[1], batch[3]])
            output = model(torch.cat((b_input, information), 2))
            output = output.type(torch.LongTensor).type(torch.FloatTensor).to(device)
            t_time = tuple(b.numpy()[0] for b in batch[2]) 
            m, d, h = t_time[1:]
            dates.append(f'{m}/{d}\n{h}:00') 
            mean, std = torch.mean(output), torch.std(output)
            prediction = (output-mean) / std  # 儲存model prediction, 並正規化  
            b_input = torch.cat((b_input[: ,: ,72:], prediction.detach()) ,2)

            pred = torch.tensor(output, dtype=int).clone().cpu().detach().numpy()
            #print(torch.argmax(torch.abs(torch.sub(b_target[0], output[0])), 1).cpu().detach().numpy())
            max_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(t_target[0], output[0]))), 60)[1].cpu().detach().numpy()
            min_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(t_target[0], output[0]))), 60, largest=False)[1].cpu().detach().numpy()
            max_points, min_points = [[idx%72, idx//72] for idx in max_idxes], [[idx%72, idx//72] for idx in min_idxes]
            points = [max_points, min_points]            
            
            tec_pred.append(torch.flatten(output.cpu()).detach().numpy()[taiwan_point])
            tec_tar.append(torch.flatten(t_target.cpu()).numpy()[taiwan_point])
            
            idx_4_h += 1
        error_ = get_error(output, t_target)
        
        target, model_pred, error = t_target.cpu().numpy()[0], pred[0], torch.sub(output[0], t_target[0]).cpu().detach().numpy()
        m, d, h = t_time[1:]
        DOY, hour = int(datetime.datetime.strptime(f'{m}-{d}', '%m-%d').strftime('%j'))+1, int(h)
        
        DOY_label.append(DOY)
        omni = read_one_day_omni(all_omni, DOY, hour)
        errors.append(error_)
        omnis.append(omni)
        plot_tec_pred(model_pred, error, [error_], [omni], DOY, datetimes=t_time, use_model=use_model)   
        idx+=1
    show_big_error(error_, DOY_label, 50)
    DOY_label = list(set(DOY_label))
    print(DOY_label)
    print(np.array(errors))
    print(np.array(omnis))
    plot_trend(np.array(errors), np.array(omnis), DOY_label)
    plt.close('all')   
    #plot_tec(tec_pred, tec_tar, 'Taiwan', dates)            

def make_gif(path_list, timestamp, use_model):
    generate_gif(path_list[0], timestamp, use_model='GIM-Pred')
    generate_gif(path_list[1], timestamp, use_model)
    generate_gif(path_list[2], timestamp)

def inference(path_save_model, in_dim, out_dim, best_model, dataloader, device, use_model):
    # target map, predict map, rmse, target date, another date format
    tec_tar, tec_pred, tec_rmse, dates, d_f = [], [], [], [], []
    mon_day = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    omni_path = 'omni/2020hourly.csv'
    all_omni = pd.read_csv(omni_path)
    tec_dict = {}
    
    # each hour tec
    taiwan_point = 25*72 + 61
    past_month = 0 #record month
    plot_init()
    device = torch.device(device)
    model = models[use_model](in_dim, out_dim, 1, device).float().to(device)
    model.load_state_dict(torch.load(path_save_model+best_model))
    
    #x, y = random.randint(0, 73), random.randint(0, 71)
    model.eval()
    for step, batch in enumerate(dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        
        b_information = batch[3].to(device)
        b_time = tuple(b.numpy()[0] for b in batch[2])
        y, m, d, h = b_time[:]
        
        if use_model == 'CNNGRU':
            output = model(b_input.unsqueeze(1))
        else:output = model(torch.cat((b_input, b_information), 2))
        pred = torch.tensor(output, dtype=int).cpu().detach().numpy()
        #print(torch.argmax(torch.abs(torch.sub(b_target[0], output[0])), 1).cpu().detach().numpy())
        #max_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(b_target[0], output[0]))), 60)[1].cpu().detach().numpy()
        #min_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(b_target[0], output[0]))), 60, largest=False)[1].cpu().detach().numpy()
        #max_points, min_points = [[idx%72, idx//72] for idx in max_idxes], [[idx%72, idx//72] for idx in min_idxes]
        #points = [max_points, min_points]        
        
        #tec_rmse.append(float(f'{rmse:02.2f}'))
        dates.append(f'{m:02d}/{d:02d} {h:02d}:00')
        d_f.append(f'{m:02d}.{d:02d}\n{h:02d}:00')
        tec_pred.append(torch.flatten(output.cpu()).detach().numpy()[taiwan_point])
        tec_tar.append(torch.flatten(b_target.cpu()).numpy()[taiwan_point])
        #print(torch.abs(torch.sub(output, b_target[0])).mean().cpu().detach().numpy())
        #plot(torch.sub(b_target[0], output[0]).cpu().detach().numpy(), datetime=b_time, type_='GIM-Pred', use_model=use_model)
        #plot(pred[0], datetime=b_time, type_='Prediction', use_model=use_model)
        #plot(b_target.cpu().numpy()[0], datetime=b_time)
        error_ = get_error(output, b_target)
        target, model_pred, error = b_target.cpu().numpy()[0], pred[0], torch.sub(output[0], b_target[0]).cpu().detach().numpy()
        DOY, hour = int(datetime.datetime.strptime(f'{m}-{d}', '%m-%d').strftime('%j'))+1, int(h)
        omni = read_one_day_omni(all_omni, DOY, hour)
        plot_tec_pred(target, error, [error_], [omni], datetimes=b_time, use_model=use_model)  
           
    plt.close('all')           

def show_big_error(rmse, dates, value):
    for e, d in zip(rmse, dates):
        if e >= value:print(d, e)


if __name__ == '__main__':
    #clean_directory()
    in_dim, out_dim = 72*4+6, 72
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tmpdata = TecDataset('txt/2020/CODG0500.20I', data_type='file', mode='test')
    #tmpdata = TecDataset('txt/monthly_testdata/', data_type='dir', mode='test')
    
    tmpdata = TecDataset('txt/test_2020/', data_type='dir', mode='test')
    tmpdataloader = DataLoader(tmpdata, batch_size = 1, shuffle = False)
    #inference('save_model/', in_dim, out_dim, 'best_train_50_Transformer.pth', tmpdataloader, device, 'Transformer')
    inference_four_hour('save_model/', in_dim, out_dim, 'best_train_44_Transformer.pth', tmpdataloader, device, 'Transformer')
    