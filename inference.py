from model import TEC_LSTM, TEC_GRU
from torch.utils.data import DataLoader
import torch, random
import torch.nn as nn
from dataloader import TecDataset
import logging
from helper import *
import matplotlib.pyplot as plt
from GIM_TXT_to_csv import plot, generate_gif, plot_init
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def inference_four_hour(path_save_model, in_dim, out_dim, best_model, dataloader, device, use_model):
    tec_tar, tec_pred, dates = [], [], []
    taiwan_point, idx = 25*73 + 61, 0
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
        
        for _ in range(4):
            if idx >= len(data_list):break
            batch = data_list[idx]
            target, information = tuple(b.to(device) for b in [batch[1], batch[3]])
            output = model(torch.cat((b_input, information), 2))
            t_time = tuple(b.numpy()[0] for b in batch[2]) 
            m, d, h = t_time[1:]
            dates.append(f'{m}/{d}\n{h}:00') 
            mean, std = torch.mean(output), torch.std(output)
            prediction = (output-mean) / std  # 儲存model prediction, 並正規化  
            b_input = torch.cat((b_input[: ,: ,73:], prediction.detach()) ,2)

            pred = torch.tensor(output, dtype=int).cpu().detach().numpy()
            #print(torch.argmax(torch.abs(torch.sub(b_target[0], output[0])), 1).cpu().detach().numpy())
            max_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(target[0], output[0]))), 60)[1].cpu().detach().numpy()
            min_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(target[0], output[0]))), 60, largest=False)[1].cpu().detach().numpy()
            max_points, min_points = [[idx%73, idx//73] for idx in max_idxes], [[idx%73, idx//73] for idx in min_idxes]
            points = [max_points, min_points]
            #print([max_idx//73, max_idx%73], [min_idx//73, min_idx%73])
            print(t_time, torch.tensor(batch[2]))
            print(output.detach().mean(), target.mean())
            #print(torch.sub(output[0], b_target[0]).cpu().detach().numpy())
            
            tec_pred.append(torch.flatten(output.cpu()).detach().numpy()[taiwan_point])
            tec_tar.append(torch.flatten(target.cpu()).numpy()[taiwan_point])
            #print(torch.abs(torch.sub(output, b_target[0])).mean().cpu().detach().numpy())
            plot(torch.sub(target[0], output[0]).cpu().detach().numpy(), datetime=t_time, type_='GIM-Pred', use_model=use_model)
            plot(pred[0], points=points, datetime=t_time, type_='Prediction', use_model=use_model)
            plot(target.cpu().numpy()[0], datetime=t_time) 
            idx += 1
        idx+=1

    plt.close()   
    generate_gif('img/2020_GIM-Pred/', b_time, use_model='GIM-Pred')
    generate_gif('img/2020_pred/', b_time, use_model)
    generate_gif('img/2020_real/', b_time)
    plot_tec(tec_pred, tec_tar, 'Taiwan', dates)            
    
def inference(path_save_model, in_dim, out_dim, best_model, dataloader, device, use_model):
    tec_tar, tec_pred, dates = [], [], []
    taiwan_point = 25*73 + 61
    plot_init()
    device = torch.device(device)
    model = models[use_model](in_dim, out_dim, 1, device).float().to(device)
    model.load_state_dict(torch.load(path_save_model+best_model))
    criterion = torch.nn.MSELoss()
    #x, y = random.randint(0, 73), random.randint(0, 71)
    model.eval()
    for step, batch in enumerate(dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        b_information = batch[3].to(device)
        b_time = tuple(b.numpy()[0] for b in batch[2])
        m, d, h = b_time[1:]
        dates.append(f'{m}/{d}\n{h}:00')
        if use_model == 'CNNGRU':
            output = model(b_input.unsqueeze(1))
        else:output = model(torch.cat((b_input, b_information), 2))
        pred = torch.tensor(output, dtype=int).cpu().detach().numpy()
        #print(torch.argmax(torch.abs(torch.sub(b_target[0], output[0])), 1).cpu().detach().numpy())
        max_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(b_target[0], output[0]))), 60)[1].cpu().detach().numpy()
        min_idxes = torch.topk(torch.flatten(torch.abs(torch.sub(b_target[0], output[0]))), 60, largest=False)[1].cpu().detach().numpy()
        max_points, min_points = [[idx%73, idx//73] for idx in max_idxes], [[idx%73, idx//73] for idx in min_idxes]
        points = [max_points, min_points]
        #print([max_idx//73, max_idx%73], [min_idx//73, min_idx%73])
        print(b_time, torch.tensor(batch[2]))
        print(output.detach().mean(), b_target.mean())
        #print(torch.sub(output[0], b_target[0]).cpu().detach().numpy())
        
        tec_pred.append(torch.flatten(output.cpu()).detach().numpy()[taiwan_point])
        tec_tar.append(torch.flatten(b_target.cpu()).numpy()[taiwan_point])
        #print(torch.abs(torch.sub(output, b_target[0])).mean().cpu().detach().numpy())
        plot(torch.sub(b_target[0], output[0]).cpu().detach().numpy(), datetime=b_time, type_='GIM-Pred', use_model=use_model)
        plot(pred[0], points=points, datetime=b_time, type_='Prediction', use_model=use_model)
        plot(b_target.cpu().numpy()[0], datetime=b_time) 
    
    plt.close()   
    generate_gif('img/2020_GIM-Pred/', b_time, use_model='GIM-Pred')
    generate_gif('img/2020_pred/', b_time, use_model)
    generate_gif('img/2020_real/', b_time)
    plot_tec(tec_pred, tec_tar, 'Taiwan', dates)


if __name__ == '__main__':
    #clean_directory()
    in_dim, out_dim = 73*4+8, 73
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #tmpdata = TecDataset('txt/2020/CODG0500.20I', data_type='file', mode='test')
    tmpdata = TecDataset('txt/test_2021/', data_type='dir', mode='test')
    tmpdataloader = DataLoader(tmpdata, batch_size = 1, shuffle = False)
    inference('save_model/', in_dim, out_dim, 'best_train_100_Transformer.pth', tmpdataloader, device, 'Transformer')
    #inference_four_hour('save_model/', in_dim, out_dim, 'best_train_100_Transformer.pth', tmpdataloader, device, 'Transformer')
