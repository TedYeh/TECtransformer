import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from model import TEC_LSTM, TEC_GRU, TEC_CNNGRU, Transformer, Multi_Transformer

models = {'LSTM':TEC_LSTM, 'GRU':TEC_GRU, 'CNNGRU':TEC_CNNGRU, 'Transformer':Transformer, "Multi_Transformer":Multi_Transformer}
# save train or validation loss
def log_loss(loss_val : float, path_to_save_loss : str, train : bool = True):
    if train:
        file_name = "train_loss.txt"
    else:
        file_name = "val_loss.txt"

    path_to_file = path_to_save_loss+file_name
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    with open(path_to_file, "a") as f:
        f.write(str(loss_val)+"\n")
        f.close()

def EMA(values, alpha=0.1):
    ema_values = [values[0]]
    for idx, item in enumerate(values[1:]):
        ema_values.append(alpha*item + (1-alpha)*ema_values[idx])
    return ema_values

def plot_tec(pred, target, show_type, dates):
    title = f'TEC - Predict v.s. Target {show_type}'
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({"figure.figsize":[14, 10]})
    plt.plot(pred, color = 'indigo', label='Prediction')
    plt.plot(target, '--', label="Target")
    plt.minorticks_on()
    plt.xticks([i for i in range(0, len(target), 8)], [dates[i] for i in range(0, len(dates), 8)])
    plt.xlabel('Time (Hour)')
    plt.ylabel('TEC')
    plt.legend()
    plt.title(title)
    plt.savefig(f'img/{title}.png')
    plt.close()

def plot_loss(path_save_loss, train=True):
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10,6.8))
    with open(path_save_loss+'train_loss.txt', 'r') as f:
        train_loss_list = [float(line) for line in f.readlines()]
    with open(path_save_loss+'val_loss.txt', 'r') as f:
        val_loss_list = [float(line) for line in f.readlines()]
    #if train:title = "Train"
    #else:title = "Validation"
    title = "Training v.s. Validation Loss"
    train_EMA_loss = EMA(train_loss_list)
    val_EMA_loss = EMA(val_loss_list)
    plt.plot(train_loss_list, color = 'indigo', label='train_Loss')
    plt.plot(train_EMA_loss, '--', label="EMA train_loss")
    plt.plot(val_loss_list, color = 'red', label='valid_Loss')
    plt.plot(val_EMA_loss, '--', label="EMA valid_loss")
    plt.minorticks_on()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title+' Loss')
    plt.savefig(path_save_loss+f'{title}.png')
    plt.close()

def clean_directory():
    if os.path.exists('save_loss'):
        shutil.rmtree('save_loss')
    if os.path.exists('save_model'): 
        shutil.rmtree('save_model')
    if os.path.exists('save_predictions'): 
        shutil.rmtree('save_predictions')
    if os.path.exists('img/2019_real'):
        shutil.rmtree('img/2019_real')
    if os.path.exists('img/2020_pred'): 
        shutil.rmtree('img/2020_pred')
    if os.path.exists('img/2020_GIM-pred'): 
        shutil.rmtree('img/2020_GIM-pred')
    os.mkdir("save_loss")
    os.mkdir("save_model")
    os.mkdir("save_predictions")
    os.mkdir('img/2019_real')
    os.mkdir('img/2020_pred')

if __name__ == "__main__":
    plot_loss('save_loss/')