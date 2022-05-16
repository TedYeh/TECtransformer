from pickletools import optimize
from model import TEC_LSTM, TEC_GRU, TEC_CNNGRU
from torch.utils.data import DataLoader
import torch, math
import torch.nn as nn
from dataloader import TecDataset
import logging, random
from inference import inference
from helper import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def flip_from_probability(p):
    return True if random.random() < p else False

def evaluation(dataloader, model, device):
    model.eval()    
    val_loss = 0
    mse = torch.nn.MSELoss()
    for step, batch in enumerate(dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        b_information = batch[3].to(device)
        #output = model(b_input)
        output = model(torch.cat((b_input, b_information), 2))
        loss = torch.sqrt(mse(output, b_target))
        #loss = torch.sqrt(mse(output[3], b_target[:, 3, :, :]))
        val_loss += loss.detach().item()
    return val_loss / len(dataloader)

def train_transformer(dataloader, valid_dataloader, in_dim, out_dim, batch_size, EPOCH, path_save_model, path_save_loss, device, use_model='LSTM'):
    clean_directory()
    k =int(EPOCH)
    device = torch.device(device)

    model = models[use_model](in_dim, out_dim, batch_size, device).float().to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mse = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0

        #training
        model.train()
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            b_input, b_target = tuple(b.to(device) for b in batch[:2])
            b_information = batch[3].to(device)
            '''
            v = k/(k+math.exp(epoch/k)) # probability of heads/tails depends on the epoch, evolves with time.
            prob_true_val = flip_from_probability(v) # starts with over 95 % probability of true val for each flip in epoch 0.
            
            if prob_true_val: 
                #output = model(b_input)
                output = model(torch.cat((b_input, b_information), 2))
                mean, std = torch.mean(output), torch.std(output)
                prediction = (output-mean) / std  # 儲存model prediction, 並正規化
                last_input = b_input # 紀錄目前input
                last_value = prob_true_val
            else:                    
                if last_value: b_input = torch.cat((b_input[: ,: ,:73*3], prediction.detach()) ,2) # model input = concate(input[1:], last_pred)
                else: b_input = torch.cat((last_input[: ,: ,73:], prediction.detach()) ,2)
                #output = model(b_input)     
                output = model(torch.cat((b_input, b_information), 2))  
                mean, std = torch.mean(output), torch.std(output)
                prediction = (output-mean) / std  # 儲存model prediction, 並正規化
                last_input = b_input # 紀錄目前input
                last_value = prob_true_val
            '''
            output = model(torch.cat((b_input, b_information), 2)) 
            loss1 = torch.sqrt(mse(output[0], b_target[:, 0, :, :]))
            loss2 = torch.sqrt(mse(output[1], b_target[:, 1, :, :]))            
            loss3 = torch.sqrt(mse(output[2], b_target[:, 2, :, :]))            
            loss4 = torch.sqrt(mse(output[3], b_target[:, 3, :, :]))            
            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.detach().item()
        
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_save_model + f'best_train_{epoch}_{use_model}.pth')
            #torch.save(optimizer.state_dict(), path_save_model + f'optimizer_{epoch}_{use_model}.pth')
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}_{use_model}.pth"

        train_loss /= len(dataloader)
        val_loss = evaluation(valid_dataloader, model, device)
        scheduler.step(val_loss)
        log_loss(train_loss, path_save_loss, train=True)
        log_loss(val_loss, path_save_loss, train=False)
        #val_loss = evaluation(val_dataloader, model, device), Validation loss: {val_loss:5.5f}
        logger.info(f"Epoch: {epoch:4d}, Training loss: {train_loss:5.5f}, Validation loss: {val_loss:5.5f}")
    plot_loss(path_save_loss)
    return best_model

def train(dataloader, valid_dataloader, in_dim, out_dim, batch_size, EPOCH, path_save_model, path_save_loss, device, use_model='LSTM'):#, val_dataloader
    same_seeds(0)
    clean_directory()

    device = torch.device(device)
    
    model = models[use_model](in_dim, out_dim, batch_size, device).float().to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mse = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0
        #training
        model.train()
        for step, batch in enumerate(dataloader):            
            #optimizer.zero_grad()
            for param in model.parameters(): param.grad = None
            b_input, b_target = tuple(b.to(device) for b in batch[:2])
            b_information = batch[3].to(device)
            if use_model == 'CNNGRU':
                output = model(b_input.unsqueeze(1))
            else:
                #output = model(b_input)
                output = model(torch.cat((b_input, b_information), 2))
            loss = torch.sqrt(mse(output, b_target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.detach().item()
        
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_save_model + f'best_train_{epoch}_{use_model}.pth')
            #torch.save(optimizer.state_dict(), path_save_model + f'optimizer_{epoch}_{use_model}.pth')
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}_{use_model}.pth"

        train_loss /= len(dataloader)
        val_loss = evaluation(valid_dataloader, model, device)
        scheduler.step(val_loss)
        log_loss(train_loss, path_save_loss, train=True)
        log_loss(val_loss, path_save_loss, train=False)
        #val_loss = evaluation(val_dataloader, model, device), Validation loss: {val_loss:5.5f}
        logger.info(f"Epoch: {epoch:4d}, Training loss: {train_loss:5.5f}, Validation loss: {val_loss:5.5f}")

    plot_loss(path_save_loss)
    return best_model

if __name__ == '__main__':
    plot_loss('save_loss/')


