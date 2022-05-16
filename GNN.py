from scipy.sparse import coo_matrix
import numpy as np
import os
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from GIM_TXT_to_csv import read_file, natural_keys
from sklearn.feature_extraction import image
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters():
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.convs[1](x, edge_index)

        return x

def graph_laoder(path, batch_sz=64, shuffle=False):
    full_data = []
    file_list = os.listdir(path)
    file_list.sort(key = natural_keys)
    for file_name in file_list:
        one_day_data, _ = read_file(path+file_name)
        for i in range(len(one_day_data)-1):
            hour_data = one_day_data[i]
            next_hour = one_day_data[i+1]
            now_graph = image.img_to_graph(hour_data)
            next_graph = image.img_to_graph(next_hour)
            x = torch.tensor(now_graph.data, dtype=torch.float)
            y = torch.tensor(next_graph.data, dtype=torch.float)
            edge_index = torch.tensor(np.array([now_graph.row, now_graph.col]), dtype=torch.long)
            full_data.append(Data(x=x, y=y, edge_index=edge_index))
    return DataLoader(full_data, batch_size=batch_sz, shuffle=shuffle)

def train():
    model.train()
    mse = torch.nn.MSELoss()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        b_target = data.y.to(device)
        loss = torch.sqrt(mse(output, b_target))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)

if __name__ == '__main__':
    path = 'txt/valid_2020/'
    loader = graph_laoder(path, 64, True)
    data = next(iter(loader))[0]
    print(help(data))
    input()
    device = torch.device('cuda')
    model = SAGE(in_channels=data.num_features,  hidden_channels=128, out_channels=data.num_classes).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    train_loader = graph_laoder(path, batch_sz=64, shuffle=True)
    for batch in train_loader:
        print(batch.num_features)
        input()
    #for epoch in range(num_epochs):
    #    loss = train()
    #    print(loss)
    '''
    print(help(data))
    print(data.num_nodes, data.is_directed())
    print(data.edge_index)
    print(data.x)
    print(data.y)
    '''
    