import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class Multi_Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, batch_size, device, num_layers=5, dropout=0.3):
        super(Multi_Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=12, dropout=dropout, norm_first=True, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder1 = nn.Linear(in_dim, out_dim, device=device)
        self.decoder2 = nn.Linear(in_dim, out_dim, device=device)
        self.decoder3 = nn.Linear(in_dim, out_dim, device=device)
        self.decoder4 = nn.Linear(in_dim, out_dim, device=device)
        self.device = device
        self.batch_size = batch_size
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data.uniform_(-initrange, initrange)
        self.decoder3.bias.data.zero_()
        self.decoder3.weight.data.uniform_(-initrange, initrange)
        self.decoder4.bias.data.zero_()
        self.decoder4.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)==1).transpose(0, 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tec):
        mask = self._generate_square_subsequent_mask(tec.size(1)).to(self.device)
        output = self.transformer_encoder(tec, mask)
        output1 = F.relu(self.decoder1(output))
        output2 = F.relu(self.decoder2(output))
        output3 = F.relu(self.decoder3(output))
        output4 = F.relu(self.decoder4(output))
        return output1, output2, output3, output4

class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, batch_size, device, num_layers=5, dropout=0.3):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=12, dropout=dropout, norm_first=True, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(in_dim, out_dim, device=device)
        self.device = device
        self.batch_size = batch_size
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)==1).transpose(0, 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tec):
        mask = self._generate_square_subsequent_mask(tec.size(1)).to(self.device)
        output = self.transformer_encoder(tec, mask)
        output = F.relu(self.decoder(output))
        return output

class TEC_LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device):
        super(TEC_LSTM, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.hidden_dim = 128
        self.fc_input = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc_output = nn.Linear(self.hidden_dim, self.out_dim)
        self.lstm1 =  nn.LSTM(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.lstm2 =  nn.LSTM(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, tec):
        h0 = torch.zeros(2, self.batch_size, self.hidden_dim//2).float().to(self.device)
        c0 = torch.zeros(2, self.batch_size, self.hidden_dim//2).float().to(self.device)
        output = self.fc_input(tec)
        output = self.relu(output)
        output = self.dropout(output)
        output, (hn, cn) = self.lstm1(output, (h0, c0))
        output = self.relu(output)
        output = self.dropout(output)
        output, (hn, cn) = self.lstm2(output, (hn, cn))
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc_output(output)
        return output

class TEC_GRU(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device):
        super(TEC_GRU, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = 128
        self.fc_input = nn.Linear(self.in_dim, self.hidden_dim)
        self.fc_output = nn.Linear(self.hidden_dim, self.out_dim)
        self.gru1 =  nn.GRU(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.gru2 =  nn.GRU(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, tec, times=None):
        output = self.relu(self.fc_input(tec))
        output = self.dropout(output)
        output, _ = self.gru1(output)
        output = self.relu(self.layer_norm(output))
        output = self.dropout(output)
        output, _ = self.gru2(output)
        output = self.relu(self.layer_norm(output))
        output = self.relu(self.fc_output(output))
        return output

class TEC_CNNGRU(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device):
        super(TEC_CNNGRU, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = 64
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,71), padding=0)  
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1,71), padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=(1,71), padding=0)
        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim//2)
        self.fc_output = nn.Linear(self.hidden_dim, self.out_dim)
        self.gru1 =  nn.GRU(82, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.gru2 =  nn.GRU(self.hidden_dim, self.hidden_dim//2, bidirectional=True, batch_first=True)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, tec):
        output = F.relu(self.bn1(self.conv1(tec)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.conv3(output))
        output = output.squeeze(1)
        output, _ = self.gru1(output)
        output = F.relu(self.ln(output))
        output, _ = self.gru2(output)
        output = F.relu(self.ln(output))
        output = F.relu(self.fc_output(output))
        return output

if __name__ == "__main__":
    from dataloader import TecDataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp_model = Transformer(73*4, 73, 16, device)
    tmp_model = tmp_model.float().to(device)
    tmpdata = TecDataset('txt/2020/CODG0500.20I', data_type='file', mode='test')
    tmpdataloader = DataLoader(tmpdata, batch_size = 16, shuffle = False)
    in_map, tar_map, time = next(iter(tmpdataloader))
    print(in_map.size())
    in_map = in_map.to(device=device, dtype=torch.float)
    output = tmp_model(in_map)
    print(output.size())
    input()
    criterion = torch.nn.MSELoss()
    #print(torch.sqrt(criterion(tmp_ae(in_map), tar_map.float().to(device))))

