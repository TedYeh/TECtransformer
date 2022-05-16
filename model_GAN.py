import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, batch_size, device, num_layers=5, dropout=0.3):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=6, dropout=dropout, norm_first=True, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(in_dim, out_dim)
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