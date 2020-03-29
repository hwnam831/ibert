import torch
import torch.nn as nn

class TFEncoder(nn.Module):
    def __init__(self, model_size=512, nhead=2, num_layers=3):
        super().__init__()
        self.model_size=model_size
        self.enclayer = nn.TransformerEncoderLayer(d_model=model_size, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.enclayer, \
            num_layers=num_layers)
    #Seq-first in-out (S,N,C)
    def forward(self, src):
        memory = self.encoder(src)
        return memory

def ConvBlockRelu(c_in, c_out, ksize, dilation=1):
    pad = ((ksize-1)//2)*dilation
    return nn.Sequential(
            nn.Conv1d(c_in, c_out, ksize, 1, pad, dilation=dilation),
            nn.BatchNorm1d(c_out),
            nn.ReLU())

class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, drop=0.1, dilation=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                ConvBlockRelu(c_in, c_mid, 1),
                ConvBlockRelu(c_mid, c_mid, 5, dilation=dilation),
                ConvBlockRelu(c_mid, c_in, 1),
                nn.Dropout(drop))

    def forward(self, x):
        out = self.block(x)
        out += x
        return out

class CNNEncoder(nn.Module):
    def __init__(self, model_size):
        super().__init__()
        self.size = model_size
        self.vocab_size = 16
        self.encoder = nn.Sequential(
            ResBlock(self.size, self.size//2, dilation=1),
            ResBlock(self.size, self.size//2, dilation=2),
            ResBlock(self.size, self.size//2, dilation=4),
            ResBlock(self.size, self.size//2, dilation=1),
            ResBlock(self.size, self.size//2, dilation=2),
            ResBlock(self.size, self.size//2, dilation=4))
    #Seq-first in (S,N,C), Seq-first out
    def forward(self, input):  
        memory = self.encoder(input.permute(1,2,0))
        return memory.permute(2,0,1)