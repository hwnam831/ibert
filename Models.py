import torch
import torch.nn as nn
from Encoder import TFEncoder, CNNEncoder
from Decoder import TFDecoder


class TfS2S(nn.Module):
    def __init__(self, model_size=512, maxlen=128):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.embedding = nn.Embedding(16, model_size)
        self.posembed = nn.Embedding(maxlen, model_size)
        self.encoder = TFEncoder(model_size, nhead=2, num_layers=3)
        self.decoder = TFDecoder(model_size, 16)
    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, target):
        input2 = input.permute(1,0)
        target2 = target.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape)
        tpos = torch.arange(target2.size(0), device=target.device)[:,None].expand(target2.shape)

        src = self.embedding(input2) + self.posembed(ipos)
        tgt = self.embedding(target2)+ self.posembed(tpos)
        memory = self.encoder(src)
        return self.decoder(tgt, memory).permute(1,2,0)

class TfAE(nn.Module):
    def __init__(self, model_size=512, maxlen=128):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.embedding = nn.Linear(16, model_size)
        self.posembed = nn.Embedding(maxlen, model_size)
        self.enclayer = nn.TransformerEncoderLayer(d_model=model_size, nhead=2)
        self.norm = nn.LayerNorm(model_size)
        self.tfmodel = nn.TransformerEncoder(self.enclayer, num_layers=6, norm=self.norm)
        self.fc = nn.Linear(model_size, 16)
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0,2)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        src = self.embedding(input2) + self.posembed(ipos)
        out = self.tfmodel(src)
        return self.fc(out).permute(1,2,0)

class CNNAE(nn.Module):
    def __init__(self, model_size=512):
        super().__init__()
        self.model_size=model_size
        self.embedding = nn.Conv1d(16, model_size, 3, padding=1)
        self.norm = nn.BatchNorm1d(model_size)
        self.encoder = CNNEncoder(model_size)
        self.fc = nn.Linear(model_size, 16)
    
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        embed = self.norm(self.embedding(input.permute(0,2,1)))
        out = self.encoder(embed.permute(2,0,1))
        return self.fc(out).permute(1,2,0)