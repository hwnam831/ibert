import torch
import torch.nn as nn
from Encoder import *
from Decoder import *
from MiniBert import *

def ConvBlockRelu(c_in, c_out, ksize, dilation=1):
    pad = ((ksize-1)//2)*dilation
    return nn.Sequential(
            nn.Conv1d(c_in, c_out, ksize, 1, pad, dilation=dilation),
            nn.BatchNorm1d(c_out),
            nn.ReLU()
            )

def ConvBlock(c_in, c_out, ksize, dilation=1):
    pad = ((ksize-1)//2)*dilation
    return nn.Sequential(
            nn.Conv1d(c_in, c_out, ksize, 1, pad, dilation=dilation),
            nn.BatchNorm1d(c_out)
            )

class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, drop=0.1, dilation=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                ConvBlockRelu(c_in, c_mid, 1),
                ConvBlockRelu(c_mid, c_mid, 5, dilation=dilation),
                ConvBlockRelu(c_mid, c_in, 1),
                nn.Dropout(drop)
                )

    def forward(self, x):
        out = self.block(x)
        out += x
        return out



class CNNAutoEncoder(nn.Module):

    def __init__(self, model_size):
        super(CNNAutoEncoder, self).__init__()
        self.size = model_size
        self.vocab_size = 16
        self.conv1 = ConvBlock(self.vocab_size, self.size, 5)
        self.encoder = nn.Sequential(
            ResBlock(self.size, self.size//2, dilation=2),
            ResBlock(self.size, self.size//4, dilation=4),
            ResBlock(self.size, self.size//2, dilation=2),
            ResBlock(self.size, self.size//2, dilation=4))

        self.classifier = ConvBlock(self.size, self.vocab_size, 1)
        self.indirect = ConvBlock(self.vocab_size, self.size, 5)   

    def forward(self, input):  
        res = self.conv1(input.permute(0,2,1))
        out = self.encoder(res)
        #out = nn.functional.relu(out + self.indirect(input))
        return out

class MiniBertAE(nn.Module):
    def __init__(self, model_size, vocab_size=16, config=MiniBertConfig()):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, model_size)
        self.fc = nn.Linear(model_size, vocab_size)
        self.encoder = MiniBertEncoder(model_size, config.intermediate_size, \
            config.num_hidden_layers, config.num_attention_heads, config)

    def forward(self, input):
        embedded = self.embedding(input)
        hidden_states = self.encoder(embedded)[0]
        logits = self.fc(hidden_states)
        return logits.permute(0,2,1)


class TfS2S(nn.Module):
    def __init__(self, model_size=512, maxlen=128):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.embedding = nn.Embedding(16, model_size)
        self.posembed = nn.Embedding(maxlen, model_size)
        self.tfmodel = nn.Transformer(d_model=model_size, nhead=2, num_encoder_layers=3, num_decoder_layers=3)
        self.fc = nn.Linear(model_size, 16)
    #Batch-first in, batch-first out (N,C,S)
    def forward(self, input, target):
        input2 = input.permute(1,0)
        target2 = target.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape)
        tpos = torch.arange(target2.size(0), device=target.device)[:,None].expand(target2.shape)
        tmask = self.tfmodel.generate_square_subsequent_mask(target2.size(0)).to(input.device)
        src = self.embedding(input2) + self.posembed(ipos)
        tgt = self.embedding(target2)+ self.posembed(tpos)
        out = self.tfmodel(src, tgt, tgt_mask=tmask)
        #out = self.tfmodel(src, tgt)
        return self.fc(out).permute(1,2,0)

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