import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import TFEncoder, CNNEncoder, XLNetEncoderLayer
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
    def __init__(self, model_size=512, maxlen=128, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        self.embedding = nn.Linear(vocab_size, model_size)
        self.posembed = nn.Embedding(maxlen, model_size)
        self.enclayer = nn.TransformerEncoderLayer(d_model=model_size, nhead=2)
        self.norm = nn.LayerNorm(model_size)
        self.tfmodel = nn.TransformerEncoder(self.enclayer, num_layers=6, norm=self.norm)
        self.fc = nn.Linear(model_size, vocab_size)
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0,2)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        src = self.embedding(input2) + self.posembed(ipos)
        out = self.tfmodel(src)
        return self.fc(out).permute(1,2,0)

class CNNAE(nn.Module):
    def __init__(self, model_size=512,vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        self.embedding = nn.Conv1d(vocab_size, model_size, 3, padding=1)
        self.norm = nn.BatchNorm1d(model_size)
        self.encoder = CNNEncoder(model_size)
        self.fc = nn.Linear(model_size, vocab_size)
    
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        embed = self.norm(self.embedding(input.permute(0,2,1)))
        out = self.encoder(embed.permute(2,0,1))
        return self.fc(out).permute(1,2,0)

class XLNetAE(nn.Module):
    def __init__(self, d_model=512, maxlen=128, num_layers=6, vocab_size=16):
        super().__init__()
        self.d_model=d_model
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        self.embedding = nn.Linear(vocab_size, d_model)
        self.posembed = nn.Embedding(maxlen, d_model)
        self.relembed = nn.Embedding(2*maxlen, d_model)
        #self.enclayer = XLNetEncoderLayer(d_model=d_model, nhead=2)
        #self.encoder = nn.TransformerEncoder(self.enclayer, num_layers=num_layers)
        self.encoder = nn.ModuleList([
            XLNetEncoderLayer(d_model=d_model, nhead=2) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # create relative positional encoding.
        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        beg, end = klen, -qlen

        fwd_pos_seq = torch.arange(beg, end, -1.0)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb.to(next(self.parameters()))
        return pos_emb

    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0,2)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        #pos_emb = self.relative_positional_encoding(input2.size(0),input2.size(0),input.size(0))
        klen = input2.shape[0]
        rpos = torch.arange(self.maxlen-klen, self.maxlen+klen, device=input.device)
        r = self.relembed(rpos[:,None].expand(2*klen,input2.shape[1]))
        h,g = (self.embedding(input2), self.posembed(ipos))
        for layer in self.encoder:
            h,g = layer(h,g,r)
        #out = torch.cat(out,dim=-1)
        return self.fc(h).permute(1,2,0)