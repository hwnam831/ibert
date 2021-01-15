import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import TFEncoder, CNNEncoder, XLNetEncoderLayer
from Decoder import TFDecoder
from dnc import DNC
from models import UTransformer

class TfS2S(nn.Module):
    def __init__(self, model_size=512, maxlen=256):
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
    def __init__(self, model_size=512, nhead=4, num_layers=6, maxlen=256, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.posembed = nn.Embedding(maxlen, model_size)
        self.enclayer = nn.TransformerEncoderLayer(d_model=model_size, nhead=nhead)
        self.norm = nn.LayerNorm(model_size)
        self.tfmodel = nn.TransformerEncoder(self.enclayer, num_layers=num_layers, norm=self.norm)
        self.fc = nn.Linear(model_size, vocab_size)

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.expand(-1, bsz, -1)

        return pos_emb

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input): 
        input2 = input.permute(1,0)
        #ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        ipos = torch.arange(input.size(1), device=input.device)

        freq_seq = torch.arange(0, self.model_size, 2.0, dtype=torch.float, device=input.device)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.model_size))

        pos_emb = self.positional_embedding(ipos, inv_freq, input.size(0))

        #src = self.embedding(input2) + self.posembed(ipos)
        src = self.embedding(input2) + pos_emb
        out = self.tfmodel(src) #S,N,C
        return self.fc(out).permute(1,2,0)

class CNNAE(nn.Module):
    def __init__(self, model_size=512,vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.norm = nn.LayerNorm(model_size)
        self.encoder = CNNEncoder(model_size)
        self.fc = nn.Linear(model_size, vocab_size)
    
    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        embed = self.norm(self.embedding(input.permute(1,0)))
        out = self.encoder(embed)
        return self.fc(out).permute(1,2,0)

class XLNetAE(nn.Module):
    def __init__(self, d_model=512, nhead=4, maxlen=256, num_layers=6, vocab_size=16):
        super().__init__()
        self.d_model=d_model
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.posembed = nn.Embedding(maxlen, d_model)
        self.relembed = nn.Embedding(2*maxlen, d_model)
        #self.enclayer = XLNetEncoderLayer(d_model=d_model, nhead=2)
        #self.encoder = nn.TransformerEncoder(self.enclayer, num_layers=num_layers)
        self.encoder = nn.ModuleList([
            XLNetEncoderLayer(d_model=d_model, nhead=nhead) for _ in range(num_layers)
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

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        r = self.relative_positional_encoding(input2.size(0),input2.size(0),input.size(0))
        klen = input2.shape[0]
        rpos = torch.arange(self.maxlen-klen, self.maxlen+klen, device=input.device)
        # r = self.relembed(rpos[:,None].expand(2*klen,input2.shape[1]))
        src = self.embedding(input2)
        h,g = (src, src)
        for layer in self.encoder:
            h,g = layer(h,g,r)
        #out = torch.cat(out,dim=-1)
        return self.fc(g).permute(1,2,0)

class GRUAE(nn.Module):
    def __init__(self, model_size=512, nhead=4, maxlen=256, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        assert model_size % 2 == 0
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.cell = nn.GRU(model_size, model_size//2, 4,\
            bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(model_size, vocab_size)
    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)
        src = self.cell(self.embedding(input2))[0]
        return self.fc(src).permute(1,2,0)

# From https://github.com/pytorch/fairseq/blob/master/fairseq/models/lstm.py
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = nn.Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids):
        # input: tgtlen x bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        #attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        attn_scores = torch.einsum('sbh,tbh->tsb', source_hids, x)

        attn_scores = F.softmax(attn_scores, dim=1)  # srclen x bsz

        # sum weighted sources
        #x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        x = torch.einsum('tsb, sbh->tbh',attn_scores, source_hids)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=-1)))
        return x, attn_scores

class LSTMAE(nn.Module):
    def __init__(self, model_size, num_layers=1, vocab_size=16, bidirectional=True):
        super().__init__()
        
        self.model_size = model_size
        self.embed = nn.Embedding(vocab_size, self.model_size)
        if bidirectional:
            assert model_size %2 == 0
            self.encoder = nn.LSTM(self.model_size, self.model_size//2, num_layers=num_layers//2, bidirectional=True)
            self.decoder = nn.LSTM(self.model_size, self.model_size//2, num_layers=num_layers//2, bidirectional=True)
        else:
            self.encoder = nn.LSTM(self.model_size, self.model_size, num_layers=num_layers//2, bidirectional=False)
            self.decoder = nn.LSTM(self.model_size, self.model_size, num_layers=num_layers//2, bidirectional=False)
        self.dropout = nn.Dropout(0.1)
        self.attn = AttentionLayer(model_size, model_size, model_size)
        self.fc = nn.Linear(model_size, vocab_size)

    def forward(self, input):
        outputs = self.dropout(self.embed(input.permute(1,0)))
        outputs, state = self.encoder(outputs)
        outputs, _ = self.attn(outputs, outputs)
        outputs, state = self.decoder(self.dropout(outputs))
        return self.fc(self.dropout(outputs)).permute(1,2,0)

class DNCAE(nn.Module):
    def __init__(self, model_size=64, nhead=4, num_layers=2, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.cell = DNC(input_size=model_size,
            hidden_size=model_size,
            batch_first=True,
            bidirectional=False,
            num_hidden_layers=num_layers,
            nr_cells=5,
            read_heads=nhead,
            cell_size=16,
            gpu_id=0,
        )
        self.fc = nn.Linear(model_size, vocab_size)
    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        src = self.cell(self.embedding(input))[0] #N, S, C
        return self.fc(src).permute(0,2,1)

class UTAE(nn.Module):
    def __init__(self, model_size=64, nhead=4, num_layers=2, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.cell = UTransformer.Encoder(model_size, model_size, num_layers, nhead, 
            total_key_depth=model_size, total_value_depth=model_size, filter_size=model_size, layer_dropout=0.1)
        self.fc = nn.Linear(model_size, vocab_size)
    
    #N,S,C version
    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)


        return pos_emb

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input):
        ipos = torch.arange(input.size(1), device=input.device)

        freq_seq = torch.arange(0, self.model_size, 2.0, dtype=torch.float, device=input.device)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.model_size))

        #pos_emb = self.positional_embedding(ipos, inv_freq)
        #src = self.cell(self.embedding(input) + pos_emb)[0] #N, S, C
        src = self.cell(self.embedding(input))[0] #N, S, C
        return self.fc(src).permute(0,2,1)