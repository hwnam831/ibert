import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def unitnorm(v):
    return v/torch.norm(v,dim=-1).unsqueeze(-1)

def unitrelu(v):
    vec = torch.relu(v)
    eps = 1e-6
    return vec/(torch.norm(vec,dim=-1)+eps).unsqueeze(-1)
def unitelu(v):
    vec = v/torch.norm(v,dim=-1).unsqueeze(-1)
    return F.elu(vec)

def eluunit(v):
    vec = F.elu(v)
    return vec/torch.norm(vec,dim=-1).unsqueeze(-1)

def unitsq(v):
    return v/torch.sqrt(torch.norm(v,dim=-1)).unsqueeze(-1)

def unitmn(v):
    mn = torch.mean(torch.norm(v,dim=-1),dim=0)
    return v/mn.unsqueeze(-1)

def softmaxnorm(v):
    return F.softmax(v,dim=-1)

def tanhnorm(v):
    return torch.tanh(v)/math.sqrt(v.size(-1))
'''
class RecurrentAM(nn.Module):
    def __init__(self, d_model, nhead, sigma=unitnorm):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.dim = d_model//nhead
        self.Wq = nn.Linear(d_model*2, d_model)
        self.Wk = nn.Linear(d_model*2, d_model)
        self.Wv = nn.Linear(d_model*2, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.sigma = sigma
    def forward(self, x):
        #assuming (S,B,C) layout
        B,C = x.size(1), x.size(2)
        r = None
        A = torch.zeros([B, self.nhead, self.dim, self.dim], device=x.device)
        for x_i in x: #(B,C) shapes
            if r is not None:
                x_r = torch.cat([x_i,r],dim=-1)
            else:
                x_r = torch.cat([x_i,torch.zeros_like(x_i)],dim=-1)
            k = self.Wk(x_r).reshape(B,self.nhead,-1) #(S,B,n,Dk/n)
            v = self.Wv(x_r).reshape(B,self.nhead,-1) #(S,B,n,Dv/n)
            #k = self.sigma(k)
            k = unitelu(k)
            A = A + torch.einsum('bnq,bnv->bnvq', k,v)
            q = self.Wq(x_r).reshape(B,self.nhead,-1) #(S,B,n,Dq=Dk/n)
            q = self.sigma(q)
            r = torch.einsum('bnq,bnvq->bnv',q,A).reshape(B,-1)
            #r = torch.matmul(q,A).view(B,-1)
        out = self.Wo(r)
        return out, (k,q)
'''

class RecurrentAM(nn.Module):
    def __init__(self, d_model, nhead, sigma=unitnorm):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_model = d_model
        self.dim = d_model//nhead
        self.Wqkv = nn.GRUCell(d_model*2, d_model*3)
        self.Wo = nn.Linear(d_model, d_model)
        self.sigma = sigma
    def forward(self, x):
        #assuming (S,B,C) layout
        B,C = x.size(1), x.size(2)
        r = None
        A = torch.zeros([B, self.nhead, self.dim, self.dim], device=x.device)
        h = None
        for x_i in x: #(B,C) shapes
            if r is not None:
                x_r = torch.cat([x_i,r],dim=-1)
            else:
                x_r = torch.cat([x_i,torch.zeros_like(x_i)],dim=-1)
            if h is None:
                h = self.Wqkv(x_r)
            else:
                h = self.Wqkv(x_r,h)
            k = unitelu(h[:,self.d_model:2*self.d_model].reshape(B,self.nhead,-1))
            v = h[:,2*self.d_model:3*self.d_model].reshape(B,self.nhead,-1)
            A = A + torch.einsum('bnq,bnv->bnvq', k,v)
            q = h[:,:self.d_model].reshape(B,self.nhead,-1)
            q = self.sigma(q)
            r = torch.einsum('bnq,bnvq->bnv',q,A).reshape(B,-1)
        out = self.Wo(r)
        return out, (k,q)

class AttentionMatrix(nn.Module):
    def __init__(self, d_model, nhead, sigma=unitnorm):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.sigma = sigma
    def forward(self, h):
        #assuming (S,B,C) layout
        S,B = h.size(0), h.size(1)
        k = self.Wk(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dk/n)
        v = self.Wv(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dv/n)
        #k = self.sigma(k)
        k = unitelu(k)
        A = torch.einsum('sbnq,sbnv->bnvq', k,v)
        q = self.Wq(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dq=Dk/n)
        q = self.sigma(q)
        out = torch.einsum('sbnq,bnvq->sbnv',q,A).reshape(S,B,-1)
        out = self.Wo(out)
        return out, (k,q)

class LinearAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.nhead = nhead
    def forward(self, h):
        #assuming (S,N,C) layout
        S,B = h.size(0), h.size(1)
        k = self.Wk(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dk/n)
        v = self.Wv(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dv/n)
        k = F.elu(k)+1
        #k = unitelu(k)
        ksum = k.sum(dim=0) #(B,n, Dk)
        A = torch.einsum('sbnq,sbnv->bnvq', k,v)
        q = self.Wq(h).reshape(S,B,self.nhead,-1) #(S,B,n,Dq=Dk/n)
        q = F.elu(q)+1
        #q = unitnorm(q)
        Z = 1/torch.einsum('sbnq,bnq->sbn',q,ksum)
        out = torch.einsum('sbnq,bnvq,sbn->sbnv',q,A,Z).reshape(S,B,-1)
        out = self.Wo(out)
        return out, (k,q)

class AMEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attn=AttentionMatrix):
        super().__init__()
        self.attn = attn(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, src):
        src2, kq = self.attn(src)
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, kq

class AMEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=6, maxlen=256, vocab_size=16, attn=AttentionMatrix):
        super().__init__()
        self.d_model=d_model
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.posembed = nn.Embedding(maxlen, d_model)
        self.encoder = nn.ModuleList([
            AMEncoderLayer(d_model=d_model, nhead=nhead, attn=attn) for _ in range(num_layers)
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

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, print_kq=False):
        input2 = input.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        h = self.embedding(input2)
        #h = h + self.posembed(ipos)
        for layer in self.encoder:
            h, kq = layer(h)
        #out = torch.cat(out,dim=-1)
        out = self.fc(h).permute(1,2,0)
        if print_kq:
            return out, kq
        else:
            return out

class AMIBERT(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=6, maxlen=256, vocab_size=16, attn=AttentionMatrix):
        super().__init__()
        self.d_model=d_model
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        assert d_model%2 == 0
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LSTM(d_model, d_model//2, 1, bidirectional=True)
        )
        self.posembed = nn.Embedding(maxlen, d_model)
        self.encoder = nn.ModuleList([
            AMEncoderLayer(d_model=d_model, nhead=nhead, attn=attn) for _ in range(num_layers)
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

    #Batch-first in (N,S), batch-first out (N,C,S)
    def forward(self, input, print_kq=False):
        input2 = input.permute(1,0)
        ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        klen = input2.shape[0]
        rpos = torch.arange(self.maxlen-klen, self.maxlen+klen, device=input.device)
        # r = self.relembed(rpos[:,None].expand(2*klen,input2.shape[1]))
        src, _ = self.embedding(input2)
        #h = src + self.posembed(ipos)
        h = src
        for layer in self.encoder:
            h, kq = layer(h)
        #out = torch.cat(out,dim=-1)
        out = self.fc(h).permute(1,2,0)
        if print_kq:
            return out, kq
        else:
            return out