import torch
import torch.nn as nn
import XLNet

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

class XLNetEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", maxlen=256):
        super().__init__()
        self.self_attn = XLNet.XLNetRelativeAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.maxlen = maxlen
        #self.posembed = nn.Embedding(2*maxlen, d_model)
        self.activation = nn.ReLU()

    def forward(self, h,g,r, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: packed two-stream h,g.
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        #h,g,r = src
        #klen = h.shape[0]
        #ipos = torch.arange(self.maxlen-klen, self.maxlen+klen, device=h.device)
        #r = self.posembed(ipos[:,None].expand(2*klen,h.shape[1]))
        h2, g2 = self.self_attn(h, g, r)
        h = h + self.dropout1(h2)
        g = g + self.dropout1(g2)
        h = self.norm1(h)
        g = self.norm1(g)
        if hasattr(self, "activation"):
            h2 = self.linear2(self.dropout(self.activation(self.linear1(h))))
            g2 = self.linear2(self.dropout(self.activation(self.linear1(g))))
        else:  # for backward compatibility
            h2 = self.linear2(self.dropout(F.relu(self.linear1(h))))
            g2 = self.linear2(self.dropout(F.relu(self.linear1(g))))
        h = h + self.dropout2(h2)
        g = g + self.dropout2(g2)
        h = self.norm2(h)
        g = self.norm2(g)
        return h,g

