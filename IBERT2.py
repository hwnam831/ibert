import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_

class RelativeAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        if d_model % n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (d_model, n_head)
            )
        
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_model = d_model
        self.scale = 1 / (self.d_head ** 0.5)

        self.q = nn.Parameter(torch.FloatTensor(d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(d_model, self.n_head, self.d_head))
        self.r = nn.Parameter(torch.FloatTensor(d_model, self.n_head, self.d_head))

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #To use the parameters, we need initializations
        xavier_uniform_(self.q)
        xavier_uniform_(self.k)
        xavier_uniform_(self.v)
        xavier_uniform_(self.o)
        xavier_uniform_(self.r)

        xavier_normal_(self.r_r_bias)
        xavier_normal_(self.r_w_bias)

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        #x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        x = x[:, :, :, :klen]

        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r):
        """Core relative positional attention operations."""

        # content based attention score
        ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # merge attention scores and perform masking
        attn_score = (ac + bd) * self.scale
        
        # attention probability
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropout(attn_prob)

        # attention output
        attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        # post-attention projection (back to `d_model`)
        attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def forward(self, h, r):

        # content heads
        q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        k_head_h = torch.einsum("ibh,hnd->ibnd", h, self.k)
        v_head_h = torch.einsum("ibh,hnd->ibnd", h, self.v)

        # positional heads
        k_head_r = torch.einsum("ibh,hnd->ibnd", r, self.r)

        # core attention ops
        attn_vec = self.rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r)

        # post processing
        output_h = self.post_attention(h, attn_vec)

        return output_h

class IBERTEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", maxlen=128):
        super().__init__()
        self.self_attn = RelativeAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.maxlen = maxlen
        self.posembed = nn.Embedding(2*maxlen, d_model)
        self.activation = nn.ReLU()

    def forward(self, h, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: packed two-stream h,g.
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        klen = h.shape[0]
        ipos = torch.arange(self.maxlen-klen, self.maxlen+klen, device=h.device)
        r = self.posembed(ipos[:,None].expand(2*klen,h.shape[1]))
        h2 = self.self_attn(h, r)
        h = h + self.dropout1(h2)
        h = self.norm1(h)
        h2 = self.linear2(self.dropout(F.relu(self.linear1(h))))
        h = h + self.dropout2(h2)
        h = self.norm2(h)

        return h

class VikEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        assert d_model % 2 == 0
        self.cell = nn.LSTM(d_model, d_model//2, 1, bidirectional=True)
        # Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.cell(src)[0]
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class IBERT2AE(nn.Module):
    def __init__(self, model_size=512, num_layers=6, nhead=8, maxlen=128, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        assert model_size % 2 == 0
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, model_size),
            nn.LSTM(model_size, model_size//2, 1, bidirectional=True)
        )
        self.posembed = nn.Embedding(maxlen, model_size)
        self.enclayer = VikEncoderLayer(d_model=model_size, nhead=nhead)
        self.norm = nn.LayerNorm(model_size)
        self.tfmodel = nn.TransformerEncoder(self.enclayer, \
            num_layers=num_layers, norm=self.norm)
        self.fc = nn.Linear(model_size, vocab_size)
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0)
        #ipos = torch.arange(input2.size(0), device=input.device)[:,None].expand(input2.shape[:2])
        #src = self.embedding(input2)[0] + self.posembed(ipos)
        src = self.embedding(input2)[0]
        out = self.tfmodel(src)
        return self.fc(out).permute(1,2,0)

class IBERTAE(nn.Module):
    def __init__(self, model_size=512, maxlen=128, vocab_size=16):
        super().__init__()
        self.model_size=model_size
        self.maxlen=maxlen
        self.vocab_size = vocab_size
        self.embedding = nn.Linear(vocab_size, model_size)
        self.posembed = nn.Embedding(maxlen, model_size)
        self.enclayer = IBERTEncoderLayer(d_model=model_size, nhead=2)
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
