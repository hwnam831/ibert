import torch
import torch.nn as nn
from Encoder import XLNetEncoderLayer
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_

class TwoStreamAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()

        if d_model % n_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (d_model, n_head)
            )
        
        self.output_attentions = False
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_model = d_model
        self.scale = 1 / (self.d_head ** 0.5)

        self.q = nn.Parameter(torch.FloatTensor(d_model, self.n_head, self.d_head))
        self.k = nn.Parameter(torch.FloatTensor(d_model, self.n_head, self.d_head))
        self.v = nn.Parameter(torch.FloatTensor(d_model, self.n_head, self.d_head))
        self.o = nn.Parameter(torch.FloatTensor(d_model, self.n_head, self.d_head))

        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        xavier_uniform_(self.q)
        xavier_uniform_(self.k)
        xavier_uniform_(self.v)
        xavier_uniform_(self.o)

        xavier_normal_(self.r_w_bias)

    def rel_attn_core(self, q_head, k_head_h, v_head_h, seg_mat=None, attn_mask=None, head_mask=None):
        """Core relative positional attention operations."""

        # content based attention score
        ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)

        # merge attention scores and perform masking
        attn_score = ac * self.scale

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

    def forward(self, h, g):

            # Two-stream attention with relative positional encoding.
            # content based attention score
        cat = h

        # content-based key head
        k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)

        # content-based value head
        v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)


        # h-stream
        # content-stream query head
        q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)

        # core attention ops
        attn_vec_h = self.rel_attn_core(
            q_head_h, k_head_h, v_head_h
        )

        # post processing
        output_h = self.post_attention(h, attn_vec_h)

        # g-stream
        # query-stream query head
        q_head_g = torch.einsum("ibh,hnd->ibnd", g, self.q)


        attn_vec_g = self.rel_attn_core(
            q_head_g, k_head_h, v_head_h
        )


        # post processing
        output_g = self.post_attention(g, attn_vec_g)


        outputs = (output_h, output_g)
        return outputs

class TwoStreamEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = TwoStreamAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, h,g):
        h2, g2 = self.self_attn(h, g)
        h = h + self.dropout1(h2)
        g = g + self.dropout1(g2)
        h = self.norm1(h)
        g = self.norm1(g)
        h2 = self.linear2(self.dropout(F.relu(self.linear1(h))))
        g2 = self.linear2(self.dropout(F.relu(self.linear1(g))))
        h = h + self.dropout2(h2)
        g = g + self.dropout2(g2)
        h = self.norm2(h)
        g = self.norm2(g)
        return h,g


class GRUEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        assert d_model % 2 == 0
        self.cell = nn.GRU(d_model, d_model//2, 1, bidirectional=True)
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


class GRUTFAE(nn.Module):
    def __init__(self, model_size=512, nhead=4, num_layers=12, vocab_size=16, dropout=0.2):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        assert model_size % 2 == 0
        self.embedding = nn.GRU(vocab_size, model_size//2, 1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.enclayer = nn.TransformerEncoderLayer(d_model=model_size, nhead=nhead, dropout=dropout)
        #self.enclayer = NamEncoderLayer(d_model=model_size, nhead=nhead, dropout=dropout)
        self.norm = nn.LayerNorm(model_size)
        self.tfmodel = nn.TransformerEncoder(self.enclayer, \
            num_layers=num_layers, norm=self.norm)
        self.fc = nn.Linear(model_size, vocab_size)
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0,2)
        src = self.embedding(input2)[0]
        out = self.tfmodel(self.dropout(src))
        return self.fc(out).permute(1,2,0)

class NamAE(nn.Module):
    def __init__(self, model_size=512, nhead=4, num_layers=12, vocab_size=16, dropout=0.2):
        super().__init__()
        self.model_size=model_size
        self.vocab_size = vocab_size
        assert model_size % 2 == 0
        self.embedding = nn.GRU(vocab_size, model_size//2, 1, bidirectional=True)
        self.embedding2 = nn.Linear(vocab_size, model_size)
        self.dropout = nn.Dropout(dropout)
        #self.enclayer = TwoStreamEncoderLayer(d_model=model_size, nhead=nhead, dropout=dropout)
        #self.enclayer = nn.TransformerEncoderLayer(d_model=model_size, nhead=nhead, dropout=dropout)
        self.norm = nn.LayerNorm(model_size)
        self.encoder = nn.ModuleList([
            TwoStreamEncoderLayer(d_model=model_size, nhead=nhead, dropout=dropout) for _ in range(num_layers)
        ])
        self.layers = nn.ModuleList
        self.fc = nn.Linear(model_size, vocab_size)
    #Batch-first in (N,S,C), batch-first out (N,C,S)
    def forward(self, input):
        input2 = input.permute(1,0,2)
        src = self.embedding(input2)[0]
        h,g = (self.dropout(self.embedding2(input2)), self.dropout(src))
        for layer in self.encoder:
            h,g = layer(h,g)
        out = self.dropout(g)
        return self.fc(out).permute(1,2,0)

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
    def __init__(self, model_size, vocab_size=16):
        super().__init__()
        assert model_size %2 == 0
        self.model_size = model_size
        self.embed = nn.Linear(vocab_size, self.model_size)
        self.encoder = nn.LSTM(self.model_size, self.model_size//2, 1, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.attn = AttentionLayer(model_size, model_size, model_size)
        self.decoder = nn.LSTM(self.model_size, self.model_size//2, 1, bidirectional=True)
        self.fc = nn.Linear(model_size, vocab_size)

    def forward(self, input):
        outputs = self.dropout(self.embed(input.permute(1,0,2)))
        outputs, state = self.encoder(outputs)
        outputs, _ = self.attn(outputs, outputs)
        outputs, state = self.decoder(self.dropout(outputs))
        return self.fc(self.dropout(outputs)).permute(1,2,0)

