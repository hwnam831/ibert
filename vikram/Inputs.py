import torch 
import torch.nn as nn
import torch.nn.functional as F


# class PositionalEmbedding(nn.Module):
    
#     def __init__(self, embd):
#         super(PositionalEmbedding, self).__init__()

#         self.embd = embd
#         inv_freq  = 1 / (10000* (torch.arrange(0.0, embd, 2.0)/embd))
#         self.register_buffer('inv_freq', inv_freq)

#     def forward(self, pos_seq, bsz=None):
#         sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
#         #
#         pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
#         pos_emb = pos_emb[:, None, :]

#         if bsz is not None:
#             pos_emb = pos_emb.expand(-1, bsz, -1)


def positional_embedding(pos_seq, inv_freq, bsz=None):

    sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    pos_emb = pos_emb[:, None, :]

    if bsz is not None:
        pos_emb = pos_emb.expand(-1, bsz, -1)

    return pos_emb

def relative_positional_encoding(self, qlen, klen, bsz= None):

    freq_seq = torch.arrange(0, self.d_model, 2.0, dtype=torch.float)
    inv_freq = 1/torch.pow(10000, (freq_seq/self.d_model))

    if self.attn_type == "bi":
        beg, end = klen, -qlen
    else if self.attn_type == "uni":
        beg, end = klen, -1
    else:
        raise ValueError("Uknown attn_type")

    
    if self.bi_data:
        fwd_pos_seq = torch.arrange(beg, end, -1.0, dtype=torch.float)
        bwd_pos_seq = torch.arrange(-beg, end, -1.0, dtype=torch.float)

        if self.clamp_len  >0:
            fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

        if self.bsz is not None:
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz //2)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz //2)
        else:
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

        pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb])

    else:
        fwd_pos_seq = torch.arrange(beg, end, -1.0)
        if self.clamp_len >0:
            fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

    pos_emb = pos_emb.to(next(self.parameter()))
    return pos_emb