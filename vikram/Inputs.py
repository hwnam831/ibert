import torch 
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    
    def __init__(self, embd):
        super(PositionalEmbedding, self).__init__()

        self.embd = embd
        inv_freq  = 1 / (10000* (torch.arrange(0.0, embd, 2.0)/embd))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        #
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else: 
            return pos_emb[:, None, :]