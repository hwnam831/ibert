import torch
import torch.nn as nn
from NSPDataset import Token

class TFDecoder(nn.Module):
    def __init__(self, model_size=512, nhead=2, num_layers=3):
        super().__init__()
        self.model_size=model_size
        self.declayer = nn.TransformerDecoderLayer(d_model=model_size, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.declayer, num_layers=num_layers)
        self.fc = nn.Linear(model_size, 16)
    #Seq-first in (S,N,C), batch-first out (N,C,S)
    def forward(self, target, memory):
        tmask = self.generate_square_subsequent_mask(target.size(0)).to(target.device)
        out = self.decoder(target, memory, tgt_mask=tmask)
        return self.fc(out)
    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class FCDecoder(nn.Module):
    def __init__(self, model_size):
        super().__init__()
        self.size = model_size
        self.fc = nn.Linear(self.size, 16)
        self.embed = nn.Linear(self.size,self.size)

    def forward(self, inputs):
        out = self.embed(inputs.view(-1,self.size))
        logits = self.fc(torch.relu(out)).view(inputs.size(0),-1,16)
        return logits