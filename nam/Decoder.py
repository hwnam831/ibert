import torch
import torch.nn as nn
from NSPDataset import Token

class RNNDecoder(nn.Module):
    def __init__(self, model_size, maxlen, cell='gru'):
        super(RNNDecoder, self).__init__()
        self.size = model_size
        self.fc = nn.Linear(self.size, 16)
        self.embed = nn.Linear(16,self.size)
        self.cellname = cell
        self.maxlen = maxlen
        assert cell in ['gru', 'lstm', 'rnn']
        if cell == 'lstm':
            self.cell = nn.LSTM(self.size, self.size, 2, 
                dropout=0.3)
        elif cell == 'gru':
            self.cell = nn.GRU(self.size, self.size, 2, 
                dropout=0.3)
        else:
            self.cell = nn.RNN(self.size, self.size, 2, 
                dropout=0.3)

    def forward(self, inputs, state):
        
        if inputs:
            embedded = self.embed(inputs.view(-1,16)).view(inputs.size(0),-1,self.size)
            outputs, hidden = self.cell(embedded, state)
            logits = self.fc(outputs.view(-1,self.size))
        else:
            logit = torch.zeros(state.size(0), 16, device=state.device)
            logit[:,Token.start] = 1
            logits = torch.zeros(self.maxlen, state.size(0), 16, device=state.device)
            hidden = state
            for i in range(self.maxlen):
                embedded = self.embed(logit).view(1,-1,self.size)
                output, hidden = self.cell(embedded, hidden)
                logit = self.fc(output.view(-1,self.size))
                logits[i] = logit
        return logits, hidden

class FCDecoder(nn.Module):
    def __init__(self, model_size):
        super(RNNDecoder, self).__init__()
        self.size = model_size
        self.fc = nn.Linear(self.size, 16)
        self.embed = nn.Linear(self.size,self.size)

    def forward(self, inputs):
        out = self.embed(inputs.view(-1,self.size))
        logits = self.fc(torch.relu(out)).view(inputs.size(0),-1,16)
        return logits