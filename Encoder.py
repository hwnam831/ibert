import torch
import torch.nn as nn
import MiniBert
from MiniBert import MiniBertEncoder, MiniBertConfig

class RNNEncoder(nn.Module):
    def __init__(self, model_size, cell='gru', bidirectional=False):
        super(RNNEncoder, self).__init__()
        self.size = model_size
        self.embed = nn.Linear(16, self.size)
        self.num_direction = 2 if bidirectional else 1
        self.cellname = cell
        assert cell in ['gru', 'lstm', 'rnn']
        if cell == 'lstm':
            self.cell = nn.LSTM(self.size, self.size, 2,
                           dropout=0.3, bidirectional=bidirectional)
        elif cell == 'gru':
            self.cell = nn.GRU(self.size, self.size, 2,
                           dropout=0.3, bidirectional=bidirectional)
        else:
            self.cell = nn.RNN(self.size, self.size, 2,
                           dropout=0.3, bidirectional=bidirectional)

    def forward(self, input):
        outputs = self.embed(input.view(-1,16))
        outputs, state = self.cell(outputs.view(input.size(0), -1, self.size))
        return outputs, state
