import torch
import torch.nn as nn

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


def ConvBlockRelu(c_in, c_out, ksize, dilation=1):
    pad = ((ksize-1)//2)*dilation
    return nn.Sequential(
            nn.Conv1d(c_in, c_out, ksize, 1, pad, dilation=dilation),
            nn.BatchNorm1d(c_out),
            nn.ReLU()
            )

def ConvBlock(c_in, c_out, ksize, dilation=1):
    pad = ((ksize-1)//2)*dilation
    return nn.Sequential(
            nn.Conv1d(c_in, c_out, ksize, 1, pad, dilation=dilation),
            nn.BatchNorm1d(c_out)
            )

class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, drop=0.1, dilation=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                ConvBlockRelu(c_in, c_mid, 1),
                ConvBlockRelu(c_mid, c_mid, 5, dilation=dilation),
                ConvBlockRelu(c_mid, c_in, 1),
                nn.Dropout(drop)
                )

    def forward(self, x):
        out = self.block(x)
        out += x
        return out

class CNNAutoEncoder(nn.Module):

    def __init__(self, model_size):
        super(CNNAutoEncoder, self).__init__()
        self.size = model_size
        self.vocab_size = 16
        self.conv1 = ConvBlock(self.vocab_size, self.size, 5)
        self.encoder = nn.Sequential(
            ResBlock(self.size, self.size//2, dilation=2),
            ResBlock(self.size, self.size//4, dilation=4),
            ResBlock(self.size, self.size//2, dilation=2),
            ResBlock(self.size, self.size//2, dilation=4))

        self.classifier = ConvBlock(self.size, self.vocab_size, 1)
        self.indirect = ConvBlock(self.vocab_size, self.size, 5)   

    def forward(self, input):  
        res = self.conv1(input)
        out = self.encoder(res)
        out = nn.functional.relu(out + self.indirect(input))
        return out