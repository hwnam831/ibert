import torch
import torch.nn as nn
import numpy as np
import Encoder
from NSPDataset import NSPDataset, Token, fib
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    model = Encoder.CNNAutoEncoder(256).cuda()
    dataset = NSPDataset(fib, 6)
    valset = NSPDataset(fib, 7, 4, size=512)
    trainloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=128, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    epoch = 50
    for e in range(epoch):
        print('\nEpoch #{}:'.format(e+1))
        model.train(mode=True)
        tcorrect = 0
        tlen = 0
        for x,y in trainloader:
            xdata = x.cuda().permute(0,2,1)
            ydata = y.cuda()
            optimizer.zero_grad()
            output = model(xdata)
            loss = criterion(output, ydata)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            pred = output.argmax(axis=1)
            seqcorrect = (pred==ydata).prod(-1)
            tcorrect = tcorrect + seqcorrect.sum().item()
            tlen = tlen + seqcorrect.nelement()
        print('train seq acc:\t'+str(tcorrect/tlen))

        model.train(mode=False)
        vcorrect = 0
        vlen = 0
        for x,y in valloader:
            xdata = x.cuda().permute(0,2,1)
            ydata2 = y.cuda()
            output = model(xdata)
            loss = criterion(output, ydata2)
            pred2 = output.argmax(axis=1)
            seqcorrect = (pred2==ydata2).prod(-1)
            vcorrect = vcorrect + seqcorrect.sum().item()
            vlen = vlen + seqcorrect.nelement()
        print("val accuracy = "+str(vcorrect/vlen))
