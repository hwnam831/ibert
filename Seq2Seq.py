import torch
import torch.nn as nn
import numpy as np
import argparse
from Options import get_args
import Models
from NSPDataset import NSPDatasetS2S, Token, fib, arith, palindrome
from torch.utils.data import Dataset, DataLoader
from SCANDataset import SCANDataset

def train(model, trainloader, criterion, optimizer, scheduler):
        model.train(mode=True)
        tcorrect = 0
        tlen     = 0
        tloss    = 0
        for x,y in trainloader:
            xdata       = x.to(DEVICE)
            ydata       = y.to(DEVICE)
            tgt         = torch.ones_like(ydata)*Token.start
            tgt[:,1:]   = ydata[:,:-1]
            optimizer.zero_grad()
            
            output      = model(xdata, tgt)
            loss        = criterion(output, ydata)
            loss.backward()
            tloss       = tloss + loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred        = output.argmax(axis=1)
            seqcorrect  = (pred==ydata).prod(-1)
            tcorrect    = tcorrect + seqcorrect.sum().item()
            tlen        = tlen + seqcorrect.nelement()
        scheduler.step()

        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        print('Current LR:' + str(scheduler.get_lr()[0]))

        return model


def validate(model, valloader, args):
        vcorrects   = [0 for i in range(args.digits-1, args.digits+2)]
        vlens       = [0 for i in range(args.digits-1, args.digits+2)]
        
        model.train(mode=False)
        for i,(x,y) in enumerate(valloader):
            xdata       = x.cuda()
            ydata2      = y.cuda()
            shard       = (3*i)//len(valloader)
            tgt         = torch.ones_like(ydata2)*Token.start
            tgt[:,1:]   = ydata2[:,:-1]
            output      = model(xdata, tgt)
            pred2       = output.argmax(axis=1)
            seqcorrect  = (pred2==ydata2).prod(-1)
            vcorrects[shard] = vcorrects[shard] + seqcorrect.sum().item()
            vlens[shard]     = vlens[shard] + seqcorrect.nelement()
        curshard = args.digits - 1

        for vc,vl in zip(vcorrects, vlens):
            print("val accuracy at {} digits = {}".format(curshard,vc/vl))
            curshard = curshard + 1

        return model



if __name__ == '__main__':
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    args = get_args()

    if args.model_size == 'base':
        dmodel = 768
        nhead = 12
        num_layers = 12
    elif args.model_size == 'mini':
        dmodel = 256
        nhead = 4
        num_layers = 4
    elif args.model_size == 'small':
        dmodel = 512
        nhead = 8
        num_layers = 4
    elif args.model_size == 'medium':
        dmodel = 512
        nhead = 8
        num_layers = 8
    elif args.model_size == 'tiny':
        dmodel = 128
        nhead = 2
        num_layers = 2
    elif args.model_size == 'custom':
        dmodel = 512
        nhead = 4
        num_layers = 4
    else:
        print('shouldnt be here')
        exit(-1)

    print('Executing Seq2Seq model with IBERTS2S Model')
    epoch = args.epochs
    if args.net == 'ibert':
        model       = Models.IBERTS2S(dmodel, nhead=nhead, num_layers=num_layers).to(DEVICE)
    elif args.net == 'lstm':
        model       = Models.LSTMS2S(dmodel, num_layers=num_layers).to(DEVICE)
    else:
        model       = Models.TfS2S(dmodel, nhead=nhead, num_layers=num_layers).to(DEVICE)
    dataset = SCANDataset(splitType='train') # Use among 'train', 'test'
    valset = SCANDataset(splitType='test') # Use among 'train', 'test'
    trainloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    valloader   = DataLoader(valset, batch_size=args.batch_size, num_workers=2)
    optimizer   = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.96)
    criterion   = nn.CrossEntropyLoss()

    for e in range(epoch):
        print('\nEpoch #{}:'.format(e+1))
        
        #train the model
        model = train(model, trainloader, criterion, optimizer, scheduler)

        #validate the model
        model = validate(model, valloader, args)

    print('Done')