import torch
import torch.nn as nn
import numpy as np
import argparse
import Options
import Models
import IBERT
import IBERT2
from NSPDataset import NSPDatasetAE, Token, fib, arith, palindrome
from PBTCDataset import PBTCDataset
from torch.utils.data import Dataset, DataLoader

def train(model, trainloader, criterion, optimizer, scheduler):
        model.train(mode=True)
        tcorrect = 0
        tlen     = 0
        tloss    = 0
        for x,y in trainloader:
            xdata       = x.cuda()
            ydata       = y.cuda()
            optimizer.zero_grad()
            
            output      = model(xdata)
            
            loss        = criterion(output, ydata)
            loss.backward()
            tloss       = tloss + loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred        = output.argmax(axis=1)

            truth = xdata.argmax(axis=2)
            maskpos = (torch.flatten(truth)!=torch.flatten(ydata))
            seqcorrect = (torch.flatten(pred)==torch.flatten(ydata))

            tcorrect    = tcorrect + seqcorrect.sum().item()
            tlen        = tlen + seqcorrect.nelement()
        scheduler.step()

        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        print('Current LR:' + str(scheduler.get_last_lr()[0]))

        return model


def validate(model, valloader, args):
        vcorrects   = 0
        vlens       = 0
        vloss = 0
        model.train(mode=False)
        for i,(x,y) in enumerate(valloader):
            xdata       = x.cuda()
            ydata2      = y.cuda()
            shard       = (4*i)//len(valloader)
            output      = model(xdata)
            loss        = criterion(output, ydata2)
            vloss       = vloss + loss.item()
            pred2       = output.argmax(axis=1)

            truth = xdata.argmax(axis=2)
            maskpos = (torch.flatten(truth)!=torch.flatten(ydata2))
            seqcorrect = (torch.flatten(pred2)==torch.flatten(ydata2))

            """
            print('pred2')
            print([[valset.ixtoword[c.item()] for c in s]
                for s in pred2[0:2].data.cpu()])
            print('ydata2')
            print([[dataset.ixtoword[c.item()] for c in s]
                for s in ydata2[0:2].data.cpu()])
            """

            vcorrects += seqcorrect[maskpos].sum().item()
            vlens += seqcorrect[maskpos].nelement()
        curshard = args.digits

        print("val accuracy:\t{}".format(vcorrects/vlens))
        print('validation loss:\t{}'.format(vloss/len(valloader)))
        return model



if __name__ == '__main__':
    
    args = Options.get_args()


    if args.seq_type == 'fib':
        dataset     = NSPDatasetAE(fib, args.digits, size=args.train_size)
        valset      = NSPDatasetAE(fib, args.digits+3, args.digits, size=args.validation_size)
    elif args.seq_type == 'arith':
        dataset     = NSPDatasetAE(arith, args.digits, size=args.train_size)
        valset      = NSPDatasetAE(arith, args.digits+3, args.digits, size=args.validation_size)
    elif args.seq_type == 'palin':
        dataset     = NSPDatasetAE(palindrome, args.digits, numbers=1, size=args.train_size)
        valset      = NSPDatasetAE(palindrome, args.digits+3, args.digits, numbers=1, size=args.validation_size)
    elif args.seq_type == 'pbtc':
        dataset     = PBTCDataset('train', minSeq = 16, maxSeq = 64) 
        valset      = PBTCDataset('test', minSeq = 64, maxSeq = 128) 
    else :
        print('Sequence type {} not supported yet'.format(args.seq_type))
        exit()

    if args.seq_type == 'pbtc': 
        vocab_size = dataset.vocab_size
    else:
        vocab_size = 16

    if args.net == 'tf':
        print('Executing Autoencoder model with TfAE Model')
        model = Models.TfAE(args.model_size, nhead=args.num_heads, num_layers=args.num_layers, vocab_size = vocab_size).cuda()
    elif args.net == 'cnn':
        print('Executing Autoencoder model with CNNAE Model')
        model = Models.CNNAE(args.model_size, vocab_size = vocab_size).cuda()
    elif args.net == 'xlnet':
        print('Executing Autoencoder model with XLNet-like Model')
        model = Models.XLNetAE(args.model_size, vocab_size = vocab_size, num_layers=args.num_layers, nhead=args.num_heads).cuda()
    elif args.net == 'ibert':
        print('Executing Autoencoder model with IBERT\'s Architecture')
        model = IBERT.IBERTAE(args.model_size, vocab_size = vocab_size, num_layers=args.num_layers, nhead=args.num_heads).cuda()
    elif args.net == 'ibertpos':
        print('Executing Autoencoder model with IBERT+Pos\'s Architecture')
        model = IBERT.IBERTPosAE(args.model_size, vocab_size = vocab_size, num_layers=args.num_layers, nhead=args.num_heads).cuda()
    elif args.net == 'ibert2':
        print('Executing Autoencoder model with IBERT2\'s Architecture')
        model = IBERT2.IBERT2AE(args.model_size, vocab_size = vocab_size, nhead=args.num_heads).cuda()
    elif args.net == 'gru':
        print('Executing Autoencoder model with GRU w.o. Attention')
        model = Models.GRUAE(args.model_size, vocab_size = vocab_size).cuda()
    else :
        print('Network {} not supported'.format(args.net))
        exit()

    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader   = DataLoader(valset, batch_size=args.batch_size, num_workers=2)
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
    criterion   = nn.CrossEntropyLoss()

    for e in range(args.epochs):
        print('\nEpoch #{}:'.format(e+1))
        
        #train the model
        model = train(model, trainloader, criterion, optimizer, scheduler)

        #validate the model
        model = validate(model, valloader, args)

    print('Done')
