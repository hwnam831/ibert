import torch
import torch.nn as nn
import numpy as np
import argparse
import Options
import Models
import Nam
import Vikram
from NSPDataset import NSPDatasetAE, Token, fib, arith, palindrome
from PTBCDataset import PTBCDataset
from PTBWDataset import PTBWDataset
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
            seqcorrect  = (pred==ydata).prod(-1)
            tcorrect    = tcorrect + seqcorrect.sum().item()
            tlen        = tlen + seqcorrect.nelement()
        scheduler.step()

        trainingResult = list()
        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        print('Current LR:' + str(scheduler.get_last_lr()[0]))
        trainingResult.append('train seq acc:\t'+str(tcorrect/tlen))
        trainingResult.append(str('train loss:\t{}'.format(tloss/len(trainloader))))
        trainingResult.append('Current LR:' + str(scheduler.get_last_lr()[0]))


        return model, trainingResult


def validate(model, valloader, args):
        vcorrects   = [0 for i in range(args.digits, args.digits+4)]
        vlens       = [0 for i in range(args.digits, args.digits+4)]
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
            seqcorrect  = (pred2==ydata2).prod(-1)
            vcorrects[shard] = vcorrects[shard] + seqcorrect.sum().item()
            vlens[shard]     = vlens[shard] + seqcorrect.nelement()
        curshard = args.digits

        accuracyResult = list()
        for vc,vl in zip(vcorrects, vlens):
            print("val accuracy at {} digits = {}".format(curshard,vc/vl))
            accuracyResult.append("val accuracy at {} digits = {}".format(curshard,vc/vl))
            curshard = curshard + 1
        print('validation loss:\t{}'.format(vloss/len(valloader)))
        accuracyResult.append('validation loss:\t{}'.format(vloss/len(valloader)))
        

        return model, accuracyResult

def logger(args, epoch, contents):
    with open(str(args.seq_type) + " " + str(args.net) +".log", "a+") as fd:
        fd.write('\nEpoch #{}:'.format(epoch))
        fd.write('\n')
        for sen in contents:
            fd.write(sen)
            fd.write('\n')


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
    elif args.seq_type == 'ptbc':
        dataset     = PTBCDataset('train', minSeq = 16, maxSeq = 128) 
        valset      = PTBCDataset('test', minSeq = 128, maxSeq = 192) 
    elif args.seq_type == 'ptbw':
        dataset     = PTBWDataset('train', minSeq = 8, maxSeq = 32) 
        valset      = PTBWDataset('test', minSeq = 32, maxSeq = 64) 
    else :
        print('Sequence type {} not supported yet'.format(args.seq_type))
        exit()

    if args.seq_type == 'ptbc': 
        vocab_size = dataset.vocab_size
    elif args.seq_type == 'ptbw': 
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
    elif args.net == 'nam':
        print('Executing Autoencoder model with Nam\'s Architecture')
        model = Nam.GRUTFAE(args.model_size, vocab_size = vocab_size, num_layers=args.num_layers, nhead=args.num_heads).cuda()
    elif args.net == 'vikram':
        print('Executing Autoencoder model with Vikram\'s Architecture')
        model = Vikram.VikramAE(args.model_size, vocab_size = vocab_size, nhead=args.num_heads).cuda()
    elif args.net == 'gru':
        print('Executing Autoencoder model with GRU w.o. Attention')
        model = Models.GRUAE(args.model_size, vocab_size = vocab_size).cuda()
    elif args.net == 'lstm':
        print('Executing Autoencoder model with LSTM including Attention')
        model = Nam.LSTMAE(args.model_size, vocab_size = vocab_size).cuda()
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
        model, result1 = train(model, trainloader, criterion, optimizer, scheduler)

        #validate the model
        model, result2 = validate(model, valloader, args)
        
        #save into logfile
        result1.extend(result2)
        logger(args, e+1, result1)

    print('Done')