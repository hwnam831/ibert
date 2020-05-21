import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import Options
import Models
import Nam
import Vikram
from NSPDataset import NSPDatasetAE, NSPDatasetAE2, Token, fib, arith, palindrome, copy
from PTBCDataset import PTBCDataset
from PTBWDataset import PTBWDataset
from torch.utils.data import Dataset, DataLoader
import time
import math

def train(model, trainloader, criterion, optimizer, scheduler):
        model.train(mode=True)
        tcorrect = 0
        tlen     = 0
        tloss    = 0
        bits = 0.0
        maskcount = 0
        for x,y in trainloader:
            xdata       = x.cuda()
            ydata       = y.cuda()
            optimizer.zero_grad()
            output      = model(xdata)

            ismask = xdata != ydata
            maskcount += ismask.sum().item()

            loss        = criterion(output, ydata)
            loss.mean().backward()
            bits += (loss*ismask).sum().item()

            tloss       = tloss + loss.mean().item()
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
        
        #Perplexity  = 2^bit
        print('Training Perplexity :\t{}'.format(math.exp((bits/maskcount) * math.log(2)))) 
        trainingResult.append('Training Perplexity :\t{}'.format(math.exp((bits/maskcount) * math.log(2))))
       


        return model, trainingResult


def validate(model, valloader, args):
        vcorrects   = [0 for i in range(args.digits+1, args.digits+5)]
        vlens       = [0 for i in range(args.digits+1, args.digits+5)]
        vloss = 0
        model.train(mode=False)
        bits = 0.0
        maskcount = 0
        with torch.no_grad():
            for i,(x,y) in enumerate(valloader):
                xdata       = x.cuda()
                ydata2      = y.cuda()
                shard       = (4*i)//len(valloader)
                output      = model(xdata)
                # xdata <- masked index
                # ydata2 <- answer 
                ismask = xdata != ydata2
                mcnt = ismask.sum().item()
                loss        = F.cross_entropy(output, ydata2, reduction='none')
                vloss       = vloss + loss.mean().item()
                bits += (loss*ismask).sum().item()
                maskcount += mcnt
                pred2       = output.argmax(axis=1)
                seqcorrect  = (pred2==ydata2).prod(-1)
                vcorrects[shard] = vcorrects[shard] + seqcorrect.sum().item()
                vlens[shard]     = vlens[shard] + seqcorrect.nelement()
        curshard = args.digits+1
            
        accuracyResult = list()
        for vc,vl in zip(vcorrects, vlens):
            print("val accuracy at {} digits = {}".format(curshard,vc/vl))
            accuracyResult.append("val accuracy at {} digits = {}".format(curshard,vc/vl))
            curshard = curshard + 1
        
        #Sequence accuracy
        print('validation loss:\t{}'.format(vloss/len(valloader)))
        accuracyResult.append('validation loss:\t{}'.format(vloss/len(valloader)))
        
        #Perplexity  = 2^bit
        print('Perplexity :\t{}'.format(math.exp((bits/maskcount) * math.log(2)))) 
        accuracyResult.append('Perplexity :\t{}'.format(math.exp((bits/maskcount) * math.log(2))))

        return model, accuracyResult

def logger(args, timestamp, epoch, contents):
    with open(str("log/") + str(time.strftime("%Y-%m-%d %H:%M:%S", timestamp)) + " " + str(args.seq_type) + " " + str(args.net) +".log", "a+") as fd:
        fd.write('\nEpoch #{}:'.format(epoch))
        fd.write('\n')
        # print model information
        if epoch == 0:
            fd.write(contents)
            fd.write('\n')
            return
        # print experiment result
        for sen in contents:
            fd.write(sen)
            fd.write('\n')

if __name__ == '__main__':
    
    args = Options.get_args()

    if args.seq_type == 'fib':
        dataset     = NSPDatasetAE2(fib, args.digits, size=args.train_size)
        valset      = NSPDatasetAE2(fib, args.digits+4, args.digits+1, size=args.validation_size)
    elif args.seq_type == 'arith':
        dataset     = NSPDatasetAE2(arith, args.digits, size=args.train_size)
        valset      = NSPDatasetAE2(arith, args.digits+4, args.digits+1, size=args.validation_size)
    elif args.seq_type == 'palin':
        dataset     = NSPDatasetAE2(palindrome, args.digits, numbers=1, size=args.train_size)
        valset      = NSPDatasetAE2(palindrome, args.digits+4, args.digits+1, numbers=1, size=args.validation_size)
    elif args.seq_type == 'copy':
        dataset     = NSPDatasetAE2(copy, args.digits, size=args.train_size)
        valset      = NSPDatasetAE2(copy, args.digits+4, args.digits+1, size=args.validation_size)
    elif args.seq_type == 'ptbc':
        dataset     = PTBCDataset('train', minSeq = 16, maxSeq = 192) 
        valset      = PTBCDataset('train', minSeq = 192, maxSeq = 224) 
    elif args.seq_type == 'ptbw':
        dataset     = PTBWDataset('train', minSeq = 2, maxSeq = 32) 
        valset      = PTBWDataset('train', minSeq = 32, maxSeq = 64) 
    else :
        print('Sequence type {} not supported yet'.format(args.seq_type))
        exit()

    if args.seq_type == 'ptbc': 
        vocab_size = dataset.vocab_size
        dictionary = dataset.wordtoix
    elif args.seq_type == 'ptbw': 
        vocab_size = dataset.vocab_size
        dictionary = dataset.wordtoix
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
        model = Nam.NamAE(args.model_size, vocab_size = vocab_size, num_layers=args.num_layers, nhead=args.num_heads).cuda()
    elif args.net == 'nampos':
        print('Executing Autoencoder model with Nam+Pos\'s Architecture')
        model = Nam.NamPosAE(args.model_size, vocab_size = vocab_size, num_layers=args.num_layers, nhead=args.num_heads).cuda()
    elif args.net == 'vikram':
        print('Executing Autoencoder model with Vikram\'s Architecture')
        model = Vikram.VikramAE(args.model_size, vocab_size = vocab_size, nhead=args.num_heads, num_layers=args.num_layers).cuda()
    elif args.net == 'gru':
        print('Executing Autoencoder model with GRU w.o. Attention')
        model = Models.GRUAE(args.model_size, vocab_size = vocab_size).cuda()
    elif args.net == 'lstm':
        print('Executing Autoencoder model with LSTM including Attention')
        model = Nam.LSTMAE(args.model_size, vocab_size = vocab_size).cuda()
    else :
        print('Network {} not supported'.format(args.net))
        exit()
    print(model)

    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader   = DataLoader(valset, batch_size=args.batch_size, num_workers=2)
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
    criterion   = nn.CrossEntropyLoss(reduction='none')
    nsamples = len(dataset)
    if args.log == 'true':
        ts = time.gmtime()
        logger(args, ts, 0, str(model))
    for e in range(args.epochs):
        print('\nEpoch #{}:'.format(e+1))
        
        trainstart = time.time()
        #train the model
        model, trainResult = train(model, trainloader, criterion, optimizer, scheduler)
        print("Train sequences per second : " + str(nsamples/(time.time()-trainstart)))
        trainResult.append("Train sequences per second : " + str(nsamples/(time.time()-trainstart)))

        #validate the model
        model, valResult = validate(model, valloader, args)
        
        if args.log == 'true':
            #save into logfile
            trainResult.extend(valResult)
            logger(args, ts, e+1, trainResult)

    print('Done')