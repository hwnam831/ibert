import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import Options
import Models
import IBERT
import IBERT2
from NSPDataset import NSPDatasetAE, NSPDatasetAE2, StringDataset, Token, fib, arith, palindrome, copy
from PTBCDataset import PTBCDataset
from PTBWDataset import PTBWDataset
from AttentionMatrix import AMEncoder, AMIBERT, LinearAttention, RecurrentAM
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
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
    with open(str("log/") + str(args.exp) + " " + str(time.strftime("%Y-%m-%d %H:%M:%S", timestamp)) + " "+ str(args.seq_type) + " " + str(args.net) +".log", "a+") as fd:
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
    elif args.seq_type == 'copy' or args.seq_type == 'palin':
        dataset     = StringDataset(args.seq_type, args.digits, size=args.train_size)
        valset      = StringDataset(args.seq_type, args.digits+4, args.digits+1, size=args.validation_size)
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


    if args.net == 'tf':
        print('Executing Autoencoder model with TfAE Model')
        model = Models.TfAE(dmodel, nhead=nhead, num_layers=num_layers, vocab_size = vocab_size).cuda()
    elif args.net == 'cnn':
        print('Executing Autoencoder model with CNNAE Model')
        model = Models.CNNAE(dmodel, vocab_size = vocab_size).cuda()
    elif args.net == 'xlnet':
        print('Executing Autoencoder model with XLNet-like Model')
        model = Models.XLNetAE(dmodel, vocab_size = vocab_size, num_layers=num_layers, nhead=nhead).cuda()
    elif args.net == 'ibert':
        print('Executing Autoencoder model with IBERT\'s Architecture')
        model = IBERT.IBERTAE(dmodel, vocab_size = vocab_size, num_layers=num_layers, nhead=nhead, bidirectional=args.bidirectional).cuda()
    elif args.net == 'ibertpos':
        print('Executing Autoencoder model with IBERT+Pos\'s Architecture')
        model = IBERT.IBERTPosAE(dmodel, vocab_size = vocab_size, num_layers=num_layers, nhead=nhead, bidirectional=args.bidirectional).cuda()
    elif args.net == 'ibert2':
        print('Executing Autoencoder model with IBERT2\'s Architecture')
        model = AMIBERT(dmodel, vocab_size = vocab_size, nhead=nhead, num_layers=num_layers, bidirectional=args.bidirectional).cuda()
    elif args.net == 'gru':
        print('Executing Autoencoder model with GRU w.o. Attention')
        model = Models.GRUAE(dmodel, vocab_size = vocab_size).cuda()
    elif args.net == 'lstm':
        print('Executing Autoencoder model with LSTM including Attention')
        model = Models.LSTMAE(dmodel, vocab_size = vocab_size, bidirectional=args.bidirectional).cuda()
    elif args.net == 'nam':
        print('Executing NAM Autoencoder model')
        model = AMEncoder(dmodel, nhead=nhead, num_layers=num_layers, vocab_size=vocab_size, attn=RecurrentAM).cuda()
    elif args.net == 'linear':
        print('Executing Linear Attention Autoencoder model')
        model = AMEncoder(dmodel, nhead=nhead, num_layers=num_layers, vocab_size=vocab_size, attn=LinearAttention).cuda()
    elif args.net == 'dnc':
        print('Executing DNC model')
        model = Models.DNCAE(dmodel, nhead, vocab_size=vocab_size).cuda()
    elif args.net == 'ut':
        print('Executing Universal Transformer model')
        model = Models.UTAE(dmodel, nhead, vocab_size=vocab_size).cuda()
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
    #torch.autograd.set_detect_anomaly(True)
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
