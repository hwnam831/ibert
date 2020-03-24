import torch
import torch.nn as nn
import numpy as np
import argparse
import Models
from NSPDataset import NSPDataset2, Token, fib
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Number sequence training benchmark")
    parser.add_argument(
            "--net",
            type=str,
            choices=['tf', 'lstm', 'gru'],
            default='tf',
            help='tf | minibert | lstm | gru')
    parser.add_argument(
            "--epochs",
            type=int,
            default='50',
            help='number of epochs')
    parser.add_argument(
            "--train_size",
            type=int,
            default='25600',
            help='number of training examples per epoch')
    parser.add_argument(
            "--validation_size",
            type=int,
            default='1024',
            help='number of validation examples')
    parser.add_argument(
            "--batch_size",
            type=int,
            default='256',
            help='batch size')
    parser.add_argument(
            "--model_size",
            type=int,
            default='512',
            help='internal channel dimension')
    parser.add_argument(
            "--digits",
            type=int,
            default='5',
            help='Max number of digits')
    parser.add_argument(
            "--seq_type",
            type=str,
            choices= ['fib', 'arith', 'palin'],
            default='fib',
            help='fib: fibonacci / arith: arithmetic / palin: palindrome')
    args = parser.parse_args()

    if args.net == 'tf':
        model = Models.TfS2S(args.model_size).cuda()
    else:
        model = Models.TfS2S(args.model_size).cuda()
    dataset = NSPDataset2(fib, args.digits, size=args.train_size)
    valset = NSPDataset2(fib, args.digits+1, args.digits-1, size=args.validation_size)
    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=1.01)
    criterion = nn.CrossEntropyLoss()

    epoch = args.epochs
    for e in range(15):
        print('\nEpoch #{}:'.format(e+1))
        model.train(mode=True)
        tcorrect = 0
        tlen = 0
        tloss = 0
        for x,y in trainloader:
            xdata = x.cuda()
            ydata = y.cuda()
            tgt = torch.ones_like(ydata)*Token.start
            tgt[:,1:] = ydata[:,:-1]
            optimizer.zero_grad()
            output = model(xdata, tgt)
            loss = criterion(output, ydata)
            loss.backward()
            tloss = tloss + loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            pred = output.argmax(axis=1)
            seqcorrect = (pred==ydata).prod(-1)
            tcorrect = tcorrect + seqcorrect.sum().item()
            tlen = tlen + seqcorrect.nelement()
        scheduler.step()
        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        print('Current LR:' + str(scheduler.get_lr()[0]))

        model.train(mode=False)
        vcorrect = 0
        vlen = 0
        for x,y in valloader:
            xdata = x.cuda()
            ydata2 = y.cuda()
            tgt = torch.ones_like(ydata2)*Token.start
            tgt[:,1:] = ydata2[:,:-1]
            output = model(xdata, tgt)
            loss = criterion(output, ydata2)
            pred2 = output.argmax(axis=1)
            seqcorrect = (pred2==ydata2).prod(-1)
            vcorrect = vcorrect + seqcorrect.sum().item()
            vlen = vlen + seqcorrect.nelement()
        print("val accuracy = "+str(vcorrect/vlen))
        #scheduler.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.96)
    for e in range(epoch):
        print('\nEpoch #{}:'.format(e+1))
        model.train(mode=True)
        tcorrect = 0
        tlen = 0
        tloss = 0
        for x,y in trainloader:
            xdata = x.cuda()
            ydata = y.cuda()
            tgt = torch.ones_like(ydata)*Token.start
            tgt[:,1:] = ydata[:,:-1]
            optimizer.zero_grad()
            output = model(xdata, tgt)
            loss = criterion(output, ydata)
            loss.backward()
            tloss = tloss + loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pred = output.argmax(axis=1)
            seqcorrect = (pred==ydata).prod(-1)
            tcorrect = tcorrect + seqcorrect.sum().item()
            tlen = tlen + seqcorrect.nelement()
        scheduler.step()
        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        print('Current LR:' + str(scheduler.get_lr()[0]))

        model.train(mode=False)
        vcorrect = 0
        vlen = 0
        for x,y in valloader:
            xdata = x.cuda()
            ydata2 = y.cuda()
            tgt = torch.ones_like(ydata2)*Token.start
            tgt[:,1:] = ydata2[:,:-1]
            output = model(xdata, tgt)
            loss = criterion(output, ydata2)
            pred2 = output.argmax(axis=1)
            seqcorrect = (pred2==ydata2).prod(-1)
            vcorrect = vcorrect + seqcorrect.sum().item()
            vlen = vlen + seqcorrect.nelement()
        print("val accuracy = "+str(vcorrect/vlen))
        #scheduler.step()
