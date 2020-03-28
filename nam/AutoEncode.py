import torch
import torch.nn as nn
import numpy as np
import argparse
import Models
from NSPDataset import NSPDatasetAE, Token, fib
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Number sequence training benchmark")
    parser.add_argument(
            "--net",
            type=str,
            choices=['cnn', 'tf', 'minibert', 'lstm', 'gru'],
            default='tf',
            help='cnn | tf | minibert | lstm | gru')
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

    if args.net == 'minibert':
        model = Models.MiniBertAE(args.model_size).cuda()
    elif args.net == 'cnn':
        model = Models.CNNAutoEncoder(args.model_size).cuda()
    else :
        model = Models.TfAE(args.model_size).cuda()
    dataset = NSPDatasetAE(fib, args.digits, size=args.train_size)
    valset = NSPDatasetAE(fib, args.digits+1, args.digits-1, size=args.validation_size)
    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.002)
    criterion = nn.CrossEntropyLoss()

    epoch = args.epochs
    for e in range(20):
        print('\nEpoch #{}:'.format(e+1))
        model.train(mode=True)
        tcorrect = 0
        tlen = 0
        tloss = 0
        for x,y in trainloader:
            xdata = x.cuda()
            ydata = y.cuda()
            optimizer.zero_grad()
            output = model(xdata)
            loss = criterion(output, ydata)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            pred = output.argmax(axis=1)
            seqcorrect = (pred==ydata).prod(-1)
            tcorrect = tcorrect + seqcorrect.sum().item()
            tlen = tlen + seqcorrect.nelement()
            tloss = tloss + loss.item()
            #scheduler.step()
        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        print('Current LR:' + str(scheduler.get_lr()[0]))
        

        model.train(mode=False)
        vcorrect = 0
        vlen = 0
        for x,y in valloader:
            xdata = x.cuda()
            ydata2 = y.cuda()
            output = model(xdata)
            loss = criterion(output, ydata2)
            pred2 = output.argmax(axis=1)
            seqcorrect = (pred2==ydata2).prod(-1)
            vcorrect = vcorrect + seqcorrect.sum().item()
            vlen = vlen + seqcorrect.nelement()
        print("val accuracy = "+str(vcorrect/vlen))
        if tcorrect/tlen > 0.3:
            break

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
    for e in range(epoch):
        print('\nEpoch #{}:'.format(e+1))
        model.train(mode=True)
        tcorrect = 0
        tlen = 0
        tloss = 0
        for x,y in trainloader:
            xdata = x.cuda()
            ydata = y.cuda()
            optimizer.zero_grad()
            output = model(xdata)
            loss = criterion(output, ydata)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            pred = output.argmax(axis=1)
            seqcorrect = (pred==ydata).prod(-1)
            tcorrect = tcorrect + seqcorrect.sum().item()
            tlen = tlen + seqcorrect.nelement()
            tloss = tloss + loss.item()
        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        print('Current LR:' + str(scheduler.get_lr()[0]))

        model.train(mode=False)
        vcorrect = 0
        vlen = 0
        for x,y in valloader:
            xdata = x.cuda()
            ydata2 = y.cuda()
            output = model(xdata)
            loss = criterion(output, ydata2)
            pred2 = output.argmax(axis=1)
            seqcorrect = (pred2==ydata2).prod(-1)
            vcorrect = vcorrect + seqcorrect.sum().item()
            vlen = vlen + seqcorrect.nelement()
        print("val accuracy = "+str(vcorrect/vlen))
        #scheduler.step()
