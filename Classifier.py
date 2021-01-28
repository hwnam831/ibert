import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import Options
import Models
import IBERT
from torch.utils.data import Dataset, DataLoader
import time

class ListopsDataset(Dataset):
    def __init__(self, tsv_file, max_len=-1):
        
        vocabs = ['0','1','2','3','4','5','6','7','8','9',
                    '(', ')', '[MIN', '[MAX', '[MED', '[FIRST', '[LAST', '[SM', ']']
        self.dict = {}
        for i,v in enumerate(vocabs):
            self.dict[v] = i
        self.vocab_size = len(vocabs)+2 #including pad
        self.inputs = []
        self.targets = []
        with open(tsv_file, "r") as fd:
            for l in fd:
                inp, tgt = l.split('\t')
                tokens = inp.split(' ')
                if len(tokens) > max_len:
                    max_len = len(tokens)
                seq = [self.dict[tok] for tok in tokens]
                self.inputs.append(seq)
                self.targets.append(int(tgt))
        self.max_len = max_len
        self.inp_arr = np.ones([len(self.targets), self.max_len+1], dtype=np.int64)*(self.vocab_size-2) #pre-fill with padding
        self.inp_arr[:,0] = self.vocab_size-1
        for idx, inp in enumerate(self.inputs):
            for i,t in enumerate(inp):
                self.inp_arr[idx, i+1] = t
        self.size = len(self.targets)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inp_arr[idx], self.targets[idx]
'''
dataset = ListopsDataset('output_dir/basic_train.tsv')
vocabs = ['0','1','2','3','4','5','6','7','8','9',
                    '(', ')', '[MIN', '[MAX', '[MED', '[FIRST', '[LAST', '[SM', ']', '_', 'CLS']
for i in range(10):
    idx = np.random.randint(0,len(dataset))
    x,y = dataset.__getitem__(i)
    xseq = [vocabs[t] for t in x]
    print('{}\n{}'.format(' '.join(xseq), y))
'''

if __name__ == '__main__':
    
    args = Options.get_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = False
    if args.tf32:
        print("TF32 computation enabled")
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        torch.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True
    dataset = ListopsDataset('output_dir/basic_train.tsv')
    valset = ListopsDataset('output_dir/basic_args.tsv')

    vocab_size = dataset.vocab_size

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
        model = Models.LSTMAE(dmodel, num_layers=num_layers, vocab_size = vocab_size, bidirectional=args.bidirectional).cuda()
    elif args.net == 'nam':
        print('Executing NAM Autoencoder model')
        model = AMEncoder(dmodel, nhead=nhead, num_layers=num_layers, vocab_size=vocab_size, attn=RecurrentAM).cuda()
    elif args.net == 'linear':
        print('Executing Linear Attention Autoencoder model')
        model = AMEncoder(dmodel, nhead=nhead, num_layers=num_layers, vocab_size=vocab_size, attn=LinearAttention).cuda()
    elif args.net == 'dnc':
        print('Executing DNC model')
        model = Models.DNCAE(128, 4, num_layers=1, vocab_size=vocab_size).cuda()
    elif args.net == 'ut':
        print('Executing Universal Transformer model')
        model = Models.UTAE(dmodel, nhead, num_layers=num_layers, vocab_size=vocab_size).cuda()
    else :
        print('Network {} not supported'.format(args.net))
        exit()
    print(model)

    

    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader   = DataLoader(valset, batch_size=args.batch_size*2, num_workers=2)
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
    criterion   = nn.CrossEntropyLoss(reduction='none')
    nsamples = len(dataset)
    #torch.autograd.set_detect_anomaly(True)

    for e in range(args.epochs):
        print('\nEpoch #{}:'.format(e+1))
        
        trainstart = time.time()
        #train the model
        model.train(mode=True)
        tcorrect = 0
        tlen     = 0
        tloss    = 0
        for x,y in trainloader:
            xdata       = x.cuda()
            ydata       = y.cuda()
            optimizer.zero_grad()
            output      = model(xdata)
            if args.net == 'dnc':
                output = output[:,:,-1]
            else:
                output = output[:,:,0] #First of N,C,S
            

            loss        = criterion(output, ydata)
            loss.mean().backward()
            tloss       = tloss + loss.mean().item()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            pred        = output.argmax(axis=1)
            seqcorrect  = (pred==ydata)
            tcorrect    = tcorrect + seqcorrect.sum().item()
            tlen        = tlen + len(seqcorrect)
        scheduler.step()

        trainingResult = list()
        print('train seq acc:\t'+str(tcorrect/tlen))
        print('train loss:\t{}'.format(tloss/len(trainloader)))
        print('Current LR:' + str(scheduler.get_last_lr()[0]))
        print("Train sequences per second : " + str(nsamples/(time.time()-trainstart)))

        #validate the model
        model.eval()
        vcorrect = 0
        vlen     = 0
        vloss    = 0
        with torch.no_grad():
            for x,y in valloader:
                xdata       = x.cuda()
                ydata       = y.cuda()
                output      = model(xdata)[:,:,0] #First of N,C,S

                loss        = criterion(output, ydata)
                vloss       = vloss + loss.mean().item()
                
                pred        = output.argmax(axis=1)
                seqcorrect  = (pred==ydata)
                vcorrect    = vcorrect + seqcorrect.sum().item()
                vlen        = vlen + len(seqcorrect)

        print('\nValidation seq acc:\t'+str(vcorrect/vlen))
        print('Validation loss:\t{}'.format(vloss/len(valloader)))
    