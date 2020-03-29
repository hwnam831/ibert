import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

'''
# 16-dim one-hot vectors
# 0-9:  decimal digits
# 10:   Delimiter (space)
# 11:   Pad
# 12:   Start
# 13:   EOS
# 14:   Mask
# 15:   Custom token    
'''

class Token():
    delim = 10 
    pad = 11
    start = 12
    eos = 13
    mask = 14
    custom = 15

def fib(seed1, seed2, numbers):
    seq = [seed1, seed2]
    for i in range(2, numbers):
        seq.append(seq[i-2]+seq[i-1])
    target = seq[-2]+seq[-1]
    return seq, target

def arith(seed1, seed2, numbers):
    seq = [seed1, seed2] if seed1<seed2 else [seed2,seed1]
    for i in range(2, numbers):
        seq.append(2*seq[i-1]-seq[i-2])
    target = 2*seq[-1]-seq[-2]
    return seq, target

def count(seed1, seed2, numbers):
    seq = [seed1]
    diff = seed2%10+10
    for i in range(1, numbers):
        seq.append(seq[i-1]+diff)
    target = seq[-1]+diff
    return seq, target

def palindrome(seed1, seed2, numbers):
    seq = [seed1]
    for i in range(1, numbers):
        nstr = str(seq[i-1])
        seq[i] = int(nstr[::-1])
    target = int(str(seq[-1])[::-1])
    return seq, target

def num2vec(num, ndigits, lendian=True):
    digits = [int(c) for c in str(num)]
    if len(digits)<ndigits:
        digits = [0 for i in range(ndigits-len(digits))]+digits
    elif len(digits)>ndigits:
        digits = digits[len(digits)-ndigits:]
    if lendian:
        digits.reverse()
    return np.array(digits)

#Dataset class for autoencoding setup (CNN, BERT, etc)
class NSPDatasetAE(Dataset):
    def __init__(self, rule, maxdigits, mindigits=1, numbers=2, size=25600, lendian=False):
        self.rule = rule
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        self.lendian = lendian
        self.numbers = numbers
        self.maxlen = (maxdigits+1)*(numbers+1) + 1
        self.inputs = np.zeros([size, self.maxlen, 16], dtype=np.float32)
        self.targets = np.ones([size, self.maxlen], dtype=np.int64)*Token.pad
        self.iscreated = [False for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.iscreated[idx]:
            ndigits = np.random.randint(self.mindigits, self.maxdigits+1)
            seed1 = np.random.randint(10**(ndigits-1), 10**ndigits)
            seed2 = np.random.randint(10**(ndigits-1), 10**ndigits)
            seq, target = self.rule(seed1, seed2, self.numbers)
            pos = 1
            self.inputs[idx][0][Token.delim] = 1
            self.targets[idx][0] = Token.delim
            
            for i in range(self.numbers):
                vec = num2vec(seq[i], ndigits, self.lendian)
                for j,v in enumerate(vec):
                    self.inputs[idx][pos+j][v] = 1
                    self.targets[idx][pos+j] = v
                self.inputs[idx][pos+ndigits][Token.delim] = 1
                self.targets[idx][pos+ndigits] = Token.delim
                pos = pos + ndigits + 1

            y = num2vec(target, ndigits, self.lendian)

            self.inputs[idx][pos:pos+len(y),Token.mask] = 1
            self.targets[idx][pos:pos+len(y)] = y
            self.inputs[idx][pos+len(y)][Token.eos] = 1
            self.targets[idx][pos+len(y)] = Token.eos

            if pos+len(y)+1 < self.maxlen:
                self.inputs[idx][pos+len(y)+1:,Token.pad] = 1
            
            self.iscreated[idx] = True
        return self.inputs[idx], self.targets[idx]

#Seq2Seq version + no one-hot encoding
class NSPDatasetS2S(Dataset):
    def __init__(self, rule, maxdigits, mindigits=1, numbers=2, size=25600, lendian=False):
        self.rule = rule
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        self.lendian = lendian
        self.numbers = numbers
        self.maxlen = (maxdigits+1)*numbers + 1
        self.inputs = np.ones([size, self.maxlen], dtype=np.int64)*Token.pad
        self.targets = np.ones([size, self.maxdigits], dtype=np.int64)*Token.pad
        self.iscreated = [False for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.iscreated[idx]:
            ndigits = ((self.maxdigits-self.mindigits+1)*idx)//self.size + self.mindigits
            seed1 = np.random.randint(10**(ndigits-1), 10**ndigits)
            seed2 = np.random.randint(10**(ndigits-1), 10**ndigits)
            seq, target = self.rule(seed1, seed2, self.numbers)
            pos = 1
            self.inputs[idx][0] = Token.delim
            
            for i in range(self.numbers):
                vec = num2vec(seq[i], ndigits, self.lendian)
                self.inputs[idx][pos:pos+ndigits] = vec
                self.inputs[idx][pos+ndigits] = Token.delim
                pos = pos + ndigits + 1

            self.targets[idx][:ndigits] = num2vec(target, ndigits, self.lendian)            
            self.iscreated[idx] = True
        return self.inputs[idx], self.targets[idx]
def printseq(x,y):
    tokenmap = ['0','1','2','3','4','5','6','7','8','9','_',' ','S','E','M','C']
    print("input:")
    xseq = np.argmax(x,-1)
    print('\t' + ' '.join([tokenmap[n] for n in xseq]))
    print("target:")
    print('\t' + ' '.join([tokenmap[n] for n in y]))

def printseq2(x,y):
    tokenmap = ['0','1','2','3','4','5','6','7','8','9','_',' ','S','E','M','C']
    print("input:")
    print('\t' + ' '.join([tokenmap[n] for n in x]))
    print("target:")
    print('\t' + ' '.join([tokenmap[n] for n in y]))
if __name__ == '__main__':
    dataset = NSPDatasetS2S(fib,5)
    loader = DataLoader(dataset, batch_size=4)
    for i in range(10):
        x,y = dataset.__getitem__(i)
        printseq2(x,y)
        #print(np.argmax(x,-1))
        #print(y)