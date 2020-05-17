import io  
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch
import random
from buildDictTxt import BuildWordDict

"""
    PennTreeBankWord Dataset
        Input:
            dataType: Please choose among "train", "valid", "test"
            ixToword: If there exists text <-> textual id dictionary, put it here
            wordToix: If there exists text <-> textual id dictionary, put it here
            THRESHOLD: Default is 5. This only adds characters to dictionary only when it appears more times than the threshhold.
        
        Output: 
            None
        
        Description: 
            This dataset uses following preprocessing strategy
                <pad> : To make all tensors have equal size 
                <bos> : Beginning of the sentence
                <eos> : End of the sentence
                <unk> : For unknown words 
                <etc> : etc. This is different from unk
                <mask> : Mask
"""
class PTBWDataset(Dataset):

    urls = ['data/penn/train.txt',
        'data/penn/valid.txt',
        'data/penn/test.txt']

    def __init__(self, dataType, ixtoword=None, wordtoix=None, minSeq = 4, maxSeq = 64, THRESHOLD=5):
        self.minSeq = minSeq
        self.maxLen = maxSeq
        self.data = self.loadData(dataType)        

        self.threshhold = THRESHOLD
        self.wordtoix, self.ixtoword = wordtoix, ixtoword
        self.textual_ids = self.build_dictionary()
        self.padded_ids = None
        self.vocab_size = len(self.wordtoix)
        self.maskedIndex = list()

        print("Sample Data Loaded")
        print(self.data[124])

        # print("Coverted into textual_ids")
        print("0: <pad>, 1: <bos>, 2: <eos>, 3: <unk>, 4: <etc> 5: <mask>")
        # print(self.textual_ids[124])
        
        print("Start Padding to make every data have same length")
        self.padded_ids = [self.pad_sequences(x, self.maxLen) for x in self.textual_ids]   
        print(self.padded_ids[124])

    def getMaxLength(self, list):
        result = max(len(t) for t in list)
        result = self.adjustMaxSize(result)
        
        return result
    
    def pad_sequences(self, x, max_len):
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def build_dictionary(self):
        """ Add to dictionary """ 
        freqDict = defaultdict(int)
        wordDict = defaultdict(str)
        ixtoword = defaultdict(int)
        temp = BuildWordDict("./data/penn/train.txt", "penn", 5, False)

        with io.open('wordDict.txt', encoding='UTF-8') as f:
            for i, line in enumerate(f):    
                for c in line.split(' '):
                    wordDict[c.rstrip('\n')] = i
                ixtoword[i] = line.split(' ')[0].rstrip('\n')

        """ Build text <-> textual id Dictionary """ 
        if self.ixtoword == None or self.wordtoix == None:
            self.wordtoix = wordDict
            self.ixtoword = {i: word for i, word in enumerate(wordDict)}

        """ 
            Convert full text into textual ids 
            Addes <bos> and <eos> at the beginning and the end of the sentence respectively 

            Description: 
                textual_ids: 
                            [  ['<bos>', 'a', 'b', 'c', '<eos>'],
                               ['<bos>', 'a', 'b', 'c', '<eos>'] ]
                vectorized_ids: 
                            [  '<bos>', 'a', 'b', 'c', '<eos>', '<bos>', 'a', 'b', 'c', <eos> ]
        """
        textual_ids = list()
        for i in range(0, len(self.data)):  
            temp = list()
            temp.append(self.wordtoix.get('<bos>'))
            for word in self.data[i]:
                if word in self.wordtoix:
                    temp.append(self.wordtoix.get(word)) 
                else:   
                    temp.append(self.wordtoix.get('<etc>'))
            temp.append(self.wordtoix.get('<eos>'))
            textual_ids.append(temp)
        return textual_ids 

    def loadData(self, dataType):
        """ Load path of text file """         
        if dataType == "train":
            f = self.urls[0]
        elif dataType == "valid":
            f = self.urls[1]
        elif dataType == "test":
            f = self.urls[2]   

        """ Load text file """
        corpus = list()
        with io.open(f, encoding='UTF-8') as f:
            for line in f:    
                words = line.split(" ")
                if len(words) > self.minSeq and len(words) < self.maxLen:
                    corpus.append(line.lstrip(" ").rstrip().split(' '))
        return corpus    
  
    def onehot_encoder(self, idxs):
        vec = np.zeros([self.maxLen, self.vocab_size], dtype=np.float32)
        for i, id in enumerate(idxs):
            vec[i, id] = 1
        return vec

    def __getitem__(self, index):
        x = self.padded_ids[index]

        masked, target = self.splitWithMask(x)
        target = self.pad_sequences(target, masked.shape[0])

        masked = self.onehot_encoder(masked)

        return masked, target

    def __len__(self):
        return len(self.textual_ids)
    
    ############################################
    # Mask Random Words
    ############################################
    def splitWithMask(self, idxs):
        whiteSpaceList = list()
        eosIdx = None
        masked = np.asarray([i for i in idxs])
        
        for i, v in enumerate(idxs):
            if v == self.wordtoix.get('<eos>'):
                eosIdx = i       

        # If there are more than two words
        # ex: [4, 10, 19, 24, 27, eosIdx]
        if eosIdx > 2:
            randomIdx = random.randint(1, eosIdx-1)
            masked[randomIdx] = self.wordtoix.get('<mask>')
            self.maskedIndex.append(randomIdx)
            return masked, idxs
        
        # If there is one word, return original text. eg) Hello 
        else:
            return idxs, idxs
    

if __name__ == '__main__':
    dataset = PTBWDataset('train') # Use among 'train', 'valid', 'test'
    dataset.__getitem__(124) 
    print(dataset.wordtoix)
    loader = DataLoader(dataset, batch_size=4)