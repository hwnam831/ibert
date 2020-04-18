import io  
import numpy as np
from torchtext import data
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch

"""
    PBTCDataset
        Input:
            dataType: Please choose among "train", "valid", "test"
            ixToword: If there exists text <-> textual id dictionary, put it here
            wordToix: If there exists text <-> textual id dictionary, put it here
            THRESHOLD: Default is 5. This only adds characters to dictionary only when it appears more times than the threshhold.
        
        Output: 
            None
        
        Description: 
            This dataset uses following preprocessing strategy
                <pad> : To make all tensors have equal size (Not used but just in case. April 8, 2020)
                <bos> : Beginning of the sentence
                <eos> : End of the sentence
                <unk> : For unknown characters 
                _     : Whitespace
"""
class PBTCDataset(Dataset):

    urls = ['data/pennchar/train.txt',
        'data/pennchar/valid.txt',
        'data/pennchar/test.txt']

    def __init__(self, dataType, ixtoword=None, wordtoix=None, THRESHOLD=5):
        self.data = self.loadData(dataType)        
        self.threshhold = THRESHOLD
        self.wordtoix, self.ixtoword = ixtoword, wordtoix
        self.textual_ids, self.vectorized_textual_ids = self.build_dictionary(ixtoword, wordtoix)
        self.maxLen = None
        self.padded_ids = None

        print("Sample Data Loaded")
        print(self.data[124])

        print("Coverted into textual_ids")
        print("0: <pad>, 1: <bos>, 2: <eos>, 3: <unk>, 4: _, 99: mask")
        print(self.textual_ids[124])

        print("Start Padding to make every data have same length")
        self.maxLen = self.getMaxLength(self.textual_ids) 
        self.padded_ids = [self.pad_sequences(x, self.maxLen) for x in self.textual_ids]   
        print(self.padded_ids[124])

    def getMaxLength(self, list):
        result = max(len(t) for t in list)
        result = self.adjustMaxSize(result)
        
        return result
    
    ''' Torch size follows: (3n + 4) * 16 where n equals digits from system argument '''
    def adjustMaxSize(self, a):
        if a % 16 == 0 and (a /16 - 4)%3 == 0:
            pass
        else:
            while (a % 16 != 0 or (a /16 - 4)%3 != 0):
                a +=1
        return a

    def pad_sequences(self, x, max_len):
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def build_dictionary(self, ixtoword, wordtoix):
        """ Add to dictionary """ 
        freqDict = defaultdict(int)
        wordDict = {'<pad>':0, '<bos>':1, '<eos>':2, '<unk>':3, '_':4}   

        index = len(wordDict)
        for sen in self.data:
            for c in sen:
                freqDict[c] += 1
                wordDict[c] = index
                index += 1

        """ Build text <-> textual id Dictionary """ 
        if ixtoword == None or wordtoix == None:
            self.wordtoix = {word: i for i, word in enumerate(wordDict)}
            self.ixtoword = {i: word for i, word in enumerate(wordDict)}
        else:
            self.ixtoword = ixtoword
            self.wordtoix = wordtoix

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
        vectorized_textual_ids = list()        
        for i in range(0, len(self.data)):  
            temp = list()
            temp.append(self.wordtoix.get('<bos>'))
            vectorized_textual_ids.append(self.wordtoix.get('<bos>'))
            for word in self.data[i]:
                if word in self.wordtoix:
                    temp.append(self.wordtoix.get(word)) 
                    vectorized_textual_ids.append(self.wordtoix.get(word)) 
                else:   
                    temp.append(self.wordtoix.get('<unk>'))
                    vectorized_textual_ids.append(self.wordtoix.get('<unk>'))
            temp.append(self.wordtoix.get('<eos>'))
            textual_ids.append(temp)
            vectorized_textual_ids.append(self.wordtoix.get('<eos>'))
        return textual_ids, vectorized_textual_ids

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
                corpus.append(line.rstrip().split(' '))
        return corpus    
  
    def __getitem__(self, index):
        x = self.padded_ids[index]
        x = np.asarray(x, dtype=np.float32)
        
        x = x.reshape(len(x), 1)
        x = x.reshape(int(x.shape[0]/16), 16)
        
        masked, target = self.splitWithMask(x)
        target = self.pad_sequences(target, masked.shape[0])
        target = np.asarray(target, dtype=np.long)

        # print("input shape: ", masked.shape)
        # print("target shape: ", target.shape)
        
        return masked, target

    def __len__(self):
        return len(self.textual_ids)
    
    ############################################
    # Toy problem. All whitespace is now masked. 
    # More rules can be added here 
    ############################################
    def splitWithMask(self, arr):
        answer = list()
        for i, row in enumerate(arr):
            for j, col in enumerate(row):
                if col == 4:
                    answer.append(col)
                    arr[i][j] = 99.  # 99 is mask
        return arr, answer

if __name__ == '__main__':
    dataset = PBTCDataset('train') # Use among 'train', 'valid', 'test'
    dataset.__getitem__(124)
    loader = DataLoader(dataset, batch_size=4)
