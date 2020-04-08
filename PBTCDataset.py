import io  
import numpy as np
from torchtext import data
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

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
        
        print("Sample Data Loaded")
        print(self.data[11])

        print("Coverted into textual_ids")
        print("0: <pad>, 1: <bos>, 2: <eos>, 3: <unk>")
        print(self.textual_ids[11])

    def build_dictionary(self, ixtoword, wordtoix):
        """ Add to dictionary """ 
        freqDict = defaultdict(int)
        wordDict = {'<pad>':0, '<bos>':1, '<eos>':2, '<unk>':3}   

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
                textual_ids.append(temp)
            temp.append(self.wordtoix.get('<eos>'))
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
        x = self.vectorized_textual_ids[index]
        target = self.vectorized_textual_ids[index + 1]
        return x, target

    def __len__(self):
        print("Total len of dataset: ", len(self.vectorized_textual_ids))
        return len(self.vectorized_textual_ids)

if __name__ == '__main__':

    dataset = PBTCDataset('train') # Use among 'train', 'valid', 'test'
    dataset.__len__()
    dataset.__getitem__(11)

    loader = DataLoader(dataset, batch_size=4)