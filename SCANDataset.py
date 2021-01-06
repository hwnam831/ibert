import io  
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch
import random
from buildDictTxt import BuildWordDict
from datasets import load_dataset


"""
    SCAN Dataset
        Input:
            splitType: Please choose between "train", "test"
            dataType: Please choose between 'length' and 'simple'
        
        Output:
            target: actions 
                    e.g.) ['I_RUN', 'I_RUN', 'I_RUN']
"""
class SCANDataset(Dataset):
    
    def __init__(self, maxSeq=64, dataType='length', splitType='train'):      
        
        self.dataset = load_dataset("scan", dataType, split=splitType) 
        self.wordtoix, self.ixtoword = defaultdict(str), defaultdict(int)
        self.textual_ids = self.build_dictionary()
        self.vocab_size = len(self.wordtoix)

        print("Sample Data Loaded")
        print(self.dataset[124])

        print("Coverting to textual_ids")
        print("0: <pad>, 1: <bos>, 2: <eos>")
        print(self.textual_ids[124])
        
        print("Pad to make every data share the same length")
        self.padded_ids = [self.pad_sequences(x, maxSeq) for x in self.textual_ids]   
        print(self.padded_ids[124])

    def pad_sequences(self, x, max_len):
        
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: 
            padded[:] = x[:max_len]
        else: 
            padded[:len(x)] = x
        return padded

    def build_dictionary(self):

        with io.open('./compDict.txt', encoding='UTF-8') as f:
            for i, line in enumerate(f):    
                self.wordtoix[line.rstrip('\n')] = i
                self.ixtoword[i] = line.rstrip('\n')

        textual_ids = list()
        for i in range(0, len(self.dataset)):  
            temp = list()
            temp.append(self.wordtoix.get('<bos>'))
            for word in self.dataset[i]['commands'].split(" "):
                temp.append(self.wordtoix[word]) 
            
            temp.append(self.wordtoix.get('<eos>'))
            textual_ids.append(temp)
        return textual_ids 
  
    def __getitem__(self, index):

        action_dict = {'I_TURN_RIGHT':0, 'I_JUMP':1, 'I_WALK':2, 'I_TURN_LEFT':3, 'I_RUN':4, 'I_LOOK':5}
        target = list()
        for word in self.dataset[index]['actions'].split(" "):
            target.append(action_dict[word])
        return target

    def __len__(self):
        
        return len(self.textual_ids)
    

if __name__ == '__main__':
    dataset = SCANDataset(splitType='train') # Use among 'train', 'test'
    dataset.__getitem__(124) 
    loader = DataLoader(dataset, batch_size=4)
