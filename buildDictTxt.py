from collections import defaultdict

class BuildWordDict():
    def __init__(self, addr, dicName, threshold, makeFile=False):
        self.freqDict = defaultdict(int)
        self.wordDict = defaultdict(int)

        self.loadDataset(addr)
        self.wordList = sorted(self.wordDict.items(), key=lambda x: x[1], reverse=True)
        self.finalDict = self.writeDict(dicName, threshold, makeFile=False)
        
    def loadDataset(self, addr):
        with open(str(addr), "r") as fd:
            for sen in fd:
                for word in sen.strip(" \n").split(" "):
                    if word == '':   continue
                    self.wordDict[word] += 1
            fd.close()
    
    def writeDict(self, dicName, threshold, makeFile = False):
        finalDict = defaultdict(int) #Threshold applied
        if makeFile == True: 
            with open("wordDict_" + str(dicName) + ".txt", "w") as fd:
                
                fd.write("<pad>\n")
                fd.write("<bos>\n")
                fd.write("<eos>\n")
                fd.write("<mask>\n")
                fd.write("<etc>\n")
                # <UNK> Will be automatically included when loading txt
                
                for k, v in self.wordList:
                    if v > int(threshold):
                        fd.write(k)
                        fd.write("\n")
                fd.close()
        elif makeFile == False:
            for k, v in self.wordList:
                    if v > int(threshold):
                        finalDict[k] += int(v)
            return finalDict

if __name__ == '__main__':
    test = BuildWordDict("./data/penn/train.txt", "penn", 5, False)
    print(test.finalDict)