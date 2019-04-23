# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for the dataset
import codecs

class ConstructVocab():

    def __init__(self, sentences, empty=False):
        self.sentences = sentences
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        if not empty:
            self.create_index()
        
    def create_index(self):
        for s in self.sentences:
            # update with individual tokens
            self.vocab.update(s.split(' '))
            
        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.word2idx['<blank>'] = 0 # PAD
        self.word2idx['<unk>'] = 1 # UNK
        self.word2idx['<s>'] = 2 # BOS
        self.word2idx['</s>'] = 3 # EOS
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 4 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word 

        # Write entries to a file.
    def writeFile(self, filename):
        with codecs.open(filename, 'w', "utf-8") as file:
            for i in range(self.size()):
                label = self.idx2word[i]
                file.write('%s %d\n' % (label, i))

        file.close() 

    # Load entries from a file.
    def loadFile(self, filename):
    
        for line in open(filename):
            fields = line.split()
            if len(fields) > 2:
                idx = int(fields[-1])
                label = ' '.join(fields[:-1])
            else:
                label = fields[0]
                idx = int(fields[1])
            self.add(label, idx)


    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        
        # label = label.lower() if self.lower else label
        if idx is not None:
            self.idx2word[idx] = label
            self.word2idx[label] = idx
        else:
            if label in self.word2idx:
                idx = self.word2idx[label]
            else:
                idx = len(self.idx2word)
                self.idx2word[idx] = label
                self.word2idx[label] = idx
        
        return idx


    def size(self):
        return len(self.idx2word)