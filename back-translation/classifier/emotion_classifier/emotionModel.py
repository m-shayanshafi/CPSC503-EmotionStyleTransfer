import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

criterion = nn.CrossEntropyLoss()

class EmoGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size):
        super(EmoGRU, self).__init__()
        self.batch_sz = batch_sz
        self.hidden_units = hidden_units
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.output_size = output_size
        
        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_units)
        self.fc = nn.Linear(self.hidden_units, self.output_size)
    
    def initialize_hidden_state(self,batch_size):
        return Variable(torch.zeros((1, batch_size, self.hidden_units)))

    def forward(self, x):
        # if len(x.size()) > 2:
        #     x = torch.max(x,2)
        
        # print(x[0])
        # print(x[1])

        if len(x.size()) > 2:
            emb = torch.mm(x.view(-1, x.size(2)), self.embedding.weight)
            # print(emb.size())
            emb = emb.view(-1, x.size(1), self.embedding_dim)
            # print(emb.size())
            x = emb
            self.hidden = self.initialize_hidden_state(x.size(1)).cuda()
        else:
            x = self.embedding(x)    
            self.hidden = self.initialize_hidden_state(x.size(1))
        # print(x.size())        
        
        output, self.hidden = self.gru(x, self.hidden) # max_len X batch_size X hidden_units
        out = output[-1, :, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return out

def loss_function(y, prediction):
    """ CrossEntropyLoss expects outputs and class indices as target """
    # convert from one-hot encoding to class indices
    target = torch.max(y, 1)[1]
    loss = criterion(prediction, target) 
    return loss   #TODO: refer the parameter of these functions as the same
    
def accuracy(target, logit):
    ''' Obtain accuracy for training round '''
    target = torch.max(target.data, 1)[1] # convert from one-hot encoding to class indices
    corrects = (torch.max(logit, 1)[1].data == target).sum()
    accuracy = 100.0 * corrects / len(logit)
    return accuracy

def accuracy2(target, logit):
    ''' Obtain accuracy for training round '''
    target = torch.max(target, 1)[1] # convert from one-hot encoding to class indices
    corrects = (torch.max(logit, 1)[1].data == target).sum()

    return corrects
    # accuracy = 100.0 * corrects / len(logit)
    # return accuracy