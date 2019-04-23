import pickle
import numpy as np
from torch.utils.data import Dataset
import numpy as np



class Data(Dataset):
	def __init__(self, X, y):
		self.data = X
		self.target = y
		self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
		
	def __getitem__(self, index):
		x = self.data[index]
		y = self.target[index]
		x_len = self.length[index]
		return x, y, x_len
	
	def __len__(self):
		return len(self.data)

def convert_to_pickle(item, directory):
	pickle.dump(item, open(directory,"wb"))

def load_from_pickle(directory):
	return pickle.load(open(directory,"rb"))

def max_length(tensor):
	return max(len(t) for t in tensor)

def pad_sequences(x, max_len):
	padded = np.zeros((max_len), dtype=np.int64)
	
	if len(x) > max_len: 
	
		padded[:] = x[:max_len]
	
	else:

		padded[:len(x)] = x
		return padded

### sort batch function to be able to use with pad_packed_sequence
def sort_batch(X, y, lengths):
	lengths, indx = lengths.sort(dim=0, descending=True)
	X = X[indx]
	y = y[indx]
	return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)

