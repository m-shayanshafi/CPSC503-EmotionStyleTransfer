# import torch
# import torch.functional as F
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import re
# import numpy as np
# import time
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import util
# import ConstructVocab as construct
# from torch.utils.data import Dataset, DataLoader
# import confusion

import sys
sys.path.append("../../utils")

import constructvocab as construct
import util 
import pandas as pd

# load data from pickle
data = util.load_from_pickle(directory="../../data/emotion/train/merged_training.pkl")

# data.emotions.value_counts().plot.bar()
# counts.show()

print(data.head(10))


# Preprocessing data
# retain only text that contain less that 70 tokens to avoid too much padding
# data = data.sample(n=50000);
data["token_size"] = data["text"].apply(lambda x: len(x.split(' ')))
data = data.loc[data['token_size'] < 70].copy()
inputs = construct.ConstructVocab(data["text"].values.tolist())
inputs.writeFile("../../models/classifier/emotion_classifier/emotion_classifier_vocab.src.dic")

print(inputs.vocab[0:10])
