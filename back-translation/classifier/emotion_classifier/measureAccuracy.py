import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import re
import numpy as np
import time
from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import util
# import ConstructVocab as construct
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
# import confusion
import sys
import os
sys.path.append("../../utils")
import util
import constructvocab as construct
import pandas as pd
import emotionModel as emoModel

# trainPath = "../../data/emotion/train/"
testPath = "../../data/emotion/style-accuracy2/"
# emotions = ["anger", "fear", "joy", "love", "sadness", "surprise"]
emotions = ["fear"]
emotion_dict = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}
emotion_dict = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}
vocabPath = "../../models/classifier/emotion_classifier/emotion_classifier_vocab.src.dic"
globalVocab = construct.ConstructVocab([], empty=True)
classifier_model= "../../models/classifier/emotion_classifier/emotion_classifier_2.pt"

def loadData(trainOrTest):

	loadPath = ""
	if trainOrTest=="train":
		loadPath = trainPath
	else:
		loadPath = testPath

	emotionDF =[]	
	
	for emotion in emotions:

		emotionFilePath = loadPath + emotion
		print(emotionFilePath) 
		if not os.path.exists(emotionFilePath):
			continue

		data = pd.read_csv(emotionFilePath, sep='\t', names=["text"])
		data.insert(1, "emotion", emotion)
		emotionDF.append(data)

	trainDF = pd.concat(emotionDF, ignore_index=True)
	trainDF = trainDF.sample(frac=1)

	input_tensor, target_tensor = pandasToTensor(trainDF)	
	return input_tensor, target_tensor

def pandasToTensor(data):

	data["token_size"] = data["text"].apply(lambda x: len(x.split(' ')))
	data = data.loc[data['token_size'] < 70].copy() 

	# load globalVocab word2idx

	if os.path.exists(vocabPath):
		globalVocab.loadFile(vocabPath)
	else:
		print("Vocabulary doesn't exist")

	input_tensor = [[globalVocab.word2idx[s] for s in es.split(' ')]  for es in data["text"].values.tolist()]

	# examples of what is in the input tensors
	# print(input_tensor[0:2])

	# calculate the max_length of input tensor
	max_length_inp = util.max_length(input_tensor)
	# print(max_length_inp)

	# inplace padding
	input_tensor = [util.pad_sequences(x, max_length_inp) for x in input_tensor]
	# print(input_tensor[0:2])

		###Binarization
	emotions = list(emotion_dict.values())
	num_emotions = len(emotion_dict)
	# print(emotions)
	# binarizer
	mlb = preprocessing.MultiLabelBinarizer(classes=emotions)
	data_labels =  [emos for emos in data[['emotion']].values]
	# print(data_labels)
	bin_emotions = mlb.fit_transform(data_labels)
	target_tensor = np.array(bin_emotions.tolist())

	# print(target_tensor[0:2])
	# print(data[0:2]) 

	get_emotion = lambda t: np.argmax(t)

	get_emotion(target_tensor[0])   
	emotion_dict[get_emotion(target_tensor[0])]

	return input_tensor, target_tensor  




# # Load dataset
# input_tensor_train, target_tensor_train = loadData('train')
input_tensor_test, target_tensor_test = loadData('test')

# print(len(input_tensor_train))
print(len(input_tensor_test))

# TRAIN_BUFFER_SIZE = len(input_tensor_train)
TEST_BUFFER_SIZE = len(input_tensor_test)
BATCH_SIZE = 64
# TRAIN_N_BATCH = TRAIN_BUFFER_SIZE // BATCH_SIZE
TEST_N_BATCH = TEST_BUFFER_SIZE // BATCH_SIZE

embedding_dim = 256
units = 1024
vocab_inp_size = len(globalVocab.word2idx)
target_size = len(emotion_dict)

torch.cuda.set_device(0)

# train_dataset = util.Data(input_tensor_train, target_tensor_train)
test_dataset = util.Data(input_tensor_test, target_tensor_test)

# train_dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
# 					 drop_last=True,
# 					 shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size = BATCH_SIZE, 
					 drop_last=False,
					 shuffle=False)

# print(val_dataset.batch_size)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.cuda.set_device(0)

class_check = torch.load(classifier_model, map_location=lambda storage, loc: storage)
class_opt = class_check['opt']
class_dict = class_check['vocabulary']
model = emoModel.EmoGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
model = emoModel.EmoGRU(class_opt["vocab_inp_size"], class_opt["embedding_dim"], class_opt["units"], BATCH_SIZE, class_opt["target_size"]) 
model.load_state_dict(class_check['model'])  
model.eval()






# model.to(device)

# # obtain one sample from the data iterator
# it = iter(train_dataset)
# x, y, x_len = next(it)

# # sort the batch first to be able to use with pac_pack sequence
# xs, ys, lens = util.sort_batch(x, y, x_len)

# print("Input size: ", xs.size())

# output, _ = model(xs)
# print(output.size())

# model = emoModel.EmoGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
# # # model.to(device)

# ### loss criterion and optimizer for training
# criterion = nn.CrossEntropyLoss() # the same as log_softmax + NLLLoss
# optimizer = torch.optim.Adam(model.parameters())

# EPOCHS = 1

# for epoch in range(EPOCHS):
# 	start = time.time()
	
# 	### Initialize hidden state
# 	# TODO: do initialization here.
# 	total_loss = 0
# 	train_accuracy, val_accuracy = 0, 0
	
# 	### Training
# 	for (batch, (inp, targ, lens)) in enumerate(train_dataset):		

# 		targ = Variable(targ)
# 		loss = 0
# 		print(inp.shape)
# 		print(inp.permute(1,0).shape)
# 		sys.exit(0)		
# 		predictions = model(Variable(inp.permute(1 ,0))) # TODO:don't need _   
			  
# 		loss += emoModel.loss_function(targ, predictions)
# 		batch_loss = (loss / int(targ.shape[1]))        
# 		total_loss += batch_loss
		
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()
		
# 		batch_accuracy = emoModel.accuracy(targ, predictions)
# 		train_accuracy += batch_accuracy
		
# 		if batch % 100 == 0:
# 			print(batch_loss.data.cpu().numpy())
# 			print('Epoch {} Batch {} Train. Loss {:.4f}'.format(epoch + 1,batch,batch_loss.data.cpu().numpy()[0]))
# 			# break
			
### Validating
test_accuracy = 0
for (batch, (inp, targ, lens)) in enumerate(test_dataset):        
	predictions = model(Variable(inp.permute(1, 0))) 
	# print(predictions)   
	# print(targ)
	batch_accuracy = emoModel.accuracy2(targ, predictions)
	print(batch_accuracy)
	test_accuracy += batch_accuracy

print(TEST_N_BATCH)
print("Test Accuracy: ", 100 * test_accuracy / len(input_tensor_test))


# model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}

#  (4) drop a checkpoint
# opt = {
# 	'vocab_inp_size':vocab_inp_size,
# 	'embedding_dim':embedding_dim, 
# 	'units':units, 
# 	'BATCH_SIZE':BATCH_SIZE, 
# 	'target_size':target_size	
# }

# checkpoint = {
#     'model': model_state_dict,
#     'vocabulary': globalVocab,   
#     'opt': opt
# }

# torch.save(checkpoint,
#            "../../models/classifier/emotion_classifier/emotion_classifier_2.pt")

