import csv
from six.moves import cPickle as pickle
import numpy as np
import pandas as pd
import util
import base64
from sklearn.model_selection import train_test_split

def main(path_pickle,path_csv):

	data = pd.read_pickle(path_pickle)

	# Divide into test/train/vaild here
	data_train, data_test = train_test_split(data,  test_size=0.2)
	
	#Sanity check	
	print(data.shape)	
	print(data_train.shape)
	print(data_test.shape)

	# Divide train by emotion
	createClassFiles(data_train, path_csv,'train')
	createClassFiles(data_test, path_csv,'test')


def createClassFiles(data, path_csv, trainOrTest):

	path_csv = path_csv+"/"+trainOrTest
	
	data_joy = data.loc[data['emotions'] == "joy"]
	data_sadness = data.loc[data['emotions'] == "sadness"]
	data_fear = data.loc[data['emotions'] == "fear"]
	data_love = data.loc[data['emotions'] == "love"]
	data_anger = data.loc[data['emotions'] == "anger"]
	data_surprise = data.loc[data['emotions'] == "surprise"]

	# Make separate file for each emotion
	with open(path_csv+"/joy",'w') as f:

		wr = csv.writer(f, delimiter='\t')

		for index, row_joy in data_joy.iterrows():
			
			line_joy = [str(row_joy['text'])]
			# print(line)
			wr.writerow(line_joy)

	with open(path_csv+"/sadness",'w') as f:

		wr = csv.writer(f, delimiter='\t')

		for index, row in data_sadness.iterrows():
			
			line = [str(row['text']) ]
			# print(line)
			wr.writerow(line)

	with open(path_csv+"/fear",'w') as f:

		wr = csv.writer(f, delimiter='\t')

		for index, row in data_fear.iterrows():
			
			line = [str(row['text']) ]
			# print(line)
			wr.writerow(line)

	with open(path_csv+"/love",'w') as f:

		wr = csv.writer(f, delimiter='\t')

		for index, row in data_love.iterrows():
			
			line = [str(row['text'])]
			# print(line)
			wr.writerow(line)

	with open(path_csv+"/anger",'w') as f:

		wr = csv.writer(f, delimiter='\t')

		for index, row in data_anger.iterrows():
			
			line = [str(row['text']) ]
			# print(line)
			wr.writerow(line)

	with open(path_csv+"/surprise",'w') as f:

		wr = csv.writer(f, delimiter='\t')

		for index, row in data_surprise.iterrows():
			
			line = [str(row['text'])]
			# print(line)
			wr.writerow(line)		

main('../data/emotion/train/merged_training.pkl', '../data/emotion/')