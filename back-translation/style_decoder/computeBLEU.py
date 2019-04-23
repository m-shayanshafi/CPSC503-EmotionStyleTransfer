from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
import argparse
import sys


parser = argparse.ArgumentParser(description='computeBLEU.py')


parser.add_argument('-source_text', required=True,
                    help='Path to source file for emotion')

parser.add_argument('-target_text', required=True,
                    help='Path to target file for emotion')

preFixPath = "bleu_scores/"
avgFilePath = "bleu_scores/finalScores"

def readFile(fname):
	
	with open(fname) as f:
	    content = f.readlines()
	fileLines = [x.strip() for x in content]
	return fileLines

def writeScoresToFile(fname, bleuScores):

	with open(fname,'w+') as f:
		f.writelines(bleuScores)

	with open(avgFilePath,'a') as f:
		emotions = fname.split("/")[-1].replace(".txt","")
		lineToWrite = emotions + "\t" + bleuScores[-1]
		f.write(lineToWrite)


def compute_bleu(reference_sentence, predicted_sentence):
    """
    Given a reference sentence, and a predicted sentence, compute the BLEU similary between them.
    """

    reference_tokenized = word_tokenize(reference_sentence.lower())
    predicted_tokenized = word_tokenize(predicted_sentence.lower())
    return sentence_bleu([reference_tokenized], predicted_tokenized)

def main():

	opt = parser.parse_args()

	print(opt.source_text)
	print(opt.target_text)

	sourceSentences = readFile(opt.source_text)
	targetSentences = readFile(opt.target_text)

	if not len(sourceSentences) == len(targetSentences):
		print("Error. Source does not match target length")
		sys.exit(0)


	
	bleuScores = []

	idx = 0

	totalBleuScore = 0

	for sourceSentence in sourceSentences:

		targetSentence = targetSentences[idx]
		score = compute_bleu(sourceSentence, targetSentence)
		idx = idx+1
		totalBleuScore = totalBleuScore + score
		bleuScores.append(str(score) + "\n")

	avgBleuScore = totalBleuScore/idx

	bleuScores.append(str(avgBleuScore) + "\n")	

	fileName= preFixPath + opt.target_text.split("/")[-1]
	print(fileName)

	writeScoresToFile(fileName,bleuScores)	


if __name__ == '__main__':


	main()





