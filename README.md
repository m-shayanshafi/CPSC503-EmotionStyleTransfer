## Emotion Style Transfer on Tweets

This project applies style transfer on tweets. Tweets expressing various emotions (anger,joy etc) are rewritten expressing a different sentiment (sadness etc) through [back-translation](https://arxiv.org/abs/1804.09000). 

### Requirements
 1. Python 3.6 
 2. PyTorch 0.3

### Dataset:													

1. Download the Twitter dataset with tweets annotated with emotions [here](https://github.com/omarsar/nlp_pytorch_tensorflow_notebooks/blob/master/data/merged_training.pkl). Store the .pkl file in ``data/emotion/train`` and create a test folder at ``data/emotion/test``

2. Convert pickle file to csv.

```
cd back-translation/utils
python pickle2csv.py
```

3. Generate the vocab of the dataset.

```
cd ../classifier/emotion_classifier/utils
python getVocabulary.py
```

4. Dowload the english--french and french--english models from the following link:

```bash
http://tts.speech.cs.cmu.edu/style_models/english_french.tar
http://tts.speech.cs.cmu.edu/style_models/french_english.tar
```
Place these models in the `models/translation` folder.

5. Train emotion classifier
```
python train.py
```

6. Train the back translation model and do style transfer.

```
cd style_decoder
bash emotion_example.sh
```

## Acknowledgements
The code used to train the back-translation model is adapted from the accompanying [open source repository](https://github.com/shrimai/Style-Transfer-Through-Back-Translation) the paper.