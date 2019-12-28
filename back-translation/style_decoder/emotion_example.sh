
# # Translation to french for each emotion - Train
python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/train/anger -output ../data/emotion/train/anger.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/train/fear -output ../data/emotion/train/fear.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/train/joy -output ../data/emotion/train/joy.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/train/love -output ../data/emotion/train/love.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/train/sadness -output ../data/emotion/train/sadness.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/train/surprise -output ../data/emotion/train/surprise.fr -replace_unk $true

# Translation to french for each emotion - Test
python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/test/anger -output ../data/emotion/test/anger.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/test/fear -output ../data/emotion/test/fear.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/test/joy -output ../data/emotion/test/joy.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/test/love -output ../data/emotion/test/love.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/test/sadness -output ../data/emotion/test/sadness.fr -replace_unk $true

python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/emotion/test/surprise -output ../data/emotion/test/surprise.fr -replace_unk $true

# # Preprocess the french data into english

#Train the emotion classifier 
array=(anger fear joy love sadness surprise)

for i in "${array[@]}"
do

	python preprocess.py -train_src ../data/emotion/train/$i.fr -train_tgt ../data/emotion/train/$i -valid_src ../data/emotion/test/$i.fr -valid_tgt ../data/emotion/test/$i -save_data data/$i\_generator -src_vocab ../models/translation/french_english/french_english_vocab.src.dict -tgt_vocab ../models/classifier/emotion_classifier/emotion_classifier_vocab.src.dic -seq_len 70
	
done

# Train the democratic style generator
python train_decoder.py -data data/anger_generator.train.pt -save_model trained_models/democratic_generator -classifier_model ../models/classifier/political_classifier/political_classifier.pt -encoder_model ../models/translation/french_english/french_english.pt -tgt_label 1

