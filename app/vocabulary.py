import nltk
import pickle
from collections import Counter
#
# This code is a Python script that use for mapping from text into numerical index
# which specify for visual storytelling (VIST) task. The input of this script is JSON
# formatted file downloadded from http://visionandlanguage.net/VIST/dataset.html.
# To download the dataset, we already provide a script downloader as follow:
# https://github.com/systems-ai-lab/visualstorytelling-codebase/blob/master/script/download_dataset.sh
# 
# If you use the download_dataset.sh script for downloading the dataset or directly 
# download from the VIST website, we will get two kind of text annotation such as 
# "SIS" and "DII". This script is intended to build the vovabulary from SIS text annotation.
#     

# Some part of this code is referenced to https://github.com/tkim-snu/GLACNet work. 

class Vocabulary(object):
    # This class is define the data structure of the vocabulary
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

class GenerateVocabulary(object):
    # This class will be initialized by the dataset class if the vocabulary file is not exist
    def __init__(self, sis_data_object, minimum_treshold, output_vocabulary):
        self.sis_data_object = sis_data_object
        self.minimum_treshold = minimum_treshold
        self.output_vocabulary = output_vocabulary
        self.generate(self.sis_data_object, self.minimum_treshold, self.output_vocabulary)

    def generate(self, sis_data_object, minimum_treshold, output_vocabulary):
        vist = sis_data_object
        counter = Counter()
        ids = vist['stories'].keys()

        for i, id in enumerate(ids):
            story = vist['stories'][id]
            for annotation in story:
                caption = annotation['text']
                tokens = []
                try:
                    tokens = nltk.tokenize.word_tokenize(caption.lower())
                except Exception:
                    pass
                counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] Tokenized the story captions." %(i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= minimum_treshold]

        vocab = Vocabulary()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        for i, word in enumerate(words):
            vocab.add_word(word)
        
        with open(output_vocabulary, 'wb') as f:
            pickle.dump(vocab, f)
            
        print("Total vocabulary size: %d" %len(vocab))
        print("Saved the vocabulary wrapper to '%s'" %output_vocabulary)

        return vocab