import nltk
import pickle
from collections import Counter

class Vocabulary(object):
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
