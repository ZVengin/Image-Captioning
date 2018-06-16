import nltk
import pickle
import os
import json

from collections import Counter
from pycocotools.coco import COCO

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
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    @classmethod
    def load_vocab(cls, data_dir):
        vocab_path=os.path.join(data_dir, 'vocabulary.pkl')
        with open(vocab_path,'rb') as f:
            vocab = pickle.load(f)

        return vocab




def build_vocab(raw_data_dir, data_dir):
    train_raw_file_path=os.path.join(raw_data_dir, 'captions_train2014.json')
    eval_raw_file_path=os.path.join(raw_data_dir, 'captions_val2014.json')
    ann_index_file_path=os.path.join(data_dir,'train_annotations.json')
    train_coco=COCO(train_raw_file_path)
    eval_coco=COCO(eval_raw_file_path)
    coco=COCO()
    count=Counter()

    coco.anns.update(train_coco.anns)
    coco.anns.update(eval_coco.anns)

    threshold=5

    with open(ann_index_file_path,'r') as f:
         ids=json.load(f)
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.word_tokenize(caption.lower())
        count.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

    words=[word for word,cnt in count.items() if cnt>=threshold]

    vocab=Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i,word in enumerate(words):
        vocab.add_word(word)

    vocab_file_path=os.path.join(data_dir,'vocabulary.pkl')
    with open(vocab_file_path,'wb') as f:
        pickle.dump(vocab,f)

    print("Total vocabulary size: {}".format(len(vocab)))

    print("Saved the vocabulary wrapper to '{}'".format(vocab_file_path))

if __name__=='__main__':
    build_vocab('../raw_data_dir/annotations','../data_dir/exp_1')
