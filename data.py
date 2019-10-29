import os
from io import open
import torch
import random
import pickle

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2count = {}
        self.char2idx = {}
        self.idx2char = []
        self.char2count = {}

    def count_token(self, word):
        if word in self.word2count:
            self.word2count[word] += 1
        else:
            self.word2count[word] = 1
        for char in word:
            if char in self.char2count:
                self.char2count[char] += 1
            else:
                self.char2count[char] = 1

    def add_token(self, cut_freq, max_vocab_size):
        # sort words by their frequencies
        # word_sorted = [(word, freq),...]
        word_sorted = sorted(self.word2count.items(), key=lambda x:x[1], reverse=True)
        for word, freq in word_sorted:
            # Skip a word whose frequency is less than cut-off limit. If the cut_freq is a minus value, there are no frequency limit.
            if freq >= cut_freq or cut_freq < 0:
                if len(self.idx2word) < max_vocab_size and word not in self.word2idx:
                    self.idx2word.append(word)
                    self.word2idx[word] = len(self.idx2word) - 1
        # sort characters by their frequencies
        # char_sorted = [(char, freq),...]
        char_sorted = sorted(self.char2count.items(), key=lambda x:x[1], reverse=True)
        for char, freq in char_sorted:
            # Skip a character whose frequency is less than cut-off limit. If the cut_freq is a minus value, there are no frequency limit.
            if freq >= cut_freq or cut_freq < 0:
                if len(self.idx2char) < max_vocab_size and char not in self.char2idx:
                    self.idx2char.append(char)
                    self.char2idx[char] = len(self.idx2char) - 1
        self.set_pad()
        self.set_unk()
        # print(len(self.idx2word))
        # print(len(self.idx2char))

    def set_unk(self):
        self.idx2word.append("<unk>")
        self.word2idx["<unk>"] = len(self.idx2word) - 1
        self.idx2char.append("<unk>")
        self.char2idx["<unk>"] = len(self.idx2char) - 1
    
    def set_pad(self):
        self.idx2word.append("<pad>")
        self.word2idx["<pad>"] = len(self.idx2char) - 1
        self.idx2char.append("<pad>")
        self.char2idx["<pad>"] = len(self.idx2char) - 1
    
    def pad_id(self):
        return self.word2idx["<pad>"]

    def conv2id(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx["<unk>"]

    def conv2word(self, idx):
        return self.idx2word[idx]
    
    def char_pad_id(self):
        return self.char2idx["<pad>"]

    def char_conv2id(self, char):
        if char in self.char2idx:
            return self.char2idx[char]
        else:
            return self.char2idx["<unk>"]

    def char_conv2word(self, idx):
        return self.idx2char[idx]

    def tok_len(self):
        return len(self.idx2word)

    def char_len(self):
        return len(self.idx2char)

class Corpus(object):
    def __init__(self, args):
        self.args = args

    def load_dict(self): 
        """
        load dictionary from a pickle file
        """
        with open(self.args.dict, "rb") as f:
            self.dictionary = pickle.load(f)

    def load_data(self, path):
        """
        Inputs
        ----------
        path: string, it indicates the location of the dataset
        """
        self.train = self.tokenize(os.path.join(path, 'train.txt'), self.args.max_length)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), self.args.max_length)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), self.args.max_length)

    def make_dict(self, path):
        """
        Inputs
        ----------
        path: string, it indicates the location of the dataset
        """
        self.dictionary = Dictionary()
        path = os.path.join(path, 'train.txt')
        """ Add words and characters to a dictionary. """
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8", errors='ignore') as f:
            for line in f:
                words = ["<s>"] + line.split() + ["</s>"]
                if len(words) > self.args.max_length:
                    continue
                for word in words:
                    self.dictionary.count_token(word)
        self.dictionary.add_token(self.args.cut_freq, self.args.max_vocab_size)

    def tokenize(self, path, max_length=-1):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        sent_ids = []
        with open(path, 'r', encoding="utf8", errors='ignore') as f:
            for line in f:
                words = ["<s>"] + line.split() + ["</s>"]
                if max_length > 0  and len(words) > max_length:
                    continue
                sent_ids.append(self.sent2ids(words))
        return sent_ids
        
    def sent2ids(self, words):
        return [[self.dictionary.conv2id(word) for word in words], [[self.dictionary.char_conv2id(char) for char in word] for word in words]]

def data2batch(data, dictionary, bsz, flag_shuf=False, flag_char=False):
    if flag_shuf:
        random.shuffle(data)
    word_list = []
    char_list = []
    for sid in range(len(data)):
        assert(len(data[sid][0]) == len(data[sid][1]))
        word_list.append(data[sid][0])
        char_list.append(data[sid][1])
        if len(word_list) == bsz:
            assert(len(word_list) == len(char_list))
            yield list2tensor(word_list, char_list, dictionary)
            word_list = []
            char_list = []
    if len(word_list) > 0:
        yield list2tensor(word_list, char_list, dictionary)

def list2tensor(sent_list, char_list, dictionary):
    word_tensor = word_list2tensor(sent_list, dictionary)
    char_tensor, batch_size, max_seq_len, max_token_len = char_list2tensor(char_list, dictionary)
    return {
        "word" : {
            "index" : word_tensor.contiguous()
        },
        "char" : {
            "index" : char_tensor.contiguous(),
            "batch_size": batch_size,
            "seq_len" : max_seq_len,
            "tok_len" : max_token_len
        }
    }

def add_word_padding(word_list, dictionary):
    max_len = max([len(sent) for sent in word_list])
    batch = []
    for sent in word_list:
        batch.append([word for word in sent])
        while len(batch[-1]) < max_len:
            batch[-1].append(dictionary.pad_id())
    return batch

def word_list2tensor(word_list, dictionary):
    """
    args
    word_list: [batch_size, seq_len, token_id]
    dictionary: Dictionary

    return
    source, target [batch_size, seq_len, token_id]
    """
    word_list_padded = add_word_padding(word_list, dictionary)
    batch = torch.LongTensor(word_list_padded)
    return batch

def add_char_padding(sent_list, dictionary):
    """
    args
    sent_list: [batch_size, seq_len, token_len, char_id]
    dictionary: Dictionary

    return
    char_list: [batch_size*seq_len, token_len, char_id]
    """
    batch_size = len(sent_list)
    max_seq_len = max([len(sent) for sent in sent_list])
    max_token_len = max([len(word) for sent in sent_list for word in sent])
    # Add padding symbols
    batch = []
    for sid in range(len(sent_list)):
        for tid in range(len(sent_list[sid])):
            batch.append([])
            for cid in range(len(sent_list[sid][tid])):
                batch[-1].append(sent_list[sid][tid][cid])
            for char_pad_id in range(max_token_len - len(sent_list[sid][tid])):
                batch[-1].append(dictionary.char_pad_id())
        for seq_pad_id in range(max_seq_len - len(sent_list[sid])):
            batch.append([dictionary.char_pad_id() for cid in range(max_token_len)])
    return batch, batch_size, max_seq_len, max_token_len

def char_list2tensor(sent_list, dictionary):
    """
    args
    sent_list: [batch_size, seq_len, token_len, char_id]
    dictionary: Dictionary

    return
    batch: [token_len, batch_size*seq_len, char_id]
    """
    sent_list_padded, batch_size, max_seq_len, max_token_len = add_char_padding(sent_list, dictionary)
    batch = torch.LongTensor(sent_list_padded).contiguous()
    return batch, batch_size, max_seq_len, max_token_len
