import pickle
import random

import torch
from torch.utils.data import Dataset
import os
import numpy as np


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, 'rb') as f:
            if self.corpus_lines is None and not on_memory:
                raise NotImplemented
                # for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                #     self.corpus_lines += 1

            if on_memory:
                self.lines = pickle.load(f)
                self.lines = list(self.lines.reshape(-1, 400))

                # self.lines = [line[:-1].split("1")
                #               for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            raise NotImplemented
            # self.file = open(corpus_path, "r", encoding=encoding)
            # self.random_file = open(corpus_path, "r", encoding=encoding)
            #
            # for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
            #     self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1 = self.get_sent(item)
        t1_random, t1_label = self.random_word(t1)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        # t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        # t2 = t2_random + [self.vocab.eos_index]

        # t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        # t2_label = t2_label + [self.vocab.pad_index]

        # segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = t1
        bert_label = t1_label

        # padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        # bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": torch.tensor(bert_input),
                  "bert_label": torch.tensor(bert_label).long()}

        return output

    def random_word(self, sentence):
        tokens = sentence
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.1:
                prob /= 0.1

                # 80% randomly change token to mask token
                # if prob < 0.8:
                tokens[i] = self.vocab - 1

                # 10% randomly change token to random token
                # elif prob < 0.9:
                # else:
                #     tokens[i] = random.randrange(self.vocab-1)

                # 10% randomly change token to current token
                # else:
                #     tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(token)

            else:
                # tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(self.vocab)

        return tokens, output_label

    def get_sent(self, index):
        if self.on_memory:
            return self.lines[index]
        else:
            raise NotImplemented

        # output_text, label(isNotNext:0, isNext:1)
        # if random.random() > 0.5:
        #     return t1, t2, 1
        # else:
        #     return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item]
        else:
            raise NotImplemented
            # line = self.file.__next__()
            # if line is None:
            #     self.file.close()
            #     self.file = open(self.corpus_path, "r", encoding=self.encoding)
            #     line = self.file.__next__()
            #
            # t1, t2 = line[:-1].split("\t")
            # return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]


class ReversedDataset(Dataset):
    def __init__(self, corpus_path, label_path, vocab, seq_len, label_seqlen, train=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.label_seqlen = label_seqlen

        self.corpus_path = corpus_path
        self.label_path = label_path

        self.split = 0.7
        self.lines = self.load_data(corpus_path, train, seq_len)
        self.lines = list(self.lines.reshape(-1, seq_len))
        self.corpus_lines = len(self.lines)

        self.labels = self.load_data(label_path, train, label_seqlen)
        self.labels = list(self.labels.reshape(-1, label_seqlen))
        self.labels_lines = len(self.lines)

        assert self.labels_lines == self.corpus_lines

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        bert_input, bert_label = self.get_data(item)
        # bert_label = self.random_label(bert_label)

        output = {"bert_input": torch.tensor(bert_input),
                  "bert_label": torch.tensor(bert_label).long()}

        return output

    def get_data(self, index):
        return self.lines[index], self.labels[index]

    def random_label(self, label):
        # currently no use
        return label

    def load_data(self, data_dir, train=True, seqlen=256):
        if os.path.isdir(data_dir):
            files = os.listdir(data_dir)
            files = [f for f in files if '.dat' in f]
            data = [np.fromfile(os.path.join(data_dir, f), dtype=np.uint8) for f in files]
            raw_data = np.concatenate(data)
        else:
            raw_data = np.fromfile(data_dir, dtype=np.uint8)
        assert len(raw_data) % seqlen == 0, len(raw_data)
        split = (int(len(raw_data) * self.split) // seqlen) * seqlen
        if train:
            raw_data = raw_data[0:split]
        else:
            raw_data = raw_data[split:]
        return raw_data
