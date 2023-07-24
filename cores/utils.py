import datetime
import json
import logging
import os
import random
import re

import joblib
import numpy as np
import torch
import pandas as pd
import torchtext
from sklearn.preprocessing import MultiLabelBinarizer
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from typing import Sequence, Optional
from torch.utils.data import Dataset
import scipy.sparse as smat

from transformers import XLNetTokenizer

# RobertaTokenizerFast

logger = logging.getLogger(__name__)


class Logger:
    def __init__(self, name):
        self.name = name

    def log(self, text):
        with open(f'./log/{self.name}.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')


def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)


TDataX = Sequence[Sequence]
TDataY = Optional[smat.csr_matrix]


class MultiLabelDataset(Dataset):
    def __init__(self, all_input_ids, all_attention_mask, all_token_type_ids, data_y: TDataY = None, training=True):
        self.all_input_ids, self.all_attention_mask, self.all_token_type_ids, self.data_y, self.training = all_input_ids, all_attention_mask, all_token_type_ids, data_y, training

    def __getitem__(self, item):
        all_input_ids = self.all_input_ids[item]
        all_attention_mask = self.all_attention_mask[item]
        all_token_type_ids = self.all_token_type_ids[item]
        if self.training and self.data_y is not None:
            data_y = self.data_y[item].toarray().squeeze(0).astype(np.float32)
            return all_input_ids, all_attention_mask, all_token_type_ids, data_y
        else:
            return all_input_ids, all_attention_mask, all_token_type_ids

    def __len__(self):
        return len(self.all_input_ids)


def get_tokenizer(self):
    if 'roberta' in self.bert_name:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True)
    elif 'xlnet' in self.bert_name:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    else:
        tokenizer = BertWordPieceTokenizer(
            "data/.bert-base-uncased-vocab.txt",
            lowercase=True)
    return tokenizer


def csr_cosine_similarity(input_csr_matrix):
    similarity = input_csr_matrix * input_csr_matrix.T
    square_mag = similarity.diagonal()
    square_mag[square_mag < 1e-6] = 1e-6
    inv_square_mag = 1 / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    inv_mag = np.sqrt(inv_square_mag)
    return similarity.multiply(inv_mag).T.multiply(inv_mag)


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),?'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()


def load_feat_data(text_path):
    xseq_list = []
    with open(text_path) as f:
        for i in tqdm(f):
            xseq_list.append(i.replace('\n', ''))
    print(f'Created X_seq list of size {len(xseq_list)}')
    return xseq_list


def truncate_text(texts, max_len=500, padding_idx=0, unknown_idx=1):
    if max_len is None:
        return texts
    texts = np.asarray([list(x[:max_len]) + [padding_idx] * (max_len - len(x)) for x in texts])
    texts[(texts == padding_idx).all(axis=1), 0] = unknown_idx
    return texts


def convert_to_binary(text_file, label_file=None, max_len=None, vocab=None, pad='<PAD>', unknown='<UNK>'):
    with open(text_file, encoding='utf-8') as fp:
        texts = np.asarray([[vocab.get(word, vocab[unknown]) for word in line.split()]
                            for line in tqdm(fp, desc='Converting token to id', leave=False)])
    labels = None
    if label_file is not None:
        with open(label_file, encoding='utf-8') as fp:
            labels = np.asarray([[label for label in line.split()]
                                 for line in tqdm(fp, desc='Converting labels', leave=False)])
    return truncate_text(texts, max_len, vocab[pad], vocab[unknown]), labels


from gensim.models import KeyedVectors
from collections import Counter
from typing import Union, Iterable


def build_vocab(texts: Iterable, w2v_model: Union[KeyedVectors, str], vocab_size=500000,
                pad='<PAD>', unknown='<UNK>', sep='/SEP/', max_times=1, freq_times=1):
    if isinstance(w2v_model, str):
        w2v_model = KeyedVectors.load(w2v_model)
    emb_size = w2v_model.vector_size
    vocab, emb_init = [pad, unknown], [np.zeros(emb_size), np.random.uniform(-1.0, 1.0, emb_size)]
    counter = Counter(token for t in texts for token in set(t.split()))
    for word, freq in sorted(counter.items(), key=lambda x: (x[1], x[0] in w2v_model), reverse=True):
        if word in w2v_model or freq >= freq_times:
            vocab.append(word)
            # We used embedding of '.' as embedding of '/SEP/' symbol.
            word = '.' if word == sep else word
            emb_init.append(w2v_model[word] if word in w2v_model else np.random.uniform(-1.0, 1.0, emb_size))
        if freq < max_times or vocab_size == len(vocab):
            break
    return np.asarray(vocab), np.asarray(emb_init)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def softmax(x):
    """ softmax function """
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


def get_embedding_from_glove(text):
    text = text.lower()
    text = clean_string(text)
    text = text.replace('/', ',')
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    if len(text.split(',')) == 1:
        return glove[text]
    embed = []
    for t in text.split(','):
        embed.append(glove[t])
    embed = torch.stack(embed)
    return torch.mean(embed, dim=0)
