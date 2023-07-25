import itertools
import json
import os
from os import path
import logging
import pickle
import re
import numpy as np
import torchtext
import torch
from tqdm import tqdm
import scipy.sparse as smat
from sklearn.preprocessing import normalize

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer, RobertaModel,
)

from cores.utils import csr_cosine_similarity, convert_to_binary, build_vocab, load_feat_data, get_embedding_from_glove

logger = logging.getLogger(__name__)


def do_label_embedding(args, id2label):
    n_label = len(id2label)

    features = torch.zeros(n_label, 300)
    for idx in tqdm(range(n_label)):
        features[idx] = get_embedding_from_fasttext(id2label[idx])
    label_embedding = smat.csr_matrix(features)
    label_embedding = normalize(label_embedding, axis=1, norm="l2")

    print("label_embedding {} {}".format(type(label_embedding), label_embedding.shape))
    label_embedding_path = "{}/L.{}.npz".format(args.output_data_dir, args.label_emb_name)

    smat.save_npz(label_embedding_path, label_embedding)


def get_embedding_from_fasttext(text):
    text = text.lower()

    text = re.sub(r"[^A-Za-z0-9(),_'`]", "", text)
    text = re.sub(r"\s{2,}", " ", text)

    fasttext = torchtext.vocab.FastText('simple')
    try:
        embed = []
        for t in text.split('_'):
            embed.append(fasttext[t])
        embed = torch.stack(embed)
        output = torch.mean(embed, dim=0)
        if torch.count_nonzero(output).item() == 0:
            return torch.randn((1, 300))
        return output
    except:
        return torch.randn((1, 300))


def do_proc_feat(args, trn_xseq_list, tst_xseq_list):
    # load pretrained model tokenizers
    args.model_type = args.model_type.lower()

    if 'roberta' in args.model_type:
        print('load roberta-base tokenizer')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    elif 'xlnet' in args.model_type:
        print('load xlnet-base-cased tokenizer')
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    else:
        print('load bert-base-uncased tokenizer')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # process train features
    # xseq_list = list(df[df['dataType'] == 'train']['text'])
    print("token train example")
    trn_features, trn_xseq_lens = proc_feat(
        args, trn_xseq_list, tokenizer,
        pad_on_left=bool(args.model_type in ["xlnet"]),
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )
    print("train text finish preprocess {}".format(args.dataset_path))
    print("trn_xseq: min={} max={} mean={} median={}".format(
        np.min(trn_xseq_lens), np.max(trn_xseq_lens),
        np.mean(trn_xseq_lens), np.median(trn_xseq_lens), )
    )

    # save trn features
    os.makedirs(args.output_data_dir, exist_ok=True)
    out_trn_feat_path = path.join(args.output_data_dir, "X.trn.{}.{}.pkl".format(args.model_type, args.max_seq_len))
    with open(out_trn_feat_path, "wb") as fout:
        pickle.dump(trn_features, fout, protocol=pickle.HIGHEST_PROTOCOL)

    # process test features
    # xseq_list = list(df[df['dataType'] == 'test']['text'])
    print("token test example")
    tst_features, tst_xseq_lens = proc_feat(
        args, tst_xseq_list, tokenizer,
        pad_on_left=bool(args.model_type in ["xlnet"]),
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )
    print("test text finish preprocess {}".format(args.dataset_path))
    print("tst_xseq: min={} max={} mean={} median={}".format(
        np.min(tst_xseq_lens), np.max(tst_xseq_lens),
        np.mean(tst_xseq_lens), np.median(tst_xseq_lens), )
    )

    # save tst features
    out_tst_feat_path = path.join(args.output_data_dir, "X.tst.{}.{}.pkl".format(args.model_type, args.max_seq_len))
    with open(out_tst_feat_path, "wb") as fout:
        pickle.dump(tst_features, fout, protocol=pickle.HIGHEST_PROTOCOL)


def proc_feat(args, xseq_list,
              tokenizer,
              pad_on_left=False,
              pad_token=0,
              pad_token_segment_id=0,
              mask_padding_with_zero=True):
    # convert raw text into tokens, and convert tokens into tok_ids

    features, xseq_lens = [], []
    for (inst_idx, xseq) in enumerate(tqdm(xseq_list)):

        inputs = tokenizer.encode_plus(
            text=xseq[:4096],
            # text=xseq,
            text_pair=None,
            add_special_tokens=True,
            max_length=args.max_seq_len,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        xseq_lens.append(len(input_ids))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # sanity check and logging
        assert len(input_ids) == args.max_seq_len, "Error with input length {} vs {}".format(len(input_ids),
                                                                                             args.max_seq_len)
        assert len(attention_mask) == args.max_seq_len, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                  args.max_seq_len)
        assert len(token_type_ids) == args.max_seq_len, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                  args.max_seq_len)

        cur_inst_dict = {
            'inst_idx': inst_idx,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        features.append(cur_inst_dict)
    # end for loop
    return features, xseq_lens


def create_Y_clusters(args, Y_trn, Y_tst):
    # load existing code
    code_path = "{}/code.npz".format(args.output_data_dir)
    label2cluster_csr = smat.load_npz(code_path)
    csr_codes = label2cluster_csr.nonzero()[1]

    assert Y_trn.shape[1] == label2cluster_csr.shape[0]
    # save C_trn and C_tst
    C_trn = Y_trn.dot(label2cluster_csr)
    C_tst = Y_tst.dot(label2cluster_csr)

    out_trn_label_path = os.path.join(args.output_data_dir, "C.trn.npz")
    out_tst_label_path = os.path.join(args.output_data_dir, "C.tst.npz")
    smat.save_npz(out_trn_label_path, C_trn)
    smat.save_npz(out_tst_label_path, C_tst)

    common_matrix = np.zeros([label2cluster_csr.shape[1], label2cluster_csr.shape[1]], int)
    n_ids = C_trn.shape[0]
    # Calculate the upper triangular matrix
    for i in range(n_ids):
        L = list(itertools.combinations(list(C_trn[i, :].indices), 2))
        for j in range(len(L)):
            common_matrix[L[j]] = common_matrix[L[j]] + 1
    top = np.triu(common_matrix)
    # Copy the lower triangular matrix
    common_matrix += top.T
    # Calculate the cumulative number of occurrences of each label
    c_sum = np.asarray(C_trn.sum(axis=0), dtype=float).squeeze()
    c_sum[c_sum <= 1e-6] = 1e-6
    adj = common_matrix / c_sum

    # filter low noisy
    adj[adj < args.t] = 0
    adj[adj >= args.t] = 1
    # reduce over-smoothing problem
    adj = adj * args.p / (adj.sum(0, keepdims=True) + 1e-6)
    adj = torch.tensor(adj + np.identity(label2cluster_csr.shape[1], np.int))
    adj = normalizeAdjacency(adj)

    out_C_adj_path = os.path.join(args.output_data_dir, "C_adj")
    np.save(out_C_adj_path, adj)
    #
    label_embedding_path = "{}/L.{}.npz".format(args.output_data_dir, args.label_emb_name)
    label_emb = smat.load_npz(label_embedding_path)
    cluster_emb = label2cluster_csr.T.dot(label_emb)
    out_cluster_emb_path = os.path.join(args.output_data_dir, "cluster_emb.npz")
    smat.save_npz(out_cluster_emb_path, cluster_emb)

    # save adj of labels #
    n_labels = Y_trn.shape[1]
    n_ids = Y_trn.shape[0]
    common_matrix = np.zeros([n_labels, n_labels], int)
    # Calculate the upper triangular matrix
    for i in range(n_ids):
        L = list(itertools.combinations(list(Y_trn[i, :].indices), 2))
        for j in range(len(L)):
            common_matrix[L[j]] = common_matrix[L[j]] + 1
    top = np.triu(common_matrix)
    # Copy the lower triangular matrix
    common_matrix += top.T
    # Calculate the cumulative number of occurrences of each label
    y_sum = np.asarray(Y_trn.sum(axis=0), dtype=float).squeeze()
    y_sum[y_sum < 1e-6] = 1e-6
    adj = common_matrix / y_sum

    # filter low noisy
    adj[adj < 0.005] = 0  #
    adj[adj >= 0.005] = 1  #
    # reduce over-smoothing problem
    adj = adj * args.p / (adj.sum(0, keepdims=True) + 1e-6)
    adj = torch.tensor(adj + np.identity(n_labels, np.int))
    adj = normalizeAdjacency(adj)

    out_Y_adj_path = os.path.join(args.output_data_dir, "Y_adj")
    smat.save_npz(out_Y_adj_path, smat.csr_matrix(adj))


def normalizeAdjacency(W):
    d = torch.sum(W, dim=1)
    d = 1 / torch.sqrt(d)
    D = torch.diag(d)
    return D @ W @ D
