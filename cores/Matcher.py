import logging
import math
import os

import tqdm
import time
import cProfile
import numpy as np
# from apex import amp
import pickle
import scipy.sparse as smat

import torch
from sklearn import metrics
from torch import nn
from torch.nn import Parameter
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from cores.utils import sigmoid, softmax, set_seed, Logger

logger = logging.getLogger(__name__)


def get_bert(bert_name, output_dir=None):
    if 'roberta' in bert_name:
        print('load roberta-base')
        model_config = RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained('roberta-base', config=model_config)
    elif 'xlnet' in bert_name:
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config)
    else:
        print('load bert-base-uncased')
        model_config = BertConfig.from_pretrained('bert-base-uncased', cache_dir='../data')
        model_config.output_hidden_states = True
        if output_dir is None:
            bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)
        else:
            bert = BertModel.from_pretrained(output_dir, config=model_config)
    return bert


def data2tensor(feature, label):
    all_input_ids = torch.tensor([f["input_ids"] for f in feature], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in feature], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in feature], dtype=torch.long)

    if label is None:
        return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

    label = torch.tensor(label.A, dtype=torch.float)
    return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, label)


class Matcher(nn.Module):
    def __init__(self, args):
        super(Matcher, self).__init__()
        self.args = args
        code_path = "{}/code.npz".format(args.output_data_dir)
        self.n_clusters = smat.load_npz(code_path).shape[1]
        self.bert_model = get_bert(args.model_type)
        self.dropout = nn.Dropout(p=0.2)
        self.loss_function = nn.BCEWithLogitsLoss()

        self.matcher_layer = nn.Linear(5 * 768, self.n_clusters)

        cluster_emb_path = os.path.join(args.output_data_dir, "cluster_emb.npz")
        self.cluster_emb = torch.tensor(smat.load_npz(cluster_emb_path).A, dtype=torch.float32)

        out_C_adj_path = os.path.join(args.output_data_dir, "C_adj.npy")
        self.C_adj = torch.tensor(np.load(out_C_adj_path), dtype=torch.float32)

        self.gc1 = GCNLayer(self.cluster_emb.shape[1], 5 * 768)

    @staticmethod
    def Ma_train(args):
        model = Matcher(args)
        trn_feat_path = "./save/{}/X.trn.bert.{}.pkl".format(args.dataset, args.max_seq_len)
        with open(trn_feat_path, "rb") as fin:
            X_trn = pickle.load(fin)

        trn_cluster_path = "./save/{}/C.trn.npz".format(args.dataset)
        C_trn = smat.load_npz(trn_cluster_path)
        C_trn[C_trn > 0] = 1

        train_data = data2tensor(X_trn, C_trn)
        trainload = DataLoader(train_data, batch_size=args.batch, num_workers=3, shuffle=True)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, },
        ]
        epochs = 1 #args.epochs // 2
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        t_total = len(trainload) // args.gradient_accumulation_steps * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

        model.to('cuda')
        model.zero_grad()
        set_seed(args)

        for epoch in range(0, epochs):
            train_maloss = 0.0
            model.train()
            loop = tqdm.tqdm(enumerate(trainload), total=len(trainload))
            for step, batch in loop:
                inputs = {
                    "input_ids": batch[0].to('cuda'),
                    "attention_mask": batch[1].to('cuda'),
                    "token_type_ids": batch[2].to('cuda')
                }
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"]
                )
                labels = batch[3]
                loss = model.loss_function(outputs, labels.to('cuda'))
                train_maloss += loss.item()
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                loop.set_postfix(maloss=[train_maloss / (step + 1)])

            log_str = f'Epoch [{epoch + 1}/{epochs}] ma_loss [{train_maloss / (step + 1)}]'
            Logger('log_' + args.exp_name).log(log_str)

        model.save_model(f'save/{args.dataset}/Matcher-{args.exp_name}-model.bin')
        return 0

    @staticmethod
    def Ma_eval(args, model):

        # load data
        tst_feat_path = "./save/{}/X.tst.bert.{}.pkl".format(args.dataset, args.max_seq_len)
        with open(tst_feat_path, "rb") as fin:
            X_tst = pickle.load(fin)
        C_tst_path = "./save/{}/C.tst.npz".format(args.dataset)
        C_tst = smat.load_npz(C_tst_path)
        C_tst[C_tst > 0] = 1

        test_data = data2tensor(X_tst, C_tst)
        test_loader = DataLoader(test_data, batch_size=args.batch, num_workers=3, shuffle=False)

        eval_ma_out = []
        model.to('cuda')
        model.eval()

        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
        for step, batch in loop:
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0].to('cuda'),
                    "attention_mask": batch[1].to('cuda'),
                    "token_type_ids": batch[2].to('cuda')
                }
                ma_out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"]
                )
                # save out
                ma_output = torch.sigmoid(ma_out).detach().cpu().numpy().tolist()
                eval_ma_out = eval_ma_out.__add__(ma_output)

        C_pre = np.asarray(eval_ma_out)
        out_C_pre_path = os.path.join(args.output_data_dir, "C_pre")
        np.save(out_C_pre_path, C_pre)

        return 0

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[-1]
        # get [cls] hidden states
        out = torch.cat([outs[-i][:, 0] for i in range(1, 6)], dim=-1)
        bert_out = self.dropout(out)

        if self.args.is_maGCN:
            H = self.gc1(self.cluster_emb.to('cuda'), self.C_adj.to('cuda'))
            H = torch.relu(H)

            W = H.transpose(0, 1)
            out = torch.matmul(bert_out, W)
        else:
            out = self.matcher_layer(bert_out)
        return out


class GCNLayer(nn.Module):
    """
       Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
