import logging
import math
import os

import numpy as np

import pickle
import scipy.sparse as smat

import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from tqdm import tqdm
from cores.Ranker import Ranker
from cores.Matcher import Matcher
from scipy import sparse
from apex import amp
from cores.utils import set_seed, Logger, MultiLabelDataset

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


def data2tensor(feature, Y_label):
    all_inst_idx = torch.tensor([f["inst_idx"] for f in feature], dtype=torch.long)
    all_input_ids = torch.tensor([f["input_ids"] for f in feature], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in feature], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in feature], dtype=torch.long)

    if Y_label is None:
        return Dataset(all_inst_idx, all_input_ids, all_attention_mask, all_token_type_ids)
    Y_Acoo = Y_label.tocoo()
    Y_label = torch.sparse.FloatTensor(torch.LongTensor([Y_Acoo.row.tolist(), Y_Acoo.col.tolist()]),
                                       torch.FloatTensor(Y_Acoo.data.astype(np.float)))
    return TensorDataset(all_inst_idx, all_input_ids, all_attention_mask, all_token_type_ids, Y_label)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        code_path = "{}/code.npz".format(args.output_data_dir)
        self.n_clusters = smat.load_npz(code_path).shape[1]
        self.n_labels = smat.load_npz(code_path).shape[0]
        self.loss_function = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(p=args.dropout)

        self.bert_model = get_bert(args.model_type)
        self.ranker = Ranker(args, self.n_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):

        outs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[-1]
        # # get [cls] hidden states
        out = torch.cat([outs[-i][:, 0] for i in range(1, 6)], dim=-1)
        out = self.dropout(out)
        ra_out = self.ranker(out)

        return ra_out

    @staticmethod
    def set_device():
        # Setup CUDA
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def training(args, model):
        if args.is_maGCN:
            Matcher.Ma_train(args)
            torch.cuda.empty_cache()

        # load data
        trn_feat_path = "./save/{}/X.trn.bert.{}.pkl".format(args.dataset, args.max_seq_len)
        with open(trn_feat_path, "rb") as fin:
            X_trn = pickle.load(fin)

        trn_label_path = os.path.join(f'./data/{args.dataset}', "Y.trn.npz")
        Y_trn = smat.load_npz(trn_label_path)

        all_input_ids = torch.tensor([f["input_ids"] for f in X_trn], dtype=torch.long)
        all_attention_mask = torch.tensor([f["attention_mask"] for f in X_trn], dtype=torch.long)
        all_token_type_ids = torch.tensor([f["token_type_ids"] for f in X_trn], dtype=torch.long)
        train_data = MultiLabelDataset(all_input_ids, all_attention_mask, all_token_type_ids, Y_trn)
        train_loader = DataLoader(train_data, batch_size=args.batch, num_workers=3, shuffle=True)

        t_total = len(train_loader) // args.gradient_accumulation_steps * args.epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, "lr": args.lr},
            {'params': [p for n, p in model.bert_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, "lr": args.lr},
            {'params': [p for n, p in model.ranker.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, "lr": args.lr * args.ra_lr},
            {'params': [p for n, p in model.ranker.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, "lr": args.lr * args.ra_lr}
        ]

        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        model.to('cuda')
        amp.initialize(model, optimizer, opt_level="O1")
        model.zero_grad()
        set_seed(args)

        for epoch in range(int(args.epochs)):

            logger.info(" Epoch = %d", epoch + 1)
            train_raloss = 0.0
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
            for step, batch in loop:
                model.train()
                ra_out = model(
                    input_ids=batch[0].cuda(),
                    attention_mask=batch[1].cuda(),
                    token_type_ids=batch[2].cuda()
                )
                ra_loss = model.loss_function(ra_out, batch[3].cuda())

                loss = ra_loss
                loss /= args.gradient_accumulation_steps
                train_raloss += ra_loss.item()
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                loop.set_description(f'Epoch [{epoch + 1}/{args.epochs}]')
                loop.set_postfix(raloss=[train_raloss / (step + 1)],
                                 lr="Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
            log_str = f'Epoch [{epoch + 1}/{args.epochs}] Train loss [{train_raloss / (step + 1)}] '
            Logger('log_' + args.exp_name).log(log_str)

        model.save_model(f'save/{args.dataset}/model-{args.exp_name}.bin')

    @staticmethod
    def evaluate(args, model):

        # load data
        tst_feat_path = "./save/{}/X.tst.bert.{}.pkl".format(args.dataset, args.max_seq_len)
        with open(tst_feat_path, "rb") as fin:
            X_tst = pickle.load(fin)
        Y_tst_path = os.path.join(args.dataset_path, "Y.tst.npz")
        Y_tst = smat.load_npz(Y_tst_path)

        all_input_ids = torch.tensor([f["input_ids"] for f in X_tst], dtype=torch.long)
        all_attention_mask = torch.tensor([f["attention_mask"] for f in X_tst], dtype=torch.long)
        all_token_type_ids = torch.tensor([f["token_type_ids"] for f in X_tst], dtype=torch.long)
        test_data = MultiLabelDataset(all_input_ids, all_attention_mask, all_token_type_ids, Y_tst)
        test_loader = DataLoader(test_data, batch_size=args.batch, num_workers=4, shuffle=False)

        eval_ra_out = []
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
                ra_out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"]
                )
                ra_output = torch.sigmoid(ra_out).detach().cpu().numpy().tolist()
                eval_ra_out = eval_ra_out.__add__(ra_output)

        Y_pre = np.asarray(eval_ra_out, dtype=np.float16)
        out_Y_pre_path = os.path.join(args.output_data_dir, "Y_pre")
        smat.save_npz(out_Y_pre_path, smat.csr_matrix(Y_pre))

        if args.is_maGCN:
            matcher = Matcher(args)
            matcher.load_state_dict(torch.load(f'save/{args.dataset}/Matcher-{args.exp_name}-model.bin'))
            Matcher.Ma_eval(args, matcher)
            out_C_pre_path = os.path.join(args.output_data_dir, "C_pre.npy")
            C_pre = np.load(out_C_pre_path)

            code_path = "{}/code.npz".format(args.output_data_dir)
            code = smat.load_npz(code_path)

            if args.kk is None:
                idx = np.argsort(-C_pre, axis=1)[:, :code.shape[1]]
            C_result = np.zeros(C_pre.shape, int)
            for i in range(C_pre.shape[0]):
                C_result[i, idx[i]] = 1
            Y_result = np.multiply(np.dot(C_result, code.A.T), Y_pre)
            out_result_path = os.path.join(args.output_data_dir, "Result")
            np.save(out_result_path, Y_result)

        return 0
