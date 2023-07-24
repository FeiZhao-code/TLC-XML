import argparse
import logging
import os
import random

import numpy as np
import torch
import scipy.sparse as smat

from cores.Matcher import Matcher
from cores.model import Model
from cores.partition import label_partition
from cores.preprocess import do_label_embedding, do_proc_feat, create_Y_clusters
from cores.utils import set_seed, load_feat_data

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=False, default='')
    parser.add_argument("--do_preprocess", action='store_true', help="Whether to run preprocess.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run train.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test.")

    parser.add_argument('--dataset', type=str, required=False, default='Eurlex-4K')
    parser.add_argument("--model_type", type=str, default="bert", help="model-type [bert | xlnet | xlm | roberta]")
    parser.add_argument('--dataset_path', type=str, required=False, metavar="DIR")
    parser.add_argument("--cache_dir", default="./data", type=str,
                        help="store the pre-trained models downloaded from s3", )
    parser.add_argument('--output_data_dir', type=str, required=False, metavar="DIR")
    parser.add_argument("--label_emb_name", type=str, default="fast_text", help="")
    parser.add_argument('--is_maGCN', type=bool, required=False, default=True)
    parser.add_argument('--raLayers', type=int, required=False, default=1)
    parser.add_argument('--fp16', type=bool, required=False, default=False)
    parser.add_argument('--p', type=float, required=False, default=0.15)
    parser.add_argument('--t', type=float, required=False, default=0.05)

    parser.add_argument('--batch', type=int, required=False, default=8)
    parser.add_argument('--lr', type=float, required=False, default=3e-5)
    parser.add_argument('--ra_lr', type=float, required=False, default=2)
    parser.add_argument('--kk', type=int, required=False)
    parser.add_argument('--dropout', type=float, required=False, default=0.2)
    parser.add_argument('--seed', type=int, required=False, default=6742)
    parser.add_argument('--epochs', type=int, required=False, default=25)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_seq_len', type=int, required=False, default=512)
    parser.add_argument('--warmup_steps', type=int, required=False, default=100)

    args = parser.parse_args()
    return args


def preprocess(args):
    args.output_data_dir = f'save/{args.dataset}'
    if not os.path.exists(args.output_data_dir):
        os.makedirs(args.output_data_dir)

    # load feature
    trn_input_text_path = os.path.join(args.dataset_path, 'train_texts.txt')
    trn_xseq_list = load_feat_data(trn_input_text_path)
    tst_input_text_path = os.path.join(args.dataset_path, 'test_texts.txt')
    tst_xseq_list = load_feat_data(tst_input_text_path)
    # load label matrix
    trn_label_path = os.path.join(args.dataset_path, "Y.trn.npz")
    tst_label_path = os.path.join(args.dataset_path, "Y.tst.npz")
    Y_trn = smat.load_npz(trn_label_path)
    Y_tst = smat.load_npz(tst_label_path)
    label_map_path = "{}/label_map.txt".format(args.dataset_path)
    id2label = [line.strip() for line in open(label_map_path, 'r', encoding='ISO-8859-1')]


    do_label_embedding(args, id2label)
    #
    do_proc_feat(args, trn_xseq_list, tst_xseq_list)

    label_partition(args, n_ids=Y_trn.shape[0], Y=Y_trn, n_label=Y_trn.shape[1])

    create_Y_clusters(args, Y_trn, Y_tst)

    print("Finish Preprocess !")


def main():
    args = get_args()
    if args.exp_name == '':
        args.exp_name = f'{args.dataset}_{args.model_type}'
    args.dataset_path = f'./data/{args.dataset}'
    args.output_data_dir = f'./save/{args.dataset}'
    # Set seed
    set_seed(args)

    print(f'do [{args.exp_name}] exp')

    # Preprocess
    # preprocess(args)

    # model
    model = Model(args)

    # Testing
    if args.do_test:
        model.load_state_dict(torch.load(f'save/{args.dataset}/model-{args.exp_name}.bin'))
        Model.evaluate(args, model)
    # Training
    Model.training(args, model)


if __name__ == "__main__":
    main()
