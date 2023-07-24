import warnings

import click

warnings.filterwarnings('ignore')

from functools import partial
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
import argparse

from sklearn.preprocessing import MultiLabelBinarizer
from typing import Union, Optional, List, Iterable, Hashable

_all__ = ['get_precision', 'get_p_1', 'get_p_3', 'get_p_5', 'get_p_10',
          'get_ndcg', 'get_n_1', 'get_n_3', 'get_n_5', 'get_n_10', ]

TPredict = np.ndarray
TTarget = Union[Iterable[Iterable[Hashable]], csr_matrix]
TMlb = Optional[MultiLabelBinarizer]
TClass = Optional[List[Hashable]]


def get_precision(prediction: TPredict, targets: TTarget, top=5):
    idx = np.argsort(-prediction, axis=1)[:, :top]
    temp = np.zeros(prediction.shape)
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            temp[i, idx[i][j]] = 1
    prediction = sparse.csr_matrix(temp)
    return prediction.multiply(targets).sum() / (top * targets.shape[0])


get_p_1 = partial(get_precision, top=1)
get_p_3 = partial(get_precision, top=3)
get_p_5 = partial(get_precision, top=5)
get_p_10 = partial(get_precision, top=10)


def get_ndcg(prediction: TPredict, targets: TTarget, classes: TClass = None, top=5):
    log = 1.0 / np.log2(np.arange(top) + 2)
    dcg = np.zeros((targets.shape[0], 1))

    for i in range(top):
        idx = np.argsort(-prediction, axis=1)[:, i:i + 1].flatten()
        p = sparse.csr_matrix((idx[:, None] == np.arange(prediction.shape[1])).astype(float))
        dcg += p.multiply(targets).sum(axis=-1) * log[i]
    return np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top) - 1])


get_n_1 = partial(get_ndcg, top=1)
get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)
get_n_10 = partial(get_ndcg, top=10)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=False, metavar="DIR", default='Eurlex-4K')
    parser.add_argument('--results_path', required=False, metavar="DIR", default='')
    parser.add_argument("--targets_path", required=False, metavar="DIR", default='')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # args.results_path = f'save/{args.dataset}/Y_pre.npz'
    args.results_path = f'save/{args.dataset}/Result.npy'
    args.targets_path = f'data/{args.dataset}/Y.tst.npz'
    args.train_labels = f'data/{args.dataset}/Y.trn.npz'
    # res, targets = sparse.load_npz(args.results_path).A, sparse.load_npz(args.targets_path)
    res, targets = np.load(args.results_path), sparse.load_npz(args.targets_path)

    print('Precision@1,3,5:', get_p_1(res, targets), get_p_3(res, targets), get_p_5(res, targets))
    print('nDCG@1,3,5:', get_n_1(res, targets), get_n_3(res, targets), get_n_5(res, targets))


if __name__ == "__main__":
    main()
