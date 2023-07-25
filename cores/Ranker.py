import logging
import os

import numpy as np
# from apex import amp
import scipy.sparse as smat
import torch

from torch import nn

# from transformers import RobertaTokenizerFast

logger = logging.getLogger(__name__)


class Ranker(nn.Module):
    def __init__(self, args, n_b, feature_layers=5):
        super(Ranker, self).__init__()

        self.args = args
        self.n_b = n_b,
        self.ranker_layer1 = nn.Linear(feature_layers * 768, n_b)

        self.relu = nn.LeakyReLU(negative_slope=0.2)

        out_Y_adj_path = os.path.join(args.output_data_dir, "Y_adj.npz")
        Y_adj = smat.load_npz(out_Y_adj_path)
        Y_Acoo = Y_adj.tocoo()
        self.Y_adj = torch.sparse.FloatTensor(torch.LongTensor([Y_Acoo.row.tolist(), Y_Acoo.col.tolist()]),
                                              torch.FloatTensor(Y_Acoo.data.astype(np.float)))

        self.agg = Aggregation(feature_layers * 768, n_b)

    def forward(self, input_feature):
        out = self.ranker_layer1(input_feature)
        for i in range(self.args.raLayers):
            out = out + self.agg(input_feature, self.Y_adj.to('cuda'))

        return out


class Aggregation(nn.Module):

    def __init__(self, in_features, out_features):
        super(Aggregation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features)

    def forward(self, input, adj):
        input = self.weight(input)
        output = torch.matmul(adj, input.T)
        output = output.T

        return output
