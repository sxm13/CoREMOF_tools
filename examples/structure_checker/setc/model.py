"""
Module defining the graph attention network models' architecture.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GAT


actfn = {
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "softmax": nn.Softmax(),
}


class ErrGAT(nn.Module):

    def __init__(
        self,
        in_chan,
        n_hiddenchannels,
        out_chan,
        n_layers,
        num_targets,
        activation="relu",
        aggregate="add",
        attention_heads=4,
        dropout=0.5,
        att_dropout=0.0,
        edgeattrs=False,
        edgedims=None,
    ):
        super().__init__()
        torch.manual_seed(11111)
        self.n_layers = n_layers
        self.activate_fn = actfn[activation.lower()]
        self.aggr = aggregate
        self.use_edgeattrs = edgeattrs
        self.gatheads = attention_heads
        self.gatdrop = att_dropout
        self.outdrop = dropout
        self.edgeattrdim = edgedims
        self.targets = num_targets

        self.gat = GAT(
            in_channels=in_chan,
            hidden_channels=n_hiddenchannels,
            out_channels=out_chan,
            v2=True,
            num_layers=self.n_layers,
            heads=self.gatheads,
            dropout=self.gatdrop,
            act=self.activate_fn,
            norm=None,
            edge_dim=self.edgeattrdim,
            aggr=self.aggr,
        )

        gat_out = n_hiddenchannels if not out_chan else out_chan
        self.output = nn.Sequential(
            Linear(gat_out, gat_out // 2),
            self.activate_fn,
            nn.Dropout(p=self.outdrop),
            Linear(gat_out // 2, self.targets),
        )

    def forward(self, data, batch=None):
        x, edge_index, edge_attrs = data.x, data.edge_index, data.edge_attr
        x = self.gat(
            x,
            edge_index,
            edge_attr=edge_attrs if self.use_edgeattrs else None,
            batch=batch,
        )
        x = global_mean_pool(x, batch)
        # classifier
        x = self.output(x)
        return torch.sigmoid(x)
