import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from typing import Union, Optional, Callable
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax
import argparse
from torch.utils.data import random_split
import numpy as np
import time
from el import EncourageLoss



class SAGPooling(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers

    if :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`:

        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    if :obj:`min_score` :math:`\tilde{\alpha}` is a value in [0, 1]:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Projections scores are learned based on a graph neural network layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            calculating projection scores (one of
            :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv`,
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.SAGEConv`). (default:
            :class:`torch_geometric.nn.conv.GraphConv`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5,
                 GNN: Callable = GraphConv, min_score: Optional[float] = None,
                 multiplier: float = 1.0, nonlinearity: Callable = torch.tanh,
                 **kwargs):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = GNN(in_channels, 1, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn, edge_index).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self):
        return '{}({}, {}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.gnn.__class__.__name__,
            self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)


# from gc_old_sagpool.py
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0,
                    help='seed , <args.runs means seed= turn')
parser.add_argument('--runs', type=int, default=1,
                    help='')
parser.add_argument('--gpu', type=int, default=-1,
                    help='-1 means cpu')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')  # 60 for top-k pooling
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,  # 0.0000 for top-k pooling
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,  # 0.8 for top-k pooling
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,   # 200 for top-k pooling
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--conv_method', type=str, default='GCNConv', choices=['GCNConv', 'GraphConv', 'GATConv','GATConvGI',"GATConvAG2","GATConvAG2Add","GATConvAG2Addr"],
                    help='how to learn features')
# 原来默认的两个conv都是graph conv， old sag pool 里默认都是GCNConv
parser.add_argument('--pooling_method', type=str, default='TopKPooling', choices=['TopKPooling', 'SAGPooling',],
                    help='how to select')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv', choices=['GCNConv', 'GraphConv', 'GATConv'],
                    help='to create alpha for top-k selection')
parser.add_argument('--save_best_model', type=int, default=0,
                    help='whether to save the best model')
parser.add_argument('--print_every_epoch', type=int, default=0,
                    help='whether to print at every epoch')
parser.add_argument('--print_every_run', type=int, default=0,
                    help='whether to print at every epoch')
parser.add_argument('--attention_heads', type=int, default=1,
                    help='')
parser.add_argument('--bonus_start', type=float, default=0,
                    help='')
parser.add_argument('--bonus_gamma', type=int, default=0,
                    help='')
parser.add_argument('--bonus_rho', type=float, default=1,
                    help='')
parser.add_argument('--log_end', type=float, default=1,
                    help='')
parser.add_argument('--whole_rho', type=float, default=0,
                    help='')

args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.nhid = args.nhid
        self.pooling_ratio = args.pooling_ratio  # 0.8 for top-k, 0.5 for sag_pool
        self.dropout_ratio = args.dropout_ratio  # 0.5 for top-k
        if args.conv_method == 'GATConv':
            self.conv1 = eval(args.conv_method)(dataset.num_features, int(self.nhid/args.attention_heads), heads=args.attention_heads,)
        else:
            self.conv1 = eval(args.conv_method)(dataset.num_features, self.nhid)

        if args.conv_method == 'GATConv':
            self.conv2 = eval(args.conv_method)(self.nhid, int(self.nhid/args.attention_heads), heads=args.attention_heads,)

        else:
            self.conv2 = eval(args.conv_method)(self.nhid, self.nhid)

        if args.conv_method == 'GATConv':
            self.conv3 = eval(args.conv_method)(self.nhid, int(self.nhid/args.attention_heads), heads=args.attention_heads,)
        else:
            self.conv3 = eval(args.conv_method)(self.nhid, self.nhid)

        if args.pooling_method == 'TopKPooling':
            self.pool1 = eval(args.pooling_method)(self.nhid, ratio=self.pooling_ratio)
        elif args.pooling_method == 'SAGPooling':
            self.pool1 = eval(args.pooling_method)(self.nhid, ratio=self.pooling_ratio, GNN=eval(args.pooling_layer_type))
        if args.pooling_method == 'TopKPooling':
            self.pool2 = eval(args.pooling_method)(self.nhid, ratio=self.pooling_ratio)
        elif args.pooling_method == 'SAGPooling':
            self.pool2 = eval(args.pooling_method)(self.nhid, ratio=self.pooling_ratio, GNN=eval(args.pooling_layer_type))
        if args.pooling_method == 'TopKPooling':
            self.pool3 = eval(args.pooling_method)(self.nhid, ratio=self.pooling_ratio)
        elif args.pooling_method == 'SAGPooling':
            self.pool3 = eval(args.pooling_method)(self.nhid, ratio=self.pooling_ratio, GNN=eval(args.pooling_layer_type))

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


def train(model, optimizer, el_loss=None):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        if el_loss is not None:
            loss, _ = el_loss(output, data.y)
        else:
            loss = F.nll_loss(output, data.y, reduction='mean')
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(training_set)


def test(model, loader):
    model.eval()
    loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)

def train_val_test_print_log(model, optimizer):
    # min_loss = 1e10
    max_val_acc = 0
    patience = 0
    best_log = ''
    time_train, time_test = 0, 0
    each_run_log_np = np.zeros(8)
    if args.bonus_gamma == 0:
        el_loss = None
    else:
        el_loss = EncourageLoss(args)
    for epoch in range(args.epochs):
        if el_loss is not None:
            el_loss.set_epoch(epoch)
        model.train()
        if epoch == 40:
            train_begin = time.perf_counter()
        train_loss = train(model,optimizer, el_loss=el_loss)
        if epoch == 40:
            train_end = time.perf_counter()
            time_train = train_end - train_begin
        train_acc, _ = test(model, train_loader)
        val_acc, val_loss = test(model, val_loader)
        if epoch == 40:
            test_begin = time.perf_counter()
        test_acc, test_loss = test(model, test_loader)
        if epoch == 40:
            test_end = time.perf_counter()
            time_test = test_end - test_begin
        epoch_log = 'Epoch: {:03d}, Train Loss: {:.3f} Acc: {:.4f}, Val:  Loss: {:.3f} Acc: {:.4f}, Test: Acc: {:.4f} '.\
            format(epoch, train_loss, train_acc, val_loss, val_acc, test_acc)
        if args.print_every_epoch:
            print(epoch_log)
        if val_acc > max_val_acc:
            if args.save_best_model:
                torch.save(model.state_dict(), args.dataset + ' latest.pth')
                print("Model saved at epoch{}".format(epoch))
            if args.print_every_run:
                print('New best at ' + epoch_log)
            best_log = epoch_log
            each_run_log_np[0] = epoch
            each_run_log_np[1] = train_loss
            each_run_log_np[2] = train_acc
            each_run_log_np[3] = val_loss
            each_run_log_np[4] = val_acc
            each_run_log_np[5] = test_acc
            max_val_acc = val_acc
            patience = 0
        else:
            patience += 1
        if args.print_every_run:
            if epoch % 5 == 1:
                print(epoch_log)
        if patience > args.patience:
            break
    each_run_log_np[6] = time_train
    each_run_log_np[7] = time_test
    if args.print_every_run:
        print('All best at ' + best_log + 'each epoch, train_time:{:.3f}, test_time:{:.3f}'.format(each_run_log_np[6], each_run_log_np[7]))
    return each_run_log_np

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
dataset = TUDataset(path, name=args.dataset)
num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
device = torch.device(args.gpu if args.gpu >= 0 else 'cpu')
runs_ar = np.zeros((args.runs, 8))

for turn in range(args.runs):
    if args.seed < args.runs:
        seed = turn
    else:
        seed = args.seed
    torch.manual_seed(seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    #  Because these tasks have less  well-classified examples (although they are binary classification tasks, trained
    #  model only get about $70$ accuracy for these tasks), in the setting of encouraging loss, bonus is added in the
    #  later stages of training ($10$ epochs before the epoch with averaged best accuracy on the valid set, we adopt the
    #  number of $10$ and since last $10$ epochs before the validated best epoch have good accuracy).
    args.defer_start = 25 if args.dataset=='PROTEINS' else 130
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    model = Net(args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    runs_ar[turn] = train_val_test_print_log(model, optimizer)

print('Epoch: {:3.1f}, Train Loss: {:.3f} Acc: {:.4f}, Val:  Loss: {:.3f} Acc: {:.4f}, \n Test: Acc: {:.4f}, std: {:.4f},'
      'each epoch train_time:{:.3f}, test_time:{:.3f}'.format(*(list(np.mean(runs_ar[:,:6],axis=0))), np.std(runs_ar[:,5]),*(list(np.mean(runs_ar[:,6:],axis=0)))))


