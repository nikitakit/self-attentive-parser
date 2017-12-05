import functools

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).cuda()
else:
    torch_t = torch
    from torch import from_numpy

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import chart_helper

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

def augment(scores, oracle_index):
    increment = torch.ones(scores.size())
    increment[oracle_index] = 0
    increment = Variable(increment, requires_grad=False)
    return scores + increment

class TopDownParser(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("No top-down parser implemented in pytorch")

class ChartParser(nn.Module):
    # We never actually call forward() end-to-end as is typical for pytorch
    # modules, but this inheritance brings in good stuff like state dict
    # management.
    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            dropout,
    ):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = nn.Embedding(tag_vocab.size, tag_embedding_dim)
        self.word_embeddings = nn.Embedding(word_vocab.size, word_embedding_dim)

        self.pre_lstm_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=tag_embedding_dim + word_embedding_dim,
            hidden_size=lstm_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=False)

        self.f_label = nn.Sequential(
            nn.Linear(2 * lstm_dim, label_hidden_dim),
            nn.ReLU(),
            nn.Linear(label_hidden_dim, label_vocab.size - 1),
            )


        if use_cuda:
            self.cuda()

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model):
        res = cls(**spec)
        if use_cuda:
            res.cpu()
        res.load_state_dict(model)
        if use_cuda:
            res.cuda()
        return res

    def parse(self, sentence, gold=None):
        is_train = gold is not None
        self.train(is_train)

        tag_idxs = np.zeros((len(sentence) + 2, 1), dtype=int)
        word_idxs = np.zeros_like(tag_idxs)

        for i, (tag, word) in enumerate([(START, START)] + sentence + [(STOP, STOP)]):
            tag_idxs[i, 0] = self.tag_vocab.index(tag)
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_idxs[i, 0] = self.word_vocab.index(word)

        tag_idxs = Variable(torch.from_numpy(tag_idxs), requires_grad=False, volatile=not is_train)
        word_idxs = Variable(torch.from_numpy(word_idxs), requires_grad=False, volatile=not is_train)

        if use_cuda:
            tag_idxs = tag_idxs.cuda()
            word_idxs = word_idxs.cuda()

        embeddings = torch.cat([
            self.tag_embeddings(tag_idxs),
            self.word_embeddings(word_idxs)
            ], dim=2)

        lstm_outputs, _ = self.lstm(self.pre_lstm_dropout(embeddings))
        lstm_outputs = lstm_outputs.squeeze(1)

        lstm_outputs_rearranged = torch.cat([
            lstm_outputs[:-1,:self.lstm_dim],
            -lstm_outputs[1:,self.lstm_dim:], # negative for compatibility with dynet code
            ], 1)

        span_features = (torch.unsqueeze(lstm_outputs_rearranged, 0)
                         - torch.unsqueeze(lstm_outputs_rearranged, 1))
        label_scores_chart = self.f_label(span_features)
        label_scores_chart = label_scores_chart.cpu()
        label_scores_chart = torch.cat([
            Variable(torch.zeros(label_scores_chart.size(0), label_scores_chart.size(1), 1), requires_grad=False),
            label_scores_chart
            ], 2)

        decoder_args = dict(
            sentence_len=len(sentence),
            label_scores_chart=label_scores_chart.data.numpy(),
            gold=gold,
            label_vocab=self.label_vocab,
            is_train=is_train)

        if is_train:
            p_score, p_i, p_j, p_label, p_augment = chart_helper.decode(False, **decoder_args)
            g_score, g_i, g_j, g_label, g_augment = chart_helper.decode(True, **decoder_args)

            p_i = torch.from_numpy(p_i)
            p_j = torch.from_numpy(p_j)
            p_label = torch.from_numpy(p_label)
            g_i = torch.from_numpy(g_i)
            g_j = torch.from_numpy(g_j)
            g_label = torch.from_numpy(g_label)

            loss = label_scores_chart[p_i, p_j, p_label].sum() + p_augment - label_scores_chart[g_i, g_j, g_label].sum()
            # during training, we don't actually need to construct a tree
            return None, loss
        else:
            # The optimized cython decoder implementation doesn't actually
            # generate trees, only scores and span indices. When converting to a
            # tree, we assume that the indices follow a preorder traversal.
            score, p_i, p_j, p_label, _ = chart_helper.decode(False, **decoder_args)
            last_splits = []
            idx = -1
            def make_tree():
                nonlocal idx
                idx += 1
                i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
                label = self.label_vocab.value(label_idx)
                if (i + 1) >= j:
                    tag, word = sentence[i]
                    tree = trees.LeafParseNode(int(i), tag, word)
                    if label:
                        tree = trees.InternalParseNode(label, [tree])
                    return [tree]
                else:
                    left_trees = make_tree()
                    right_trees = make_tree()
                    children = left_trees + right_trees
                    if label:
                        return [trees.InternalParseNode(label, children)]
                    else:
                        return children

            tree = make_tree()[0]
            return tree, score
