import functools

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

def augment(scores, oracle_index):
    # NOCUDA
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

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model):
        res = cls(**spec)
        res.load_state_dict(model)
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

        # NOCUDA
        embeddings = torch.cat([
            self.tag_embeddings(tag_idxs),
            self.word_embeddings(word_idxs)
            ], dim=2)

        lstm_outputs, _ = self.lstm(embeddings)
        lstm_outputs = lstm_outputs.squeeze(1)

        lstm_outputs_rearranged = torch.cat([
            lstm_outputs[:-1,:self.lstm_dim],
            -lstm_outputs[1:,self.lstm_dim:], # negative for compatibility with dynet code
            ], 1)

        span_features = (torch.unsqueeze(lstm_outputs_rearranged, 0)
                         - torch.unsqueeze(lstm_outputs_rearranged, 1))
        label_scores_chart = self.f_label(span_features)
        label_scores_chart = torch.cat([
            Variable(torch.zeros(label_scores_chart.size(0), label_scores_chart.size(1), 1), requires_grad=False),
            label_scores_chart
            ], 2)

        def helper(force_gold):
            if force_gold:
                assert is_train

            chart = {}

            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length

                    label_scores = label_scores_chart[left, right]

                    if is_train:
                        oracle_label = gold.oracle_label(left, right)
                        oracle_label_index = self.label_vocab.index(oracle_label)

                    if force_gold:
                        label = oracle_label
                        label_score = label_scores[oracle_label_index]
                    else:
                        if is_train:
                            label_scores = augment(label_scores, oracle_label_index)
                        label_scores_np = label_scores.data.numpy()
                        argmax_label_index = int(
                            label_scores_np.argmax() if length < len(sentence) else
                            label_scores_np[1:].argmax() + 1)
                        argmax_label = self.label_vocab.value(argmax_label_index)
                        label = argmax_label
                        label_score = label_scores[argmax_label_index]

                    if length == 1:
                        tag, word = sentence[left]
                        tree = trees.LeafParseNode(left, tag, word)
                        if label:
                            tree = trees.InternalParseNode(label, [tree])
                        chart[left, right] = [tree], label_score
                        continue

                    if force_gold:
                        oracle_splits = gold.oracle_splits(left, right)
                        oracle_split = min(oracle_splits)
                        best_split = oracle_split
                    else:
                        best_split = max(
                            range(left + 1, right),
                            key=lambda split:
                                chart[left, split][1].data[0] +
                                chart[split, right][1].data[0])

                    left_trees, left_score = chart[left, best_split]
                    right_trees, right_score = chart[best_split, right]

                    children = left_trees + right_trees
                    if label:
                        children = [trees.InternalParseNode(label, children)]

                    chart[left, right] = (
                        children, label_score + left_score + right_score)

            children, score = chart[0, len(sentence)]
            assert len(children) == 1
            return children[0], score

        tree, score = helper(False)
        if is_train:
            oracle_tree, oracle_score = helper(True)
            assert oracle_tree.convert().linearize() == gold.convert().linearize()
            correct = tree.convert().linearize() == gold.convert().linearize()
            loss = Variable(torch.zeros(1)) if correct else score - oracle_score # NOCUDA
            return tree, loss
        else:
            return tree, score
