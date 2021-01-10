import dataclasses
from typing import List, Optional, Tuple

import nltk
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
import tokenizations
import torch


@dataclasses.dataclass
class ParsingExample:
    """A single parse tree and sentence."""

    words: List[str]
    space_after: List[bool]
    tree: Optional[nltk.Tree] = None
    _pos: Optional[List[Tuple[str, str]]] = None

    def leaves(self):
        if self.tree is not None:
            return self.tree.leaves()
        elif self._pos is not None:
            return [word for word, tag in self._pos]
        else:
            return None

    def pos(self):
        if self.tree is not None:
            return self.tree.pos()
        else:
            return self._pos

    def without_gold_annotations(self):
        return dataclasses.replace(self, tree=None, _pos=self.pos())


class Treebank(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    @property
    def trees(self):
        return [x.tree for x in self.examples]

    @property
    def sents(self):
        return [x.words for x in self.examples]

    @property
    def tagged_sents(self):
        return [x.pos() for x in self.examples]

    def filter_by_length(self, max_len):
        return Treebank([x for x in self.examples if len(x.leaves()) <= max_len])

    def without_gold_annotations(self):
        return Treebank([x.without_gold_annotations() for x in self.examples])


def load_trees(const_path, text_path):
    """Load a treebank.

    The standard tree format presents an abstracted view of the raw text: the
    text is tokenized in a linguistically-motivated way, the tokenization does
    not preserve whitespace, and various characters are escaped or (in some
    languages) transliterated. This presents a mismatch for pre-trained
    transformer models, which typically do their own tokenization starting with
    raw unicode strings. A mismatch compared to pre-training often doesn't
    affect performance if you just want to report F1 scores within the same
    treebank, but it becomes a problem when releasing a parser for general use.
    The `text_path` argument allows specifying an auxiliary file that can be
    used to recover the original unicode string for the text. Files in the
    CoNLL-U format (https://universaldependencies.org/format.html) are
    accepted, but the parser also accepts similarly-formatted files with just
    three fields (ID, FORM, MISC) instead of the usual ten.

    TODO(nikita): add support for empty nodes with indexes i.1, i.2, etc.
        per https://universaldependencies.org/format.html#untokenized-text

    Args:
        const_path: Path to the file with one tree per line.
        text_path: Path to a file that provides the correct spelling for all
            tokens (without any escaping, transliteration, or other mangling)
            and information about whether there is whitespace after each token.

    Returns:
        A list of ParsingExample objects, which have the following attributes:
            - `tree` is an instance of nltk.Tree
            - `words` is a list of strings
            - `space_after` is a list of booleans
    """
    reader = BracketParseCorpusReader("", [const_path])
    trees = reader.parsed_sents()

    sents = []
    sent = []
    end_of_multiword = 0
    multiword_combined = ""
    multiword_separate = []
    multiword_sp_after = False
    with open(text_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                if sent:
                    sents.append(([w for w, sp in sent], [sp for w, sp in sent]))
                    sent = []
                    assert end_of_multiword == 0
                continue
            fields = line.split("\t", 2)
            num_or_range = fields[0]
            w = fields[1]

            if "-" in num_or_range:
                end_of_multiword = int(num_or_range.split("-")[1])
                multiword_combined = w
                multiword_separate = []
                multiword_sp_after = "SpaceAfter=No" not in fields[-1]
                continue
            elif int(num_or_range) <= end_of_multiword:
                multiword_separate.append(w)
                if int(num_or_range) == end_of_multiword:
                    _, separate_to_combined = tokenizations.get_alignments(
                        multiword_combined, multiword_separate
                    )
                    have_up_to = 0
                    for i, char_idxs in enumerate(separate_to_combined):
                        if i == len(multiword_separate) - 1:
                            word = multiword_combined[have_up_to:]
                            sent.append((word, multiword_sp_after))
                        elif char_idxs:
                            word = multiword_combined[have_up_to : max(char_idxs) + 1]
                            sent.append((word, False))
                            have_up_to = max(char_idxs) + 1
                        else:
                            sent.append(("", False))
                    assert int(num_or_range) == len(sent)
                    end_of_multiword = 0
                    multiword_combined = ""
                    multiword_separate = []
                    multiword_sp_after = False
                continue
            else:
                assert int(num_or_range) == len(sent) + 1
                sp = "SpaceAfter=No" not in fields[-1]
                sent.append((w, sp))
    assert len(trees) == len(sents)
    return Treebank(
        [
            ParsingExample(tree=tree, words=words, space_after=space_after)
            for tree, (words, space_after) in zip(trees, sents)
        ]
    )
