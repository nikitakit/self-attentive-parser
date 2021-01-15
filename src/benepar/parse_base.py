from abc import ABC, abstractmethod
import dataclasses
from typing import Any, Iterable, List, Optional, Tuple, Union

import nltk
import numpy as np


class BaseInputExample(ABC):
    """Parser input for a single sentence (abstract interface)."""

    # Subclasses must define the following attributes or properties.
    # `words` is a list of unicode representations for each word in the sentence
    # and `space_after` is a list of booleans that indicate whether there is
    # whitespace after a word. Together, these should form a reversible
    # tokenization of raw text input. `tree` is an optional gold parse tree.
    words: List[str]
    space_after: List[bool]
    tree: Optional[nltk.Tree]

    @abstractmethod
    def leaves(self) -> Optional[List[str]]:
        """Returns leaves to use in the parse tree.

        While `words` must be raw unicode text, these should be whatever is
        standard for the treebank. For example, '(' in words might correspond to
        '-LRB-' in leaves, and leaves might include other transformations such
        as transliteration.
        """
        pass

    @abstractmethod
    def pos(self) -> Optional[List[Tuple[str, str]]]:
        """Returns a list of (leaf, part-of-speech tag) tuples."""
        pass


@dataclasses.dataclass
class CompressedParserOutput:
    """Parser output, encoded as a collection of numpy arrays.

    By default, a parser will return nltk.Tree objects. These have much nicer
    APIs than the CompressedParserOutput class, and the code involved is simpler
    and more readable. As a trade-off, code dealing with nltk.Tree objects is
    slower: the nltk.Tree type itself has some overhead, and algorithms dealing
    with it are implemented in pure Python as opposed to C or even CUDA. The
    CompressedParserOutput type is an alternative that has some optimizations
    for the sole purpose of speeding up inference.

    If trying a new parser type for research purposes, it's safe to ignore this
    class and the return_compressed argument to parse(). If the parser works
    well and is being released, the return_compressed argument can then be added
    with a dedicated fast implementation, or simply by using the from_tree
    method defined below.
    """

    # A parse tree is represented as a set of constituents. In the case of
    # non-binary trees, only the labeled non-terminal nodes are included: there
    # are no dummy nodes inserted for binarization purposes. However, single
    # words are always included in the set of constituents, and they may have a
    # null label if there is no phrasal category above the part-of-speech tag.
    # All constituents are sorted according to pre-order traversal, and each has
    # an associated start (the index of the first word in the constituent), end
    # (1 + the index of the last word in the constituent), and label (index
    # associated with an external label_vocab dictionary.) These are then stored
    # in three numpy arrays:
    starts: Iterable[int]  # Must be a numpy array
    ends: Iterable[int]  # Must be a numpy array
    labels: Iterable[int]  # Must be a numpy array

    # Part of speech tag ids as output by the parser (may be None if the parser
    # does not do POS tagging). These indices are associated with an external
    # tag_vocab dictionary.
    tags: Optional[Iterable[int]] = None # Must be None or a numpy array

    def without_predicted_tags(self):
        return dataclasses.replace(self, tags=None)

    def with_tags(self, tags):
        return dataclasses.replace(self, tags=tags)

    @classmethod
    def from_tree(
        cls, tree: nltk.Tree, label_vocab: dict, tag_vocab: Optional[dict] = None
    ) -> "CompressedParserOutput":
        num_words = len(tree.leaves())
        starts = np.empty(2 * num_words, dtype=int)
        ends = np.empty(2 * num_words, dtype=int)
        labels = np.empty(2 * num_words, dtype=int)

        def helper(tree, start, write_idx):
            nonlocal starts, ends, labels
            label = []
            while len(tree) == 1 and not isinstance(tree[0], str):
                if tree.label() != "TOP":
                    label.append(tree.label())
                tree = tree[0]

            if len(tree) == 1 and isinstance(tree[0], str):
                starts[write_idx] = start
                ends[write_idx] = start + 1
                labels[write_idx] = label_vocab["::".join(label)]
                return start + 1, write_idx + 1

            label.append(tree.label())
            starts[write_idx] = start
            labels[write_idx] = label_vocab["::".join(label)]

            end = start
            new_write_idx = write_idx + 1
            for child in tree:
                end, new_write_idx = helper(child, end, new_write_idx)

            ends[write_idx] = end
            return end, new_write_idx

        _, num_constituents = helper(tree, 0, 0)
        starts = starts[:num_constituents]
        ends = ends[:num_constituents]
        labels = labels[:num_constituents]

        if tag_vocab is None:
            tags = None
        else:
            tags = np.array([tag_vocab[tag] for _, tag in tree.pos()], dtype=int)

        return cls(starts=starts, ends=ends, labels=labels, tags=tags)

    def to_tree(self, leaves, label_from_index: dict, tag_from_index: dict = None):
        if self.tags is not None:
            if tag_from_index is None:
                raise ValueError(
                    "tags_from_index is required to convert predicted pos tags"
                )
            predicted_tags = [tag_from_index[i] for i in self.tags]
            assert len(leaves) == len(predicted_tags)
            leaves = [
                nltk.Tree(tag, [leaf[0] if isinstance(leaf, tuple) else leaf])
                for tag, leaf in zip(predicted_tags, leaves)
            ]
        else:
            leaves = [
                nltk.Tree(leaf[1], [leaf[0]])
                if isinstance(leaf, tuple)
                else (nltk.Tree("UNK", [leaf]) if isinstance(leaf, str) else leaf)
                for leaf in leaves
            ]

        idx = -1

        def helper():
            nonlocal idx
            idx += 1
            i, j, label = (
                self.starts[idx],
                self.ends[idx],
                label_from_index[self.labels[idx]],
            )
            if (i + 1) >= j:
                children = [leaves[i]]
            else:
                children = []
                while (
                    (idx + 1) < len(self.starts)
                    and i <= self.starts[idx + 1]
                    and self.ends[idx + 1] <= j
                ):
                    children.extend(helper())

            if label:
                for sublabel in reversed(label.split("::")):
                    children = [nltk.Tree(sublabel, children)]

            return children

        children = helper()
        return nltk.Tree("TOP", children)


class BaseParser(ABC):
    """Parser (abstract interface)"""

    @classmethod
    @abstractmethod
    def from_trained(
        cls, model_name: str, config: dict = None, state_dict: dict = None
    ) -> "BaseParser":
        """Load a trained parser."""
        pass

    @abstractmethod
    def parallelize(self, *args, **kwargs):
        """Spread out pre-trained model layers across GPUs."""
        pass

    @abstractmethod
    def parse(
        self,
        examples: Iterable[BaseInputExample],
        return_compressed: bool = False,
        return_scores: bool = False,
        subbatch_max_tokens: Optional[int] = None,
    ) -> Union[Iterable[nltk.Tree], Iterable[Any]]:
        """Parse sentences."""
        pass

    @abstractmethod
    def encode_and_collate_subbatches(
        self, examples: List[BaseInputExample], subbatch_max_tokens: int
    ) -> List[dict]:
        """Split batch into sub-batches and convert to tensor features"""
        pass

    @abstractmethod
    def compute_loss(self, batch: dict):
        pass
