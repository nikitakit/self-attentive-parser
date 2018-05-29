import numpy as np
import spacy
from spacy.tokens import Doc, Span, Token

from .base_parser import BaseParser, PTB_TOKEN_ESCAPE

__all__ = ['BeneparComponent', 'NonConstituentException']

# None is not allowed as a default extension value!
NOT_PARSED_SENTINEL = object()
Doc.set_extension('_constituent_data', default=NOT_PARSED_SENTINEL)

class NonConstituentException(Exception):
    pass

#%%
class ConstituentData():
    def __init__(self, starts, ends, labels, loc_to_constituent, label_vocab):
        self.starts = starts
        self.ends = ends
        self.labels = labels
        self.loc_to_constituent = loc_to_constituent
        self.label_vocab = label_vocab

class PartialConstituentData():
    def __init__(self):
        self.starts = [np.array([], dtype=int)]
        self.ends = [np.array([], dtype=int)]
        self.labels = [np.array([], dtype=int)]

    def finalize(self, doc, label_vocab):
        self.starts = np.hstack(self.starts)
        self.ends = np.hstack(self.ends)
        self.labels = np.hstack(self.labels)

        # TODO(nikita): Python for loops aren't very fast
        loc_to_constituent = np.full(len(doc), -1, dtype=int)
        prev = None
        for position in range(self.starts.shape[0]):
            if self.starts[position] != prev:
                prev = self.starts[position]
                loc_to_constituent[self.starts[position]] = position

        return ConstituentData(self.starts, self.ends, self.labels, loc_to_constituent, label_vocab)

#%%
class BeneparComponent(BaseParser):
    """
    Berkeley Neural Parser (benepar) component for spaCy.

    Sample usage:
    >>> nlp = spacy.load('en')
    >>> nlp.add_pipe(BeneparComponent("benepar_en"))
    >>> doc = nlp("The quick brown fox jumps over the lazy dog.")
    >>> sent = list(doc.sents)[0]
    >>> print(sent._.parse_string)

    This component is only responsible for constituency parsing. It should be
    preceded in the pipeline by other components that can perform sentence
    segmentation, tokenization, and part-of-speech tagging.
    """

    name = 'benepar'

    def __init__(self, filename, batch_size=64):
        """
        Load a parsing model given a short model name (e.g. "benepar_en") or a
        filename on disk.

        name (str): Model name, or path to uncompressed TensorFlow model graph
        batch_size (int): Maximum number of sentences to process per batch
        """
        super(BeneparComponent, self).__init__(filename, batch_size)

    def __call__(self, doc):
        constituent_data = PartialConstituentData()
        for parse_raw, sent in self._batched_parsed_raw(self._process_doc(doc)):
            # The optimized cython decoder implementation doesn't actually
            # generate trees, only scores and span indices. Indices follow a
            # preorder traversal, which is also the order the ConstituentData
            # class requires.
            score, p_i, p_j, p_label = parse_raw

            # To remove null-labelled constituents that are used for
            # binarization, but not null-labeled terminal nodes
            valid = (p_label != 0) | (p_i + 1 == p_j)

            constituent_data.starts.append(p_i[valid] + sent.start)
            constituent_data.ends.append(p_j[valid] + sent.start)
            constituent_data.labels.append(p_label[valid])

        doc._._constituent_data = constituent_data.finalize(doc, self._label_vocab)
        return doc

    def _process_doc(self, doc):
        for sent in doc.sents:
            yield [token.text for token in sent], sent

#%%

def get_constituent(span):
    constituent_data = span.doc._._constituent_data
    if constituent_data is NOT_PARSED_SENTINEL:
        raise Exception("No constituency parse is available for this document. Consider adding a BeneparComponent to the pipeline.")

    search_start = constituent_data.loc_to_constituent[span.start]
    if span.start + 1 < len(constituent_data.ends):
        search_end = constituent_data.loc_to_constituent[span.start + 1]
    else:
        search_end = len(constituent_data.ends)
    found_position = None
    for position in range(search_start, search_end):
        if constituent_data.ends[position] <= span.end:
            if constituent_data.ends[position] == span.end:
                found_position = position
            break

    if found_position is None:
        raise NonConstituentException("Span is not a constituent: {}".format(span))
    return constituent_data, found_position

def get_labels(span):
    constituent_data, position = get_constituent(span)
    label_num = constituent_data.labels[position]
    return constituent_data.label_vocab[label_num]

def parse_string(span):
    constituent_data, position = get_constituent(span)
    label_vocab = constituent_data.label_vocab
    doc = span.doc

    # Python 2 doesn't support "nonlocal", so wrap idx in a list
    idx_cell = [position - 1]
    def make_str():
        idx_cell[0] += 1
        i, j, label_idx = constituent_data.starts[idx_cell[0]], constituent_data.ends[idx_cell[0]], constituent_data.labels[idx_cell[0]]
        label = label_vocab[label_idx]
        if (i + 1) >= j:
            token = doc[i]
            s = "({} {})".format(token.tag_, PTB_TOKEN_ESCAPE.get(token.text, token.text))
        else:
            children = []
            while ((idx_cell[0] + 1) < len(constituent_data.starts)
                and i <= constituent_data.starts[idx_cell[0] + 1]
                and constituent_data.ends[idx_cell[0] + 1] <= j):
                children.append(make_str())

            s = " ".join(children)

        for sublabel in reversed(label):
            s = "({} {})".format(sublabel, s)
        return s

    return make_str()

def get_subconstituents(span):
    constituent_data, position = get_constituent(span)
    label_vocab = constituent_data.label_vocab
    doc = span.doc

    while position < len(constituent_data.starts):
        start = constituent_data.starts[position]
        end = constituent_data.ends[position]

        if span.end <= start or span.end < end:
            break

        yield doc[start:end]
        position += 1

def get_child_spans(span):
    constituent_data, position = get_constituent(span)
    label_vocab = constituent_data.label_vocab
    doc = span.doc

    child_start_expected = span.start
    position += 1
    while position < len(constituent_data.starts):
        start = constituent_data.starts[position]
        end = constituent_data.ends[position]

        if span.end <= start or span.end < end:
            break

        if start == child_start_expected:
            yield doc[start:end]
            child_start_expected = end

        position += 1

def get_parent_span(span):
    constituent_data, position = get_constituent(span)
    label_vocab = constituent_data.label_vocab
    doc = span.doc
    sent = span.sent

    position -= 1
    while position >= 0:
        start = constituent_data.starts[position]
        end = constituent_data.ends[position]

        if start <= span.start and span.end <= end:
            return doc[start:end]
        if end < span.sent.start:
            break
        position -= 1

    return None

#%%

Span.set_extension('labels', getter=get_labels)
Span.set_extension('parse_string', getter=parse_string)
Span.set_extension('constituents', getter=get_subconstituents)
Span.set_extension('parent', getter=get_parent_span)
Span.set_extension('children', getter=get_child_spans)

Token.set_extension('labels', getter=lambda token: get_labels(token.doc[token.i:token.i+1]))
Token.set_extension('parse_string', getter=lambda token: parse_string(token.doc[token.i:token.i+1]))
Token.set_extension('parent', getter=lambda token: get_parent_span(token.doc[token.i:token.i+1]))
