import numpy as np

from .downloader import load_trained_model
from ..parse_base import BaseParser, BaseInputExample
from .spacy_extensions import ConstituentData, NonConstituentException

import torch


class PartialConstituentData:
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

        return ConstituentData(
            self.starts, self.ends, self.labels, loc_to_constituent, label_vocab
        )


class SentenceWrapper(BaseInputExample):
    TEXT_NORMALIZATION_MAPPING = {
        "`": "'",
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‹": "'",
        "›": "'",
        "—": "--",  # em dash
    }

    def __init__(self, spacy_sent):
        self.sent = spacy_sent

    @property
    def words(self):
        return [
            self.TEXT_NORMALIZATION_MAPPING.get(token.text, token.text)
            for token in self.sent
        ]

    @property
    def space_after(self):
        return [bool(token.whitespace_) for token in self.sent]

    @property
    def tree(self):
        return None

    def leaves(self):
        return self.words

    def pos(self):
        return [(word, "UNK") for word in self.words]


class BeneparComponent:
    """
    Berkeley Neural Parser (benepar) component for spaCy.

    Sample usage:
    >>> nlp = spacy.load('en_core_web_md')
    >>> if spacy.__version__.startswith('2'):
            nlp.add_pipe(BeneparComponent("benepar_en3"))
        else:
            nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    >>> doc = nlp("The quick brown fox jumps over the lazy dog.")
    >>> sent = list(doc.sents)[0]
    >>> print(sent._.parse_string)

    This component is only responsible for constituency parsing and (for some
    trained models) part-of-speech tagging. It should be preceded in the
    pipeline by other components that can, at minimum, perform tokenization and
    sentence segmentation.
    """

    name = "benepar"

    def __init__(
        self,
        name,
        subbatch_max_tokens=500,
        disable_tagger=False,
        batch_size="ignored",
    ):
        """Load a trained parser model.

        Args:
            name (str): Model name, or path to pytorch saved model
            subbatch_max_tokens (int): Maximum number of tokens to process in
                each batch
            disable_tagger (bool, default False): Unless disabled, the parser
                will set predicted part-of-speech tags for the document,
                overwriting any existing tags provided by spaCy models or
                previous pipeline steps. This option has no effect for parser
                models that do not have a part-of-speech tagger built in.
            batch_size: deprecated and ignored; use subbatch_max_tokens instead
        """
        self._parser = load_trained_model(name)
        if torch.cuda.is_available():
            self._parser.cuda()

        self.subbatch_max_tokens = subbatch_max_tokens
        self.disable_tagger = disable_tagger

        self._label_vocab = self._parser.config["label_vocab"]
        label_vocab_size = max(self._label_vocab.values()) + 1
        self._label_from_index = [()] * label_vocab_size
        for label, i in self._label_vocab.items():
            if label:
                self._label_from_index[i] = tuple(label.split("::"))
            else:
                self._label_from_index[i] = ()
        self._label_from_index = tuple(self._label_from_index)

        if not self.disable_tagger:
            tag_vocab = self._parser.config["tag_vocab"]
            tag_vocab_size = max(tag_vocab.values()) + 1
            self._tag_from_index = [()] * tag_vocab_size
            for tag, i in tag_vocab.items():
                self._tag_from_index[i] = tag
            self._tag_from_index = tuple(self._tag_from_index)
        else:
            self._tag_from_index = None

    def __call__(self, doc):
        """Update the input document with predicted constituency parses."""
        # TODO(https://github.com/nikitakit/self-attentive-parser/issues/16): handle
        # tokens that consist entirely of whitespace.
        constituent_data = PartialConstituentData()
        wrapped_sents = [SentenceWrapper(sent) for sent in doc.sents]
        for sent, parse in zip(
            doc.sents,
            self._parser.parse(
                wrapped_sents,
                return_compressed=True,
                subbatch_max_tokens=self.subbatch_max_tokens,
            ),
        ):
            constituent_data.starts.append(parse.starts + sent.start)
            constituent_data.ends.append(parse.ends + sent.start)
            constituent_data.labels.append(parse.labels)

            if parse.tags is not None and not self.disable_tagger:
                for i, tag_id in enumerate(parse.tags):
                    sent[i].tag_ = self._tag_from_index[tag_id]

        doc._._constituent_data = constituent_data.finalize(doc, self._label_from_index)
        return doc


def create_benepar_component(
    nlp,
    name,
    model: str,
    subbatch_max_tokens: int,
    disable_tagger: bool,
):
    return BeneparComponent(
        model,
        subbatch_max_tokens=subbatch_max_tokens,
        disable_tagger=disable_tagger,
    )


def register_benepar_component_factory():
    # Starting with spaCy 3.0, nlp.add_pipe no longer directly accepts
    # BeneparComponent instances. We must instead register a component factory.
    import spacy

    if spacy.__version__.startswith("2"):
        return

    from spacy.language import Language

    Language.factory(
        "benepar",
        default_config={
            "subbatch_max_tokens": 500,
            "disable_tagger": False,
        },
        func=create_benepar_component,
    )


try:
    register_benepar_component_factory()
except ImportError:
    pass
