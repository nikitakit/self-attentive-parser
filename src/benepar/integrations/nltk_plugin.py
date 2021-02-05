import dataclasses
import itertools
from typing import List, Optional, Tuple

import nltk
import torch

from .downloader import load_trained_model
from ..parse_base import BaseParser, BaseInputExample
from ..ptb_unescape import ptb_unescape, guess_space_after


TOKENIZER_LOOKUP = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "pl": "polish",
    "sv": "swedish",
}

LANGUAGE_GUESS = {
    "ar": ("X", "XP", "WHADVP", "WHNP", "WHPP"),
    "zh": ("VSB", "VRD", "VPT", "VNV"),
    "en": ("WHNP", "WHADJP", "SINV", "SQ"),
    "de": ("AA", "AP", "CCP", "CH", "CNP", "VZ"),
    "fr": ("P+", "P+D+", "PRO+", "PROREL+"),
    "he": ("PREDP", "SYN_REL", "SYN_yyDOT"),
    "pl": ("formaczas", "znakkonca"),
    "sv": ("PSEUDO", "AVP", "XP"),
}


def guess_language(label_vocab):
    """Guess parser language based on its syntactic label inventory.

    The parser training scripts are designed to accept arbitrary input tree
    files with minimal language-specific behavior, but at inference time we may
    need to know the language identity in order to invoke other pipeline
    elements, such as tokenizers.
    """
    for language, required_labels in LANGUAGE_GUESS.items():
        if all(label in label_vocab for label in required_labels):
            return language
    return None


@dataclasses.dataclass
class InputSentence(BaseInputExample):
    """Parser input for a single sentence.

    At least one of `words` and `escaped_words` is required for each input
    sentence. The remaining fields are optional: the parser will attempt to
    derive the value for any missing fields using the fields that are provided.

    `words` and `space_after` together form a reversible tokenization of the
    input text: they represent, respectively, the Unicode text for each word and
    an indicator for whether the word is followed by whitespace. These are used
    as inputs by the parser.

    `tags` is a list of part-of-speech tags, if available prior to running the
    parser. The parser does not actually use these tags as input, but it will
    pass them through to its output. If `tags` is None, the parser will perform
    its own part of speech tagging (if the parser was not trained to also do
    tagging, "UNK" part-of-speech tags will be used in the output instead).

    `escaped_words` are the representations of each leaf to use in the output
    tree. If `words` is provided, `escaped_words` will not be used by the neural
    network portion of the parser, and will only be incorporated when
    constructing the output tree. Therefore, `escaped_words` may be used to
    accommodate any dataset-specific text encoding, such as transliteration.

    Here is an example of the differences between these fields for English PTB:
        (raw text):     "Fly safely."
        words:          "       Fly     safely  .       "
        space_after:    False   True    False   False   False
        tags:           ``      VB      RB      .       ''
        escaped_words:  ``      Fly     safely  .       ''
    """

    words: Optional[List[str]] = None
    space_after: Optional[List[bool]] = None
    tags: Optional[List[str]] = None
    escaped_words: Optional[List[str]] = None

    @property
    def tree(self):
        return None

    def leaves(self):
        return self.escaped_words

    def pos(self):
        if self.tags is not None:
            return list(zip(self.escaped_words, self.tags))
        else:
            return [(word, "UNK") for word in self.escaped_words]


class Parser:
    """Berkeley Neural Parser (benepar), integrated with NLTK.

    Use this class to apply the Berkeley Neural Parser to pre-tokenized datasets
    and treebanks, or when integrating the parser into an NLP pipeline that
    already performs tokenization, sentence splitting, and (optionally)
    part-of-speech tagging. For parsing starting with raw text, it is strongly
    encouraged that you use spaCy and benepar.BeneparComponent instead.

    Sample usage:
    >>> parser = benepar.Parser("benepar_en3")
    >>> input_sentence = benepar.InputSentence(
        words=['"', 'Fly', 'safely', '.', '"'],
        space_after=[False, True, False, False, False],
        tags=['``', 'VB', 'RB', '.', "''"],
        escaped_words=['``', 'Fly', 'safely', '.', "''"],
    )
    >>> parser.parse(input_sentence)

    Not all fields of benepar.InputSentence are required, but at least one of
    `words` and `escaped_words` must not be None. The parser will attempt to
    guess the value for missing fields. For example,
    >>> input_sentence = benepar.InputSentence(
        words=['"', 'Fly', 'safely', '.', '"'],
    )
    >>> parser.parse(input_sentence)

    Although this class is primarily designed for use with data that has already
    been tokenized, to help with interactive use and debugging it also accepts
    simple text string inputs. However, using this class to parse from raw text
    is STRONGLY DISCOURAGED for any application where parsing accuracy matters.
    When parsing from raw text, use spaCy and benepar.BeneparComponent instead.
    The reason is that parser models do not ship with a tokenizer or sentence
    splitter, and some models may not include a part-of-speech tagger either. A
    toolkit must be used to fill in these pipeline components, and spaCy
    outperforms NLTK in all of these areas (sometimes by a large margin).
    >>> parser.parse('"Fly safely."')  # For debugging/interactive use only.
    """

    def __init__(self, name, batch_size=64, language_code=None):
        """Load a trained parser model.

        Args:
            name (str): Model name, or path to pytorch saved model
            batch_size (int): Maximum number of sentences to process per batch
            language_code (str, optional): language code for the parser (e.g.
                'en', 'he', 'zh', etc). Our official trained models will set
                this automatically, so this argument is only needed if training
                on new languages or treebanks.
        """
        self._parser = load_trained_model(name)
        if torch.cuda.is_available():
            self._parser.cuda()
        if language_code is not None:
            self._language_code = language_code
        else:
            self._language_code = guess_language(self._parser.config["label_vocab"])
        self._tokenizer_lang = TOKENIZER_LOOKUP.get(self._language_code, None)

        self.batch_size = batch_size

    def parse(self, sentence):
        """Parse a single sentence

        Args:
            sentence (InputSentence or List[str] or str): Sentence to parse.
                If the input is of List[str], it is assumed to be a sequence of
                words and will behave the same as only setting the `words` field
                of InputSentence. If the input is of type str, the sentence will
                be tokenized using the default NLTK tokenizer (not recommended:
                if parsing from raw text, use spaCy and benepar.BeneparComponent
                instead).

        Returns:
            nltk.Tree
        """
        return list(self.parse_sents([sentence]))[0]

    def parse_sents(self, sents):
        """Parse multiple sentences in batches.

        Args:
            sents (Iterable[InputSentence]): An iterable of sentences to be
                parsed. `sents` may also be a string, in which case it will be
                segmented into sentences using the default NLTK sentence
                splitter (not recommended: if parsing from raw text, use spaCy
                and benepar.BeneparComponent instead). Otherwise, each element
                of `sents` will be treated as a sentence. The elements of
                `sents` may also be List[str] or str: see Parser.parse() for
                documentation regarding these cases.

        Yields:
            nltk.Tree objects, one per input sentence.
        """
        if isinstance(sents, str):
            if self._tokenizer_lang is None:
                raise ValueError(
                    "No tokenizer available for this language. "
                    "Please split into individual sentences and tokens "
                    "before calling the parser."
                )
            sents = nltk.sent_tokenize(sents, self._tokenizer_lang)

        end_sentinel = object()
        for batch_sents in itertools.zip_longest(
            *([iter(sents)] * self.batch_size), fillvalue=end_sentinel
        ):
            batch_inputs = []
            for sent in batch_sents:
                if sent is end_sentinel:
                    break
                elif isinstance(sent, str):
                    if self._tokenizer_lang is None:
                        raise ValueError(
                            "No word tokenizer available for this language. "
                            "Please tokenize before calling the parser."
                        )
                    escaped_words = nltk.word_tokenize(sent, self._tokenizer_lang)
                    sent = InputSentence(escaped_words=escaped_words)
                elif isinstance(sent, (list, tuple)):
                    sent = InputSentence(words=sent)
                elif not isinstance(sent, InputSentence):
                    raise ValueError(
                        "Sentences must be one of: InputSentence, list, tuple, or str"
                    )
                batch_inputs.append(self._with_missing_fields_filled(sent))

            for inp, output in zip(
                batch_inputs, self._parser.parse(batch_inputs, return_compressed=True)
            ):
                # If pos tags are provided as input, ignore any tags predicted
                # by the parser.
                if inp.tags is not None:
                    output = output.without_predicted_tags()
                yield output.to_tree(
                    inp.pos(),
                    self._parser.decoder.label_from_index,
                    self._parser.tag_from_index,
                )

    def _with_missing_fields_filled(self, sent):
        if not isinstance(sent, InputSentence):
            raise ValueError("Input is not an instance of InputSentence")
        if sent.words is None and sent.escaped_words is None:
            raise ValueError("At least one of words or escaped_words is required")
        elif sent.words is None:
            sent = dataclasses.replace(sent, words=ptb_unescape(sent.escaped_words))
        elif sent.escaped_words is None:
            escaped_words = [
                word.replace("(", "-LRB-")
                .replace(")", "-RRB-")
                .replace("{", "-LCB-")
                .replace("}", "-RCB-")
                .replace("[", "-LSB-")
                .replace("]", "-RSB-")
                for word in sent.words
            ]
            sent = dataclasses.replace(sent, escaped_words=escaped_words)
        else:
            if len(sent.words) != len(sent.escaped_words):
                raise ValueError(
                    f"Length of words ({len(sent.words)}) does not match "
                    f"escaped_words ({len(sent.escaped_words)})"
                )

        if sent.space_after is None:
            if self._language_code == "zh":
                space_after = [False for _ in sent.words]
            elif self._language_code in ("ar", "he"):
                space_after = [True for _ in sent.words]
            else:
                space_after = guess_space_after(sent.words)
            sent = dataclasses.replace(sent, space_after=space_after)
        elif len(sent.words) != len(sent.space_after):
            raise ValueError(
                f"Length of words ({len(sent.words)}) does not match "
                f"space_after ({len(sent.space_after)})"
            )

        assert len(sent.words) == len(sent.escaped_words) == len(sent.space_after)
        return sent
