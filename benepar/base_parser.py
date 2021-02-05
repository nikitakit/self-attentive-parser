try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
import numpy as np
import os
import sys
import json
import codecs

from . import chart_decoder
from .downloader import load_model
from .bert_tokenization import BertTokenizer

#%%

IS_PY2 = sys.version_info < (3,0)

if IS_PY2:
    STRING_TYPES = (str, unicode)
else:
    STRING_TYPES = (str,)

ELMO_START_SENTENCE = 256
ELMO_STOP_SENTENCE = 257
ELMO_START_WORD = 258
ELMO_STOP_WORD = 259
ELMO_CHAR_PAD = 260

PTB_TOKEN_ESCAPE = {u"(": u"-LRB-",
    u")": u"-RRB-",
    u"{": u"-LCB-",
    u"}": u"-RCB-",
    u"[": u"-LSB-",
    u"]": u"-RSB-"}

BERT_TOKEN_MAPPING = {u"-LRB-": u"(",
    u"-RRB-": u")",
    u"-LCB-": u"{",
    u"-RCB-": u"}",
    u"-LSB-": u"[",
    u"-RSB-": u"]",
    u"``": u'"',
    u"''": u'"',
    u"`": u"'",
    u"\u201c": u'"',
    u"\u201d": u'"',
    u"\u2018": u"'",
    u"\u2019": u"'",
    u"\xab": u'"',
    u"\xbb": u'"',
    u"\u201e": u'"',
    u"\u2039": u"'",
    u"\u203a": u"'",
    u"\u2013": u"--", # en dash
    u"\u2014": u"--", # em dash
    }

# Label vocab is made immutable because it is potentially exposed to users
# through the spacy plugin
LABEL_VOCAB = ((),
 ('S',),
 ('PP',),
 ('NP',),
 ('PRN',),
 ('VP',),
 ('ADVP',),
 ('SBAR', 'S'),
 ('ADJP',),
 ('QP',),
 ('UCP',),
 ('S', 'VP'),
 ('SBAR',),
 ('WHNP',),
 ('SINV',),
 ('FRAG',),
 ('NAC',),
 ('WHADVP',),
 ('NP', 'QP'),
 ('PRT',),
 ('S', 'PP'),
 ('S', 'NP'),
 ('NX',),
 ('S', 'ADJP'),
 ('WHPP',),
 ('SBAR', 'S', 'VP'),
 ('SBAR', 'SINV'),
 ('SQ',),
 ('NP', 'NP'),
 ('SBARQ',),
 ('SQ', 'VP'),
 ('CONJP',),
 ('ADJP', 'QP'),
 ('FRAG', 'NP'),
 ('FRAG', 'ADJP'),
 ('WHADJP',),
 ('ADJP', 'ADJP'),
 ('FRAG', 'PP'),
 ('S', 'ADVP'),
 ('FRAG', 'SBAR'),
 ('PRN', 'S'),
 ('PRN', 'S', 'VP'),
 ('INTJ',),
 ('X',),
 ('NP', 'NP', 'NP'),
 ('FRAG', 'S', 'VP'),
 ('ADVP', 'ADVP'),
 ('RRC',),
 ('VP', 'PP'),
 ('VP', 'VP'),
 ('SBAR', 'FRAG'),
 ('ADVP', 'ADJP'),
 ('LST',),
 ('NP', 'NP', 'QP'),
 ('PRN', 'SBAR'),
 ('VP', 'S', 'VP'),
 ('S', 'UCP'),
 ('FRAG', 'WHNP'),
 ('NP', 'PP'),
 ('NP', 'SBAR', 'S', 'VP'),
 ('WHNP', 'QP'),
 ('VP', 'FRAG', 'ADJP'),
 ('FRAG', 'WHADVP'),
 ('NP', 'ADJP'),
 ('VP', 'SBAR'),
 ('NP', 'S', 'VP'),
 ('X', 'PP'),
 ('S', 'VP', 'VP'),
 ('S', 'VP', 'ADVP'),
 ('WHNP', 'WHNP'),
 ('NX', 'NX'),
 ('FRAG', 'ADVP'),
 ('FRAG', 'VP'),
 ('VP', 'ADVP'),
 ('SBAR', 'WHNP'),
 ('FRAG', 'SBARQ'),
 ('PP', 'PP'),
 ('PRN', 'PP'),
 ('VP', 'NP'),
 ('X', 'NP'),
 ('PRN', 'SINV'),
 ('NP', 'SBAR'),
 ('PP', 'NP'),
 ('NP', 'INTJ'),
 ('FRAG', 'INTJ'),
 ('X', 'VP'),
 ('PRN', 'NP'),
 ('FRAG', 'UCP'),
 ('NP', 'ADVP'),
 ('SBAR', 'SBARQ'),
 ('SBAR', 'SBAR', 'S'),
 ('SBARQ', 'WHADVP'),
 ('ADVP', 'PRT'),
 ('UCP', 'ADJP'),
 ('PRN', 'FRAG', 'WHADJP'),
 ('FRAG', 'S'),
 ('S', 'S'),
 ('FRAG', 'S', 'ADJP'),
 ('INTJ', 'S'),
 ('ADJP', 'NP'),
 ('X', 'ADVP'),
 ('FRAG', 'WHPP'),
 ('NP', 'FRAG'),
 ('NX', 'QP'),
 ('NP', 'S'),
 ('SBAR', 'WHADVP'),
 ('X', 'SBARQ'),
 ('NP', 'PRN'),
 ('NX', 'S', 'VP'),
 ('NX', 'S'),
 ('UCP', 'PP'),
 ('RRC', 'VP'),
 ('ADJP', 'ADVP'))

SENTENCE_MAX_LEN = 300
BERT_MAX_LEN = 512

#%%
class BaseParser(object):
    def __init__(self, name, batch_size=64):
        self._graph = tf.Graph()

        with self._graph.as_default():
            if isinstance(name, STRING_TYPES) and '/' not in name:
                model = load_model(name)
            elif not os.path.exists(name):
                raise Exception("Argument is neither a valid module name nor a path to an existing file/folder: {}".format(name))
            else:
                if not os.path.isdir(name):
                    with open(name, 'rb') as f:
                        model = f.read()
                else:
                    model = {}
                    with open(os.path.join(name, 'meta.json')) as f:
                        model['meta'] = json.load(f)
                    with open(os.path.join(name, 'model.pb'), 'rb') as f:
                        model['model'] = f.read()
                    with codecs.open(os.path.join(name, 'vocab.txt'), encoding='utf-8') as f:
                        model['vocab'] = f.read()

            if isinstance(model, dict):
                graph_def = tf.GraphDef.FromString(model['model'])
            else:
                graph_def = tf.GraphDef.FromString(model)
            tf.import_graph_def(graph_def, name='')

        self._sess = tf.Session(graph=self._graph)
        if not isinstance(model, dict):
            # Older model format (for ELMo-based models)
            self._chars = self._graph.get_tensor_by_name('chars:0')
            self._charts = self._graph.get_tensor_by_name('charts:0')
            self._label_vocab = LABEL_VOCAB
            self._language_code = 'en'
            self._provides_tags = False
            self._make_feed_dict = self._make_feed_dict_elmo
        else:
            # Newer model format (for BERT-based models)
            meta = model['meta']
            # Label vocab is made immutable because it is potentially exposed to
            # users through the spacy plugin
            self._label_vocab = tuple([tuple(label) for label in meta['label_vocab']])
            self._language_code = meta['language_code']
            self._provides_tags = meta['provides_tags']

            self._input_ids = self._graph.get_tensor_by_name('input_ids:0')
            self._word_end_mask = self._graph.get_tensor_by_name('word_end_mask:0')
            self._charts = self._graph.get_tensor_by_name('charts:0')
            if self._provides_tags:
                self._tag_vocab = meta['tag_vocab']
                self._tags = self._graph.get_tensor_by_name('tags:0')

            self._bert_tokenizer = BertTokenizer(
                model['vocab'], do_lower_case=meta['bert_do_lower_case'])
            self._make_feed_dict = self._make_feed_dict_bert

        self.batch_size = batch_size

    def _make_feed_dict_elmo(self, sentences):
        padded_len = max([len(sentence) + 2 for sentence in sentences])
        if padded_len > SENTENCE_MAX_LEN:
            raise ValueError("Sentence of length {} exceeds the maximum supported length of {}".format(
                padded_len - 2, SENTENCE_MAX_LEN - 2))

        all_chars = np.zeros((len(sentences), padded_len, 50), dtype=np.int32)

        for snum, sentence in enumerate(sentences):
            all_chars[snum, :len(sentence)+2,:] = ELMO_CHAR_PAD

            all_chars[snum, 0, 0] = ELMO_START_WORD
            all_chars[snum, 0, 1] = ELMO_START_SENTENCE
            all_chars[snum, 0, 2] = ELMO_STOP_WORD

            for i, word in enumerate(sentence):
                word = PTB_TOKEN_ESCAPE.get(word, word)
                if IS_PY2:
                    chars = [ELMO_START_WORD] + [ord(char) for char in word.encode('utf-8', 'ignore')[:(50-2)]] + [ELMO_STOP_WORD]
                else:
                    chars = [ELMO_START_WORD] + list(word.encode('utf-8', 'ignore')[:(50-2)]) + [ELMO_STOP_WORD]
                all_chars[snum, i+1, :len(chars)] = chars

            all_chars[snum, len(sentence)+1, 0] = ELMO_START_WORD
            all_chars[snum, len(sentence)+1, 1] = ELMO_STOP_SENTENCE
            all_chars[snum, len(sentence)+1, 2] = ELMO_STOP_WORD

            # Add 1; 0 is a reserved value for signaling words past the end of the
            # sentence, which we don't have because batch_size=1
            all_chars[snum, :len(sentence)+2,:] += 1

        return {self._chars: all_chars}

    def _make_feed_dict_bert(self, sentences):
        all_input_ids = np.zeros((len(sentences), BERT_MAX_LEN), dtype=int)
        all_word_end_mask = np.zeros((len(sentences), BERT_MAX_LEN), dtype=int)

        subword_max_len = 0
        for snum, sentence in enumerate(sentences):
            tokens = []
            word_end_mask = []

            tokens.append(u"[CLS]")
            word_end_mask.append(1)

            cleaned_words = []
            for word in sentence:
                word = BERT_TOKEN_MAPPING.get(word, word)
                # BERT is pre-trained with a tokenizer that doesn't split off
                # n't as its own token
                if word == u"n't" and cleaned_words:
                    cleaned_words[-1] = cleaned_words[-1] + u"n"
                    word = u"'t"
                cleaned_words.append(word)

            for word in cleaned_words:
                word_tokens = self._bert_tokenizer.tokenize(word)
                if not word_tokens:
                    # The tokenizer used in conjunction with the parser may not
                    # align with BERT; in particular spaCy will create separate
                    # tokens for whitespace when there is more than one space in
                    # a row, and will sometimes separate out characters of
                    # unicode category Mn (which BERT strips when do_lower_case
                    # is enabled). Substituting UNK is not strictly correct, but
                    # it's better than failing to return a valid parse.
                    word_tokens = ["[UNK]"]
                for _ in range(len(word_tokens)):
                    word_end_mask.append(0)
                word_end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append(u"[SEP]")
            word_end_mask.append(1)

            input_ids = self._bert_tokenizer.convert_tokens_to_ids(tokens)
            if len(sentence) + 2 > SENTENCE_MAX_LEN or len(input_ids) > BERT_MAX_LEN:
                raise ValueError("Sentence of length {} is too long to be parsed".format(
                    len(sentence)))

            subword_max_len = max(subword_max_len, len(input_ids))

            all_input_ids[snum, :len(input_ids)] = input_ids
            all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

        all_input_ids = all_input_ids[:, :subword_max_len]
        all_word_end_mask = all_word_end_mask[:, :subword_max_len]

        return {
            self._input_ids: all_input_ids,
            self._word_end_mask: all_word_end_mask
        }

    def _make_charts_and_tags(self, sentences):
        feed_dict = self._make_feed_dict(sentences)
        if self._provides_tags:
            charts_val, tags_val = self._sess.run((self._charts, self._tags), feed_dict)
        else:
            charts_val = self._sess.run(self._charts, feed_dict)
        for snum, sentence in enumerate(sentences):
            chart_size = len(sentence) + 1
            chart = charts_val[snum,:chart_size,:chart_size,:]
            if self._provides_tags:
                tags = tags_val[snum,1:chart_size]
            else:
                tags = None
            yield chart, tags

    def _batched_parsed_raw(self, sentence_data_pairs):
        batch_sentences = []
        batch_data = []
        for sentence, datum in sentence_data_pairs:
            batch_sentences.append(sentence)
            batch_data.append(datum)
            if len(batch_sentences) >= self.batch_size:
                for (chart_np, tags_np), datum in zip(self._make_charts_and_tags(batch_sentences), batch_data):
                    yield chart_decoder.decode(chart_np), tags_np, datum
                batch_sentences = []
                batch_data = []
        if batch_sentences:
            for (chart_np, tags_np), datum in zip(self._make_charts_and_tags(batch_sentences), batch_data):
                yield chart_decoder.decode(chart_np), tags_np, datum
