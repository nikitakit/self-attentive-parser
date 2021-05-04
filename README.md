# Berkeley Neural Parser

A high-accuracy parser with models for 11 languages, implemented in Python. Based on [Constituency Parsing with a Self-Attentive Encoder](https://arxiv.org/abs/1805.01052) from ACL 2018, with additional changes described in [Multilingual Constituency Parsing with Self-Attention and Pre-Training](https://arxiv.org/abs/1812.11760).

**New February 2021:** Version 0.2.0 of the Berkeley Neural Parser is now out, with higher-quality pre-trained models for all languages. Inference now uses PyTorch instead of TensorFlow (training has always been PyTorch-only). Drops support for Python 2.7 and 3.5. Includes updated support for training and using your own parsers, based on your choice of [pre-trained model](https://huggingface.co/models).

## Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Available Models](#available-models)
4. [Training](#training)
5. [Reproducing Experiments](#reproducing-experiments)
6. [Citation](#citation)
7. [Credits](#credits)

If you are primarily interested in training your own parsing models, skip to the [Training](#training) section of this README.

## Installation

To install the parser, run the command:
```bash
$ pip install benepar
```
*Note: benepar 0.2.0 is a major upgrade over the previous version, and comes with entirely new and higher-quality parser models. If you are not ready to upgrade, you can pin your benepar version to [the previous release (0.1.3)](https://github.com/nikitakit/self-attentive-parser/tree/acl2019).*

Python 3.6 (or newer) and [PyTorch](https://pytorch.org/) 1.6 (or newer) are required. See the PyTorch website for instruction on how to select between GPU-enabled and CPU-only versions of PyTorch; benepar will automatically use the GPU if it is available to pytorch.

The recommended way of using benepar is through integration with [spaCy](https://spacy.io/). If using spaCy, you should install a spaCy model for your language. For English, the installation command is:
```sh
$ python -m spacy download en_core_web_md
```

The spaCy model is only used for tokenization and sentence segmentation. If language-specific analysis beyond parsing is not required, you may also forego a language-specific model and instead use a multi-language model that only performs tokenization and segmentation. [One such model](https://spacy.io/models/xx#xx_sent_ud_sm), newly added in spaCy 3.0, should work for English, German, Korean, Polish, and Swedish (but not Chinese, since it doesn't seem to support Chinese word segmentation).

Parsing models need to be downloaded separately, using the commands:
```python
>>> import benepar
>>> benepar.download('benepar_en3')
```

See the [Available Models](#available-models) section below for a full list of models.

## Usage

### Usage with spaCy (recommended)

The recommended way of using benepar is through its integration with spaCy:
```python
>>> import benepar, spacy
>>> nlp = spacy.load('en_core_web_md')
>>> if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
>>> doc = nlp("The time for action is now. It's never too late to do something.")
>>> sent = list(doc.sents)[0]
>>> print(sent._.parse_string)
(S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))
>>> sent._.labels
('S',)
>>> list(sent._.children)[0]
The time for action
```

Since spaCy does not provide an official constituency parsing API, all methods are accessible through the extension namespaces `Span._` and `Token._`.

The following extension properties are available:
- `Span._.labels`: a tuple of labels for the given span. A span may have multiple labels when there are unary chains in the parse tree.
- `Span._.parse_string`: a string representation of the parse tree for a given span.
- `Span._.constituents`: an iterator over `Span` objects for sub-constituents in a pre-order traversal of the parse tree.
- `Span._.parent`: the parent `Span` in the parse tree.
- `Span._.children`: an iterator over child `Span`s in the parse tree.
- `Token._.labels`, `Token._.parse_string`, `Token._.parent`: these behave the same as calling the corresponding method on the length-one Span containing the token.

These methods will raise an exception when called on a span that is not a constituent in the parse tree. Such errors can be avoided by traversing the parse tree starting at either sentence level (by iterating over `doc.sents`) or with an individual `Token` object.

### Usage with NLTK

There is also an NLTK interface, which is designed for use with pre-tokenized datasets and treebanks, or when integrating the parser into an NLP pipeline that already performs (at minimum) tokenization and sentence splitting. For parsing starting with raw text, it is **strongly encouraged** that you use spaCy and `benepar.BeneparComponent` instead.

Sample usage with NLTK:
```python
>>> import benepar
>>> parser = benepar.Parser("benepar_en3")
>>> input_sentence = benepar.InputSentence(
    words=['"', 'Fly', 'safely', '.', '"'],
    space_after=[False, True, False, False, False],
    tags=['``', 'VB', 'RB', '.', "''"],
    escaped_words=['``', 'Fly', 'safely', '.', "''"],
)
>>> tree = parser.parse(input_sentence)
>>> print(tree)
(TOP (S (`` ``) (VP (VB Fly) (ADVP (RB safely))) (. .) ('' '')))
```

Not all fields of `benepar.InputSentence` are required, but at least one of `words` and `escaped_words` must be specified. The parser will attempt to guess the value for missing fields, for example:
```python
>>> input_sentence = benepar.InputSentence(
    words=['"', 'Fly', 'safely', '.', '"'],
)
>>> parser.parse(input_sentence)
```

Use `parse_sents` to parse multiple sentences.
```python
>>> input_sentence1 = benepar.InputSentence(
    words=['The', 'time', 'for', 'action', 'is', 'now', '.'],
)
>>> input_sentence2 = benepar.InputSentence(
    words=['It', "'s", 'never', 'too', 'late', 'to', 'do', 'something', '.'],
)
>>> parser.parse_sents([input_sentence1, input_sentence2])
```

Some parser models also allow Unicode text input for debugging/interactive use, but passing in raw text strings is *strongly discouraged* for any application where parsing accuracy matters.
```python
>>> parser.parse('"Fly safely."')  # For debugging/interactive use only.
```
When parsing from raw text, we recommend using spaCy and `benepar.BeneparComponent` instead. The reason is that parser models do not ship with a tokenizer or sentence splitter, and some models may not include a part-of-speech tagger either. A toolkit must be used to fill in these pipeline components, and spaCy outperforms NLTK in all of these areas (sometimes by a large margin). 



## Available Models

The following trained parser models are available. To use spaCy integration, you will also need to install a [spaCy model for the appropriate language](https://spacy.io/models).

Model       | Language | Info
----------- | -------- | ----
`benepar_en3` | English | 95.40 F1 on [revised](https://catalog.ldc.upenn.edu/LDC2015T13) WSJ test set. The training data uses revised tokenization and syntactic annotation based on the same guidelines as the English Web Treebank and OntoNotes, which better matches modern tokenization practices in libraries like spaCy. Based on T5-small.
`benepar_en3_large` | English | 96.29 F1 on [revised](https://catalog.ldc.upenn.edu/LDC2015T13) WSJ test set. The training data uses revised tokenization and syntactic annotation based on the same guidelines as the English Web Treebank and OntoNotes, which better matches modern tokenization practices in libraries like spaCy. Based on T5-large.
`benepar_zh2` | Chinese | 92.56 F1 on CTB 5.1 test set. Usage with spaCy allows supports parsing from raw text, but the NLTK API only supports parsing previously tokenized sentences. Based on Chinese ELECTRA-180G-large.
`benepar_ar2` | Arabic | 90.52 F1 on SPMRL2013/2014 test set. Only supports using the NLTK API for parsing previously tokenized sentences. Parsing from raw text and spaCy integration are not supported. Based on XLM-R.
`benepar_de2` | German | 92.10 F1 on SPMRL2013/2014 test set. Based on XLM-R.
`benepar_eu2` | Basque | 93.36 F1 on SPMRL2013/2014 test set. Usage with spaCy first requires implementing Basque support in spaCy. Based on XLM-R.
`benepar_fr2` | French | 88.43 F1 on SPMRL2013/2014 test set. Based on XLM-R.
`benepar_he2` | Hebrew | 93.98 F1 on SPMRL2013/2014 test set. Only supports using the NLTK API for parsing previously tokenized sentences. Parsing from raw text and spaCy integration are not supported. Based on XLM-R.
`benepar_hu2` | Hungarian | 96.19 F1 on SPMRL2013/2014 test set. Usage with spaCy requires a [Hungarian model for spaCy](https://github.com/oroszgy/spacy-hungarian-models). The NLTK API only supports parsing previously tokenized sentences. Based on XLM-R.
`benepar_ko2` | Korean | 91.72 F1 on SPMRL2013/2014 test set. Can be used with spaCy's [multi-language sentence segmentation model](https://spacy.io/models/xx#xx_sent_ud_sm) (requires spaCy v3.0). The NLTK API only supports parsing previously tokenized sentences. Based on XLM-R.
`benepar_pl2` | Polish | 97.15 F1 on SPMRL2013/2014 test set. Based on XLM-R.
`benepar_sv2` | Swedish | 92.21 F1 on SPMRL2013/2014 test set. Can be used with spaCy's [multi-language sentence segmentation model](https://spacy.io/models/xx#xx_sent_ud_sm) (requires spaCy v3.0). Based on XLM-R.
`benepar_en3_wsj` | English | **Consider using `benepar_en3` or `benepar_en3_large` instead**. 95.55 F1 on [canonical](https://catalog.ldc.upenn.edu/LDC99T42) WSJ test set used for decades of English constituency parsing publications. Based on BERT-large-uncased. We believe that the revised annotation guidelines used for training `benepar_en3`/`benepar_en3_large` are more suitable for downstream use because they better handle language usage in web text, and are more consistent with modern practices in dependency parsing and libraries like spaCy. Nevertheless, we provide the `benepar_en3_wsj` model for cases where using the revised treebanking conventions are not appropriate, such as benchmarking different models on the same dataset.

## Training

Training requires cloning this repository from GitHub. While the model code in `src/benepar` is distributed in the `benepar` package on PyPI, the training and evaluation scripts directly under `src/` are not.

#### Software Requirements for Training
* Python 3.7 or higher.
* [PyTorch](http://pytorch.org/) 1.6.0, or any compatible version.
* All dependencies required by the `benepar` package, including: [NLTK](https://www.nltk.org/) 3.2, [torch-struct](https://github.com/harvardnlp/pytorch-struct) 0.4, [transformers](https://github.com/huggingface/transformers) 4.3.0, or compatible.
* [pytokenizations](https://github.com/tamuhey/tokenizations/) 0.7.2 or compatible.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. If training on the SPMRL datasets, you will need to run `make` inside the `EVALB_SPMRL/` directory instead.

### Training Instructions

A new model can be trained using the command `python src/main.py train ...`. Some of the available arguments are:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--train-path` | Path to training trees | `data/wsj/train_02-21.LDC99T42`
`--train-path-text` | Optional non-destructive tokenization of the training data | Guess raw text; see `--text-processing`
`--dev-path` | Path to development trees | `data/wsj/dev_22.LDC99T42`
`--dev-path-text` | Optional non-destructive tokenization of the development data | Guess raw text; see `--text-processing`
`--text-processing` | Heuristics for guessing raw text from descructively tokenized tree files. See `load_trees()` in `src/treebanks.py` | Default rules for languages other than Arabic, Chinese, and Hebrew
`--subbatch-max-tokens` | Maximum number of tokens to process in parallel while training (a full batch may not fit in GPU memory) | 2000
`--parallelize` | Distribute pre-trained model (e.g. T5) layers across multiple GPUs. | Use at most one GPU
`--batch-size` | Number of examples per training update | 32
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--numpy-seed` | NumPy random seed | Random
`--use-pretrained` | Use pre-trained encoder | Do not use pre-trained encoder
`--pretrained-model` | Model to use if `--use-pretrained` is passed. May be a path or a model id from the [HuggingFace Model Hub](https://huggingface.co/models)| `bert-base-uncased`
`--predict-tags` | Adds a part-of-speech tagging component and auxiliary loss to the parser | Do not predict tags
`--use-chars-lstm` | Use learned CharLSTM word representations | Do not use CharLSTM
`--use-encoder` | Use learned transformer layers on top of pre-trained model or CharLSTM | Do not use extra transformer layers
`--num-layers` | Number of transformer layers to use if `--use-encoder` is passed | 8
`--encoder-max-len` | Maximum sentence length (in words) allowed for extra transformer layers | 512

Additional arguments are available for other hyperparameters; see `make_hparams()` in `src/main.py`. These can be specified on the command line, such as `--num-layers 2` (for numerical parameters), `--predict-tags` (for boolean parameters that default to False), or `--no-XXX` (for boolean parameters that default to True).

For each development evaluation, the F-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development F-score.

Prior to training the parser, you will first need to obtain appropriate training data. We provide [instructions on how to process standard datasets like PTB, CTB, and the SMPRL 2013/2014 Shared Task data](data/README.md). After following the instructions for the English WSJ data, you can use the following command to train an English parser using the default hyperparameters:

```
python src/main.py train --use-pretrained --model-path-base models/en_bert_base
```

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for more examples of good hyperparameter choices.

### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path` | Path of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-path` | Path to test trees | `data/23.auto.clean`
`--test-path-text` | Optional non-destructive tokenization of the test data | Guess raw text; see `--text-processing`
`--text-processing` | Heuristics for guessing raw text from descructively tokenized tree files. See `load_trees()` in `src/treebanks.py` | Default rules for languages other than Arabic, Chinese, and Hebrew
`--test-path-raw` | Alternative path to test trees that is used for evalb only (used to double-check that evaluation against pre-processed trees does not contain any bugs) | Compare to trees from `--test-path`
`--subbatch-max-tokens` | Maximum number of tokens to process in parallel (a GPU does not have enough memory to process the full dataset in one batch) | 500
`--parallelize` | Distribute pre-trained model (e.g. T5) layers across multiple GPUs. | Use at most one GPU
`--output-path` | Path to write predicted trees to (use `"-"` for stdout). | Do not save predicted trees
`--no-predict-tags` | Use gold part-of-speech tags when running EVALB. This is the standard for publications, and omitting this flag may give erroneously high F1 scores. | Use predicted part-of-speech tags for EVALB, if available

As an example, you can evaluate a trained model using the following command:
```
python src/main.py test --model-path models/en_bert_base_dev=*.pt
```

### Exporting Models for Inference

The `benepar` package can directly use saved checkpoints by replacing a model name like `benepar_en3` with a path such as `models/en_bert_base_dev_dev=95.67.pt`. However, releasing the single-file checkpoints has a few shortcomings:
* Single-file checkpoints do not include the tokenizer or pre-trained model config. These can generally be downloaded automatically from the HuggingFace model hub, but this requires an Internet connection and may also (incidentally and unnecessarily) download pre-trained weights from the HuggingFace Model Hub
* Single-file checkpoints are 3x larger than necessary, because they save optimizer state

Use `src/export.py` to convert a checkpoint file into a directory that encapsulates everything about a trained model. For example,
```
python src/export.py export \
  --model-path models/en_bert_base_dev=*.pt \
  --output-dir=models/en_bert_base
```

When exporting, there is also a `--compress` option that slightly adjusts model weights, so that the output directory can be compressed into a ZIP archive of much smaller size. We use this for our official model releases, because it's a hassle to distribute model weights that are 2GB+ in size. When using the `--compress` option, it is recommended to specify a test set in order to verify that compression indeed has minimal impact on parsing accuracy. Using the development data for verification is not recommended, since the development data was already used for the model selection criterion during training.
```
python src/export.py export \
  --model-path models/en_bert_base_dev=*.pt \
  --output-dir=models/en_bert_base \
  --test-path=data/wsj/test_23.LDC99T42
```

The `src/export.py` script also has a `test` subcommand that's roughly similar to `python src/main.py test`, except that it supports exported models and has slightly different flags. We can run the following command to verify that our English parser using BERT-large-uncased indeed achieves 95.55 F1 on the canonical WSJ test set:
```
python src/export.py test --model-path benepar_en3_wsj --test-path data/wsj/test_23.LDC99T42
```

## Reproducing Experiments

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for instructions on how to reproduce experiments reported in our ACL 2018 and 2019 papers.

## Citation

If you use this software for research, please cite our papers as follows:

```
@inproceedings{kitaev-etal-2019-multilingual,
    title = "Multilingual Constituency Parsing with Self-Attention and Pre-Training",
    author = "Kitaev, Nikita  and
      Cao, Steven  and
      Klein, Dan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1340",
    doi = "10.18653/v1/P19-1340",
    pages = "3499--3505",
}

@inproceedings{kitaev-klein-2018-constituency,
    title = "Constituency Parsing with a Self-Attentive Encoder",
    author = "Kitaev, Nikita  and
      Klein, Dan",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P18-1249",
    doi = "10.18653/v1/P18-1249",
    pages = "2676--2686",
}
```

## Credits

The code in this repository and portions of this README are based on https://github.com/mitchellstern/minimal-span-parser
