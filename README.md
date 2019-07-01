# Berkeley Neural Parser

A high-accuracy parser with models for 11 languages, implemented in Python. Based on [Constituency Parsing with a Self-Attentive Encoder](https://arxiv.org/abs/1805.01052) from ACL 2018, with additional changes described in [Multilingual Constituency Parsing with Self-Attention and Pre-Training](https://arxiv.org/abs/1812.11760).

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

To install the parser, run the commands:
```bash
$ pip install cython numpy
$ pip install benepar[cpu]
```

Cython and numpy should be installed separately prior to installing benepar. Note that `pip install benepar[cpu]` has a dependency on the `tensorflow` pip package, which is a CPU-only version of tensorflow. Use `pip install benepar[gpu]` to instead introduce a dependency on `tensorflow-gpu`. Installing a GPU-enabled version of TensorFlow will likely require additional steps; see the [official TensorFlow installation instructions](https://www.tensorflow.org/install/) for details.

Benepar integrates with one of two NLP libraries for Python: [NLTK](http://www.nltk.org/) or [spaCy](https://spacy.io/).

If using NLTK, you should install the NLTK sentence and word tokenizers:
```python
>>> import nltk
>>> nltk.download('punkt')
```

If using spaCy, you should install a spaCy model for your language. For English, the installation command is:
```sh
$ python -m spacy download en
```

Parsing models need to be downloaded separately, using the commands:
```python
>>> import benepar
>>> benepar.download('benepar_en2')
```

See the [Available Models](#available-models) section below for a full list of models.

## Usage
### Usage with NLTK

```python
>>> import benepar
>>> parser = benepar.Parser("benepar_en2")
>>> tree = parser.parse("Short cuts make long delays.")
>>> print(tree)
(S
  (NP (JJ Short) (NNS cuts))
  (VP (VBP make) (NP (JJ long) (NNS delays)))
  (. .))
```

Speed note: the first call to `parse` will take much longer that subsequent calls, as caches are being warmed up.

The parser can also parse pre-tokenized text. For some languages (including Chinese), this is required due to the lack of a built-in tokenizer.
```python
>>> parser.parse(['Short', 'cuts', 'make', 'long', 'delays', '.'])
```

Use `parse_sents` to parse multiple sentences. It accepts an entire document as a string, or a list of sentences.
```python
>>> parser.parse_sents("The time for action is now. It's never too late to do something.")
>>> parser.parse_sents(["The time for action is now.", "It's never too late to do something."])
>>> parser.parse_sents([['The', 'time', 'for', 'action', 'is', 'now', '.'], ['It', "'s", 'never', 'too', 'late', 'to', 'do', 'something', '.']])
```

All parse trees returned are represented using `nltk.Tree` objects.

### Usage with spaCy

Benepar also ships with a component that integrates with spaCy:
```python
>>> import spacy
>>> from benepar.spacy_plugin import BeneparComponent
>>> nlp = spacy.load('en')
>>> nlp.add_pipe(BeneparComponent("benepar_en2"))
>>> doc = nlp(u"The time for action is now. It's never too late to do something.")
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

## Available Models

The following trained parser models are available:

Model       | Language | Info
----------- | -------- | ----
`benepar_en2` | English | 95.17 F1 on WSJ test set, 94 MB on disk.
`benepar_en2_large` | English | 95.52 F1 on WSJ test set, 274 MB on disk. This model is up to 3x slower than `benepar_en2` when running on CPU; we recommend running it on a GPU instead.
`benepar_zh` | Chinese | 91.69 F1 on CTB 5.1 test set. Usage with NLTK requires tokenized sentences (untokenized raw text is not supported.) Use a package such as [jieba](https://github.com/fxsjy/jieba) for tokenization. Usage with spaCy first requires implementing Chinese support in spaCy. There is no official Chinese support in spaCy at the time of writing, but unofficial packages such as [this one](https://github.com/howl-anderson/Chinese_models_for_SpaCy) may work.
`benepar_ar` | Arabic | Usage with NLTK requires tokenized sentences (untokenized raw text is not supported.) Usage with spaCy first requires implementing Arabic support in spaCy. Accepts Unicode as input, but was trained on transliterated text (see `src/transliterate.py`); please let us know if there are any bugs.
`benepar_de` | German | Full support for NLTK and spaCy; use `python -m spacy download de` to download spaCy model for German.
`benepar_eu` | Basque | Usage with NLTK requires tokenized sentences (untokenized raw text is not supported.) Usage with spaCy first requires implementing Basque support in spaCy.
`benepar_fr` | French | Full support for NLTK and spaCy; use `python -m spacy download fr` to download spaCy model for French.
`benepar_he` | Hebrew | Usage with NLTK requires tokenized sentences (untokenized raw text is not supported.) Usage with spaCy first requires implementing Hebrew support in spaCy. Accepts Unicode as input, but was trained on transliterated text (see `src/transliterate.py`); please let us know if there are any bugs.
`benepar_hu` | Hungarian | Usage with NLTK requires tokenized sentences (untokenized raw text is not supported.) Usage with spaCy requires a [Hungarian model for spaCy](https://github.com/oroszgy/spacy-hungarian-models).
`benepar_ko` | Korean | Usage with NLTK requires tokenized sentences (untokenized raw text is not supported.) Usage with spaCy first requires implementing Korean support in spaCy.
`benepar_pl` | Polish | Full support for NLTK (including parsing from raw text.) Usage with spaCy first requires implementing Polish support in spaCy.
`benepar_sv` | Swedish | Full support for NLTK (including parsing from raw text.) Usage with spaCy first requires implementing Swedish support in spaCy.
`benepar_en` | English | **No part-of-speech tagging capabilities**: we recommend using `benepar_en2` instead. When using this model, part-of-speech tags will be inherited from either NLTK (requires `nltk.download('averaged_perceptron_tagger')`) or spaCy; however, we've found that our own tagger in models such as `benepar_en2` gives better results. This model was released to accompany our ACL 2018 paper, and is retained for compatibility. 95.07 F1 on WSJ test set.
`benepar_en_small` | English | **No part-of-speech tagging capabilities**: we recommend using `benepar_en2` instead. This model was released to accompany our ACL 2018 paper, and is retained for compatibility. A smaller model that is 3-4x faster than the `benepar_en` when running on CPU because it uses a smaller version of ELMo. 94.65 F1 on WSJ test set.
`benepar_en_ensemble` | English | **No part-of-speech tagging capabilities**: we recommend using `benepar_en2_large` instead. This model was released to accompany our ACL 2018 paper, and is retained for compatibility. An ensemble of two parsers: one that uses the original ELMo embeddings and one that uses the 5.5B ELMo embeddings. A GPU is highly recommended for running the ensemble. 95.43 F1 on WSJ test set.


## Training

The code used to train our parsing models is currently different from the code used to parse sentences in the release version described above, though both are stored in this repository. The training code uses PyTorch and can be obtained by cloning this repository from GitHub. The release version uses TensorFlow instead, because it allows serializing the parsing model into a single file on disk in a way that minimizes software dependencies and reduces file size on disk.

#### Software Requirements for Training
* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 0.4.1, 1.0/1.1, or any compatible version.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. If training on the SPMRL datasets, you will need to run `make` inside the `EVALB_SPMRL/` directory instead.
* [AllenNLP](http://allennlp.org/) 0.7.0 or any compatible version (only required when using ELMo word representations)
* [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT) 0.4.0 or any compatible version (only required when using BERT word representations)

#### Pre-trained Models (PyTorch)

The following pre-trained parser models are available for download:
* [`en_charlstm_dev.93.61.pt`](https://github.com/nikitakit/self-attentive-parser/releases/download/models/en_charlstm_dev.93.61.pt): Our best English single-system parser that does not rely on external word representations
* [`en_elmo_dev.95.21.pt`](https://github.com/nikitakit/self-attentive-parser/releases/download/models/en_elmo_dev.95.21.pt): The best English single-system parser from our ACL 2018 paper. Using this parser requires ELMo weights, which must be downloaded separately.

To use ELMo embeddings, download the following files into the `data/` folder (preserving their names):

* [`elmo_2x4096_512_2048cnn_2xhighway_options.json`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
* [`elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)

There is currently no command-line option for configuring the locations/names of the ELMo files.

Pre-trained BERT weights will be automatically downloaded as needed by the `pytorch-pretrained-bert` package.

### Training Instructions

A new model can be trained using the command `python src/main.py train ...`. Some of the available arguments are:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--train-path` | Path to training trees | `data/02-21.10way.clean`
`--dev-path` | Path to development trees | `data/22.auto.clean`
`--batch-size` | Number of examples per training update | 250
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--subbatch-max-tokens` | Maximum number of words to process in parallel while training (a full batch may not fit in GPU memory) | 2000
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the development set | 100
`--numpy-seed` | NumPy random seed | Random
`--use-words` | Use learned word embeddings | Do not use word embeddings
`--use-tags` | Use predicted part-of-speech tags as input | Do not use predicted tags
`--use-chars-lstm` | Use learned CharLSTM word representations | Do not use CharLSTM
`--use-elmo` | Use pre-trained ELMo word representations | Do not use ELMo
`--use-bert` | Use pre-trained BERT word representations | Do not use BERT
`--bert-model` | Pre-trained BERT model to use if `--use-bert` is passed | `bert-base-uncased`
`--no-bert-do-lower-case` | Instructs the BERT tokenizer to retain case information (setting should match the BERT model in use) | Perform lowercasing
`--predict-tags` | Adds a part-of-speech tagging component and auxiliary loss to the parser | Do not predict tags

Additional arguments are available for other hyperparameters; see `make_hparams()` in `src/main.py`. These can be specified on the command line, such as `--num-layers 2` (for numerical parameters), `--use-tags` (for boolean parameters that default to False), or `--no-partitioned` (for boolean parameters that default to True).

If `--use-tags` is passed, the training and development trees are assumed to have predicted part-of-speech tags. If `--predict-tags` is passed, the data is assumed to have ground-truth tags instead. As a result, these two options can't be used simultaneously. Note that the files we provide in the `data/` folder have predicted tags, and that data with gold tags must be obtained separately.

For each development evaluation, the F-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development F-score.

As an example, to train an English parser using the default hyperparameters, you can use the command:

```
python src/main.py train --use-words --use-chars-lstm --model-path-base models/en_charlstm --d-char-emb 64
```

To train an English parser that uses ELMo embeddings, the command is:

```
python src/main.py train --use-elmo --model-path-base models/en_elmo --num-layers 4
```

To train an English parser that uses BERT, the command is:

```
python src/main.py train --use-bert --model-path-base models/en_bert --bert-model "bert-large-uncased" --num-layers 2 --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500
```

### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-path` | Path to test trees | `data/23.auto.clean`
`--test-path-raw` | Alternative path to test trees that is used for evalb only (used to double-check that evaluation against pre-processed trees does not contain any bugs) | Compare to trees from `--test-path`
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the test set | 100

If the parser was trained to have predicted part-of-speech tags as input (via the `--use-tags` flag) the test trees are assumed to have predicted part-of-speech tags. Otherwise, the tags in the test trees are not used as input to the parser.

As an example, after extracting the pre-trained model, you can evaluate it on the test set using the following command:

```
python src/main.py test --model-path-base models/nk_base6_lstm_dev.93.61.pt
```

The pre-trained model with CharLSTM embeddings obtains F-scores of 93.61 on the development set and 93.55 on the test set. The pre-trained model with ELMo embeddings obtains F-scores of 95.21 on the development set and 95.13 on the test set.

### Using the Trained Models

See the `run_parse` function in `src/main.py` for an example of how a parser can be loaded from disk and used to parse sentences using the PyTorch codebase.

The `export/export.py` file contains the code we used to convert our ELMo-based parser to a TensorFlow graph (for use in the release version of the parser). For our BERT-based parsers, consult `export/export_bert.py` instead. This exporting code hard-codes certain hyperparameter choices, so you will likely need to tweak it to export your own models. Exporting the model to TensorFlow allows it to be stored in a single file, including all ELMo/BERT weights. We also use TensorFlow's graph transforms to shrink the model size on disk with only a tiny impact on parsing accuracy: the compressed ELMo model obtains an F1-score of 95.07 on the test set, compared to 95.13 for the uncompressed model.

## Reproducing Experiments

The code used for our ACL 2018 paper is tagged `acl2018` in git. The `EXPERIMENTS.md` file in that version of the code contains additional notes about the command-line arguments we used to perform the experiments reported in our ACL 2018 paper.

The version of the code currently in this repository has added new features (such as BERT support and part-of-speech tag prediction), eliminated some of the less-performant parser variations (e.g. the CharConcat word representation), and has updated to a newer version of PyTorch. The `EXPERIMENTS.md` file now describes the commands used to train our best-performing single-system parser for each language that we evaluate on.

## Citation

If you use this software for research, please cite our paper as follows:

```
@InProceedings{Kitaev-2018-SelfAttentive,
  author    = {Kitaev, Nikita and Klein, Dan},
  title     = {Constituency Parsing with a Self-Attentive Encoder},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2018},
  address   = {Melbourne, Australia},
  publisher = {Association for Computational Linguistics},
}
```

## Credits

The code in this repository and portions of this README are based on https://github.com/mitchellstern/minimal-span-parser
