# Berkeley Neural Parser

This is a Python implementation of the parsers described in "Constituency Parsing with a Self-Attentive Encoder" from ACL 2018.

## Contents
1. [Installation](#installation)
1. [Usage](#usage)
2. [Training](#training)
3. [Reproducing Experiments](#reproducing-experiments)
4. [Citation](#citation)
5. [Credits](#credits)

If you are primarily interested in training your own parsing models, skip to the [Training](#training) section of this README.

## Installation

To install the parser, run the commands:
```bash
$ pip install cython numpy
$ pip install benepar[cpu]
```

Cython and numpy should be installed separately prior to installing benepar. Note that `pip install benepar[cpu]` has a dependency on the `tensorflow` pip package, which is a CPU-only version of tensorflow. Use `pip install benepar[gpu]` to instead introduce a dependency on `tensorflow-gpu`. Installing a GPU-enabled version of TensorFlow will likely require additional steps; see the [official TensorFlow installation instructions](https://www.tensorflow.org/install/) for details.

Parsing models need to be downloaded separately, using the commands:
```python
>>> import benepar
>>> benepar.download('benepar_en')
```

The following English parsing models are available:
* `benepar_en` (95.07 F1 on test, 91.2 MB on disk): default English model, uses the original ELMo embeddings
* `benepar_en_small` (94.65 F1 on test, 25.4 MB on disk): A smaller model that is 3-4x faster than the original when running on CPU. Uses the same self-attentive architecture as the original, but small ELMo embeddings.
* `benepar_en_ensemble` (95.43 F1 on test, 214 MB on disk): An ensemble of two parsers: one that uses the original ELMo embeddings and one that uses the 5.5B ELMo embeddings. A GPU is highly recommended for running the ensemble.

### Additional Required Models

Benepar is designed to add parsing capabilities to an existing NLP pipeline that has support for tokenization and part-of-speech tagging. To do this, it should be used in conjunction with one of two NLP libraries for Python: [NLTK](http://www.nltk.org/) or [spaCy](https://spacy.io/).

If using NLTK, the commands to install additional models are:
```python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
```

For spaCy, the command to install models for English is:
```sh
$ python -m spacy download en
```

## Usage
### Usage with NLTK

```python
>>> import benepar
>>> parser = benepar.Parser("benepar_en")
>>> tree = parser.parse("Short cuts make long delays.")
>>> print(tree)
(S
  (NP (JJ Short) (NNS cuts))
  (VP (VBP make) (NP (JJ long) (NNS delays)))
  (. .))
```

Speed note: the first call to `parse` will take much longer that subsequent calls, as caches are being warmed up.

The parser can also parse pre-tokenized text:
```python
>>> parser.parse(['Short', 'cuts', 'make', 'long', 'delays', '.'])
```

Use `parse_sents` to parse multiple sentences. It accepts an entire document as a string, or a list of sentences.
```python
>>> parser.parse_sents("The time for action is now. It's never too late to do something.")
>>> parser.parse_sents(["The time for action is now.", "It's never too late to do something."])
```

All parse trees returned are represented using `nltk.Tree` objects.

### Usage with spaCy

Benepar also ships with a component that integrates with spaCy:
```python
>>> import spacy
>>> from benepar.spacy_plugin import BeneparComponent
>>> nlp = spacy.load('en')
>>> nlp.add_pipe(BeneparComponent("benepar_en"))
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

## Training

The code used to train our parsing models is currently different from the code used to parse sentences in the release version described above, though both are stored in this repository. The training code uses PyTorch and can be obtained by cloning this repository from GitHub. The release version uses TensorFlow instead, because it allows serializing the parsing model into a single file on disk in a way that minimizes software dependencies and reduces file size on disk.

#### Software Requirements for Training
* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 0.3.x. This code has not been tested with newer releases of PyTorch.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. If training on the SPMRL datasets, you will need to run `make` inside the `EVALB_SPMRL/` directory instead.
* [AllenNLP](http://allennlp.org/) 0.4.0 or any compatible version (only required when using ELMo word representations)

#### Pre-trained Models (PyTorch)

The following pre-trained parser models are available for download:
* [`en_charlstm_dev.93.61.pt`](https://github.com/nikitakit/self-attentive-parser/releases/download/models/en_charlstm_dev.93.61.pt): Our best English single-system parser that does not rely on external word representations
* [`en_elmo_dev.95.21.pt`](https://github.com/nikitakit/self-attentive-parser/releases/download/models/en_elmo_dev.95.21.pt): Our best English single-system parser. Using this parser requires ELMo weights, which must be downloaded separately.

To use ELMo embeddings, download the following files into the `data/` folder (preserving their names):

* [`elmo_2x4096_512_2048cnn_2xhighway_options.json`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
* [`elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5`](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)

There is currently no command-line option for configuring the locations/names of the ELMo files.

### Training Instructions

A new model can be trained using the command `python3 src/main.py train ...`. Some of the available arguments are:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--train-path` | Path to training trees | `data/02-21.10way.clean`
`--dev-path` | Path to development trees | `data/22.auto.clean`
`--batch-size` | Number of examples per training update | 250
`--epochs` | Number of training epochs | No limit
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--subbatch-max-tokens` | Maximum number of words to process in parallel while training (a full batch may not fit in GPU memory) | 2000
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the development set | 100
`--print-vocabs` | Print the vocabularies before training | Do not print the vocabularies
`--numpy-seed` | NumPy random seed | Random
`--use-words` | Use learned word embeddings | Do not use word embeddings
`--use-chars-lstm` | Use learned CharLSTM word representations | Do not use CharLSTM
`--use-elmo` | Use pre-trained ELMo word representations | Do not use ELMo

Additional arguments are available for other hyperparameters; see `make_hparams()` in `src/main.py`. These can be specified on the command line, such as `--d-char-emb 64` (for numerical paramters), `--use-tags` (for boolean parameters that default to False), or `--no-partitioned` (for boolean parameters that default to True).

The training and development trees are assumed to have predicted part-of-speech tags, but they only affect final parser output if the `--use-tags` option is passed.

For each development evaluation, the F-score on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development F-score.

As an example, to train an English parser using the default hyperparameters, you can use the command:

```
python src/main.py train --use-words --use-chars-lstm --model-path-base models/en_charlstm --d-char-emb 64
```

To train an English parser that uses ELMo embeddings, the command is:

```
python src/main.py train --use-elmo --model-path-base models/en_elmo --num-layers 4
```

The above commands were used to train our two best English parsers; the `EXPERIMENTS.md` file contains additional notes about the experiments reported in our paper.

### Evaluation Instructions

A saved model can be evaluated on a test corpus using the command `python3 src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-path` | Path to test trees | `data/23.auto.clean`
`--test-path-raw` | Alternative path to test trees that is used for evalb only (used to double-check that evaluation against pre-processed trees does not contain any bugs) | Compare to trees from `--test-path`
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the test set | 100

The test trees are assumed to have predicted part-of-speech tags.

As an example, after extracting the pre-trained model, you can evaluate it on the test set using the following command:

```
python src/main.py test --model-path-base models/nk_base6_lstm_dev.93.61.pt
```

The pre-trained model with CharLSTM embeddings obtains F-scores of 93.61 on the development set and 93.55 on the test set. The pre-trained model with ELMo embeddings obtains F-scores of 95.21 on the development set and 95.13 on the test set.

### Using the Trained Models

See the `run_test` function in `src/main.py` for an example of how a parser can be loaded from disk and used to parse sentences using the PyTorch codebase.

The `export/export.py` file contains the code we used to convert our best parser to a TensorFlow graph (for use in the release version of the parser). This exporting code hard-codes certain hyperparameter choices, so you will likely need to tweak it to export your own models. Exporting the model to TensorFlow allows it to be stored in a single file, including all ELMo weights. We also use TensorFlow's graph transforms to shrink the model size on disk with only a tiny impact on parsing accuracy: the compressed model obtains an F1-score of 95.07 on the test set, compared to 95.13 for the uncompressed model.

## Reproducing Experiments

The `EXPERIMENTS.md` file contains additional notes about the command-line arguments we used to perform the experiments reported in our paper. The current version of the code is sufficient to run all the commands listed in `EXPERIMENTS.md`.

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
