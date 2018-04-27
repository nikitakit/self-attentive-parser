# Constituency Parsing with a Self-Attentive Encoder

This is a Python implementation of the parsers described in "Constituency Parsing with a Self-Attentive Encoder" from ACL 2018.

## Requirements

### Software
* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 0.3.x. This code has not been tested with newer releases of PyTorch.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation. If training on the SPMRL datasets, you will need to run `make` inside the `EVALB_SPMRL/` directory instead.
* [AllenNLP](http://allennlp.org/) 0.4.0 or any compatible version (only required when using ELMo word representations)

### Pre-trained models

The following pre-trained parser models are available for download:
* [`en_charlstm_dev.93.61.pt`](https://github.com/nikitakit/self-attentive-parser/releases/download/models/en_charlstm_dev.93.61.pt): Our best English single-system parser that does not rely on external word representations
* [`en_elmo_dev.95.21.pt`](https://github.com/nikitakit/self-attentive-parser/releases/download/models/en_elmo_dev.95.21.pt): Our best English parser. Using this parser requires ELMo weights, which must be downloaded separately.

To use ELMo embeddings, download the following files into the `data/` folder (preserving their names):


* https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
* https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5

There is currently no command-line option for configuring the locations/names of the ELMo files.

## Training

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

## Evaluation

A saved model can be evaluated on a test corpus using the command `python3 src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-path` | Path to test trees | `data/23.auto.clean`
`--test-path-raw` | Alternative path to test trees that is used for evalb only (used to double-check that evaluation against pre-processed trees does not contain any bugs) | Compare to trees from `--test-path`
`--eval-batch-size` | Number of examples to process in parallel when evaluating on the test set | 100

The test trees are assumed to have predicted part-of-speech tags.

As an example, after extracting the pre-trained top-down model, you can evaluate it on the test set using the following command:

```
python src/main.py test --model-path-base models/nk_base6_lstm_dev.93.61.pt
```

The pre-trained model with CharLSTM embeddings obtains F-scores of 93.61 on the development set and 93.55 on the test set. The pre-trained model with ELMo embeddings obtains F-scores of 95.21 on the development set and 95.13 on the test set.

## Parsing New Sentences

See the `run_test` function in `src/main.py` for an example of how a parser can be loaded from disk and used to parse sentences.

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
