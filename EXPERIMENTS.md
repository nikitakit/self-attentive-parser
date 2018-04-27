# Experiments

This file contains commands used for the different experiments reported in our paper.

Note that none of these commands specify a termination criterion for training (we handled termination at the job-scheduling level). In practice it is pretty clear when training has finished because the learning rate is decayed and eventually all new iterates produce indistinguishable results on the development set.

## Unfactored base model (Section 2.4)

To train the base parser with factoring disabled, run

```bash
python src/main.py train \
    --use-tags --use-words \
    --model-path-base models/nk_base6_nopartition --no-partitioned
```

## Disabling content-based attention

This experiment is briefly mentioned in Section 3 to motivate factoring of the model.
```bash
python src/main.py train \
    --use-tags --use-words \
    --model-path-base models/nk_base6_pos8 --no-partitioned --num-layers-position-only 8
```

## Factored model variations (Table 4)

Using learned word embeddings and an external tagger:
```bash
python src/main.py train \
    --use-tags --use-words \
    --model-path-base models/nk_base6_wordstags
```

Using learned word embeddings only:
```bash
python src/main.py train \
    --use-words \
    --model-path-base models/nk_base6_words
```

Using learned word embeddings, and a character LSTM:
```bash
python src/main.py train \
    --use-words --use-chars-lstm \
    --model-path-base models/nk_base6_wordslstm --d-char-emb 64
```

Using a character LSTM only:
```bash
python src/main.py train \
    --use-chars-lstm \
    --model-path-base models/nk_base6_lstm --d-char-emb 64
```

Using learned word embeddings, and CharConcat:
```bash
python src/main.py train \
    --use-words --use-chars-concat \
    --model-path-base models/nk_base6_wordsconcat
```

Using CharConcat only:
```bash
python src/main.py train \
    --use-chars-concat \
    --model-path-base models/nk_base6_concat
```

## SPMRL models (Section 6.2)

Below is our code for training and evaluating models on the SPMRL dataset. Note that the data itself must be obtained separately.

To train the parser, update the `SPMRL_BASE_PATH` and `SPMRL_LANG` variables accordingly. Note that each language has some hardcoded parameters that relate to preprocessing of the treebanks; see `trees.py::load_trees`. These parameters require the presence of the language name in the path to the text files containing the trees; if your files are named differently you may have to modify the code in order for it to run.

Code for training:
```bash
SPMRL_BASE_PATH="your path here" # ours is a folder named READY_TO_SHIP_FINAL/
SPMRL_LANG=Arabic
echo "Language: ${SPMRL_LANG}"

SPMRL_PATH=${SPMRL_BASE_PATH}/${SPMRL_LANG^^}_SPMRL/pred/ptb
TRAIN_PATH=${SPMRL_PATH}/train/train.${SPMRL_LANG}.pred.ptb
DEV_PATH=${SPMRL_PATH}/dev/dev.${SPMRL_LANG}.pred.ptb

if [ ! -e "${TRAIN_PATH}" ]; then
    echo "Only train5k data condition is available for this language"
    TRAIN_PATH=${SPMRL_PATH}/train5k/train5k.${SPMRL_LANG}.pred.ptb
fi

python src/main.py train \
    --train-path ${TRAIN_PATH} \
    --dev-path ${DEV_PATH} \
    --evalb-dir EVALB_SPMRL \
    --use-chars-lstm --d-char-emb 64 \
    --model-path-base models/${SPMRL_LANG}_nk_base6_lstm
```

We used the following additional parameters for individual experiments:
* To use word embeddings, add `--use-words` (do not remove `--use-chars-lstm`!)
* For Arabic, add `--sentence-max-len 1000`
* For Hebrew, add `--learning-rate 0.002`
* For Polish, add `--learning-rate 0.0015`
* For Swedish, add `--learning-rate 0.002`

Code for evaluation:
```bash
SPMRL_BASE_PATH="your path here" # ours is a folder named READY_TO_SHIP_FINAL/
SPMRL_LANG=Arabic
echo "Language: ${SPMRL_LANG}"

SPMRL_PATH=${SPMRL_BASE_PATH}/${SPMRL_LANG^^}_SPMRL/pred/ptb
TEST_PATH=${SPMRL_PATH}/test/test.${SPMRL_LANG}.pred.ptb

python src/main.py test --test-path ${TEST_PATH} --evalb-dir EVALB_SPMRL --model-path-base models/${SPMRL_LANG}_nk_base6_lstm
```

## Other experiments (Tables 1,2,3)
Unfortunately, the code for these experiments is not included in the public release.
