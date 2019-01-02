# Experiments

This file contains commands used for the best parsers reported in our arXiv submission.

## English models

Without pre-training:
```bash
python src/main.py train \
    --use-chars-lstm \
    --model-path-base models/nk_base6_lstm --d-char-emb 64
```

With ELMo:
```bash
python src/main.py train \
    --use-elmo \
    --model-path-base models/nk_base6_elmo1_layers=4_trainproj_nogamma_fixtoks --num-layers 4
```

With BERT (single model, large, uncased):
```bash
python src/main.py train \
    --use-bert --predict-tags \
    --model-path-base models/nk_base9_large --bert-model "bert-large-uncased" \
    --train-path data/02-21.goldtags --dev-path data/22.goldtags \
    --learning-rate 0.00005 --num-layers 2 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500
```

Note that the last model enables part-of-speech tag prediction, which requires using a version of the WSJ data that contains gold tags. This data format is not provided in our repository and must be obtained separately. Disabling part-of-speech tag prediction and training on the data provided in this repository should give comparable parsing accuracies (but it's potentially less helpful for downstream use).

## SPMRL models

Below is our code for training and evaluating models on the SPMRL dataset. Note that the data itself must be obtained separately.

To train the parser, update the `SPMRL_BASE_PATH` and `SPMRL_LANG` variables accordingly. Note that each language has some hardcoded parameters that relate to preprocessing of the treebanks; see `trees.py::load_trees`. These parameters require the presence of the language name in the path to the text files containing the trees; if your files are named differently you may have to modify the code in order for it to run.

Code for training:
```bash
SPMRL_BASE_PATH="your path here" # ours is a folder named READY_TO_SHIP_FINAL/
SPMRL_LANG=Arabic
echo "Language: ${SPMRL_LANG}"

SPMRL_PATH=${SPMRL_BASE_PATH}/${SPMRL_LANG^^}_SPMRL/gold/ptb
TRAIN_PATH=${SPMRL_PATH}/train/train.${SPMRL_LANG}.gold.ptb
DEV_PATH=${SPMRL_PATH}/dev/dev.${SPMRL_LANG}.gold.ptb

if [ ! -e "${TRAIN_PATH}" ]; then
    echo "Only train5k data condition is available for this language"
    TRAIN_PATH=${SPMRL_PATH}/train5k/train5k.${SPMRL_LANG}.gold.ptb
fi

EXTRA_ARGS=
if [ "$SPMRL_LANG" = "Arabic" ]; then
    # There are sentences in the train and dev sets that are too long for BERT.
    # Fortunately, there are no such long sentences in the test set
    EXTRA_ARGS="--bert-transliterate arabic --max-len-train 266 --max-len-dev 494 --sentence-max-len 512"
fi
if [ "$SPMRL_LANG" = "Hebrew" ]; then
    EXTRA_ARGS="--bert-transliterate hebrew"
fi
if [ "$SPMRL_LANG" = "Hungarian" ]; then
    # Prevents out-of-memory issues on a K80 GPU
    EXTRA_ARGS="--subbatch-max-tokens 500"
fi

python src/main.py train \
    --train-path ${TRAIN_PATH} \
    --dev-path ${DEV_PATH} \
    --evalb-dir EVALB_SPMRL \
    --use-bert --predict-tags \
    --model-path-base models/${SPMRL_LANG}_nk_base9 \
    --bert-model "bert-base-multilingual-cased" --no-bert-do-lower-case \
    --learning-rate 0.00005 --num-layers 2 --batch-size 32 --eval-batch-size 32 \
    $EXTRA_ARGS
```

Code for evaluation:
```bash
SPMRL_BASE_PATH="your path here" # ours is a folder named READY_TO_SHIP_FINAL/
SPMRL_LANG=Arabic
echo "Language: ${SPMRL_LANG}"

SPMRL_PATH=${SPMRL_BASE_PATH}/${SPMRL_LANG^^}_SPMRL/pred/ptb
TEST_PATH=${SPMRL_PATH}/test/test.${SPMRL_LANG}.pred.ptb

python src/main.py test --test-path ${TEST_PATH} --evalb-dir EVALB_SPMRL --model-path-base --model-path-base models/${SPMRL_LANG}_nk_base9_*.pt
```

## Chinese models

Below is our code for training models on the Chinese Treebank 5.1. Note that the data itself must be obtained separately, and that it must be converted to the format accepted by our parser.

```bash
CTB_DIR="your path here"

python src/main.py train \
    --train-path ${CTB_DIR}/train.gold.stripped \
    --dev-path ${CTB_DIR}/dev.gold.stripped \
    --use-bert --predict-tags \
    --model-path-base models/Chinese_nk_base9 \
    --bert-model "bert-base-chinese" \
    --learning-rate 0.00005 --num-layers 2 --batch-size 32 --eval-batch-size 32
```
