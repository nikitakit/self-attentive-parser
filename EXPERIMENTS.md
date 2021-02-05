# Experiments

This file contains commands that can be used to replicate the best parsing numbers that we reported in our papers. See `data/README.md` for information on how to prepare the required datasets.

Note that the code in this repository has been updated since the time the papers were published: some of the less-performant parser variations (e.g. the CharConcat word representation) are no longer supported, and there are minor architectural differences (e.g. whether the epsilon value in Layer Normalization is inside or outside the square root).

These commands should suffice for most applications, but our older code remains available under the `acl2018` and `acl2019` git tags in this repository. Refer to the `EXPERIMENTS.md` file in those tagged versions for the exact code and commands we used for our publications.

Also note that as of January 2021 a plethora of pre-trained models are now available. The versions of BERT used here will result in lower F1 scores than selecting the best available model on the HuggingFace model hub. These commands are primarily aimed at comparing different parsing approaches with the choice of pre-trained model held fixed.

## English models

Without pre-training:
```bash
python src/main.py train \
    --train-path "data/wsj/train_02-21.LDC99T42" \
    --dev-path "data/wsj/dev_22.LDC99T42" \
    --use-chars-lstm --use-encoder --num-layers 8 \
    --batch-size 250 --learning-rate 0.0008 \
    --model-path-base models/English_charlstm
```

With BERT (single model, large, uncased):
```bash
python src/main.py train \
    --train-path "data/wsj/train_02-21.LDC99T42" \
    --dev-path "data/wsj/dev_22.LDC99T42" \
    --use-pretrained --pretrained-model "bert-large-uncased" \
    --use-encoder --num-layers 2 \
    --predict-tags \
    --model-path-base models/English_bert_large_uncased
```

To evaluate:
```bash
python src/main.py test \
    --test-path "data/wsj/test_23.LDC99T42" \
    --no-predict-tags \
    --model-path models/English_bert_large_uncased_*.pt
```

## SPMRL models

To train:
```bash
SPMRL_LANG=Arabic
echo "Language: ${SPMRL_LANG}"

EXTRA_ARGS=
if [ "$SPMRL_LANG" = "Arabic" ]; then
    # There are sentences in the train and dev sets that are too long for BERT.
    # Fortunately, there are no such long sentences in the test set
    EXTRA_ARGS="--text-processing arabic-translit --max-len-train 266 --max-len-dev 494"
fi
if [ "$SPMRL_LANG" = "Hebrew" ]; then
    EXTRA_ARGS="--text-processing hebrew"
fi

python src/main.py train \
    --train-path data/spmrl/${SPMRL_LANG}.train \
    --dev-path data/spmrl/${SPMRL_LANG}.dev \
    --evalb-dir EVALB_SPMRL \
    --use-pretrained --pretrained-model "bert-base-multilingual-cased" \
    --use-encoder --num-layers 2 \
    --predict-tags \
    --model-path-base models/${SPMRL_LANG}_bert_base_multilingual_cased \
    $EXTRA_ARGS
```

Code for evaluation:
```bash
SPMRL_LANG=Arabic
echo "Language: ${SPMRL_LANG}"

EXTRA_ARGS=
if [ "$SPMRL_LANG" = "Arabic" ]; then
    EXTRA_ARGS="--text-processing arabic-translit"
fi
if [ "$SPMRL_LANG" = "Hebrew" ]; then
    EXTRA_ARGS="--text-processing hebrew"
fi

python src/main.py test \
    --test-path data/spmrl/${SPMRL_LANG}.test \
    --evalb-dir EVALB_SPMRL \
    --no-predict-tags \
    --model-path models/${SPMRL_LANG}_bert_base_multilingual_cased_*.pt \
    $EXTRA_ARGS
```

*Note*: the Hebrew data comes in two varieties: one that uses Hebrew characters, and one that uses transliterated characters. If your copy of the treebank uses transliterated characters, use `--text-processing hebrew-translit` instead of `--text-processing hebrew`.


## Chinese models

To train:
```bash
python src/main.py train \
    --train-path "data/ctb_5.1/ctb.train" \
    --dev-path "data/ctb_5.1/ctb.dev" \
    --text-processing "chinese" \
    --use-pretrained --pretrained-model "bert-base-chinese" \
    --predict-tags \
    --model-path-base models/Chinese_bert_base_chinese
```

To evaluate:
```bash
python src/main.py test \
    --test-path "data/ctb_5.1/ctb.test" \
    --text-processing "chinese" \
    --no-predict-tags \
    --model-path models/Chinese_bert_base_chinese_*.pt
```
