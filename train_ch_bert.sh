
#python src/main.py train --use-words \
#    --use-chars-lstm \
#    --model-path-base models/en_charlstm \
#    --d-char-emb 64

CUDA_VISIBLE_DEVICES=4 python src/main.py train --lang "zh" --use-bert \
    --model-path-base models/cn_roberta_aux \
    --bert-model "/data2/lfsong/data.pre_lm/roberta_wwm_ext.tar.gz" \
    --train-path "data/ctb51_train_berkeley.clean" \
    --dev-path "data/ctb51_dev_berkeley.clean" \
    --num-layers 2 \
    --learning-rate 0.00005 \
    --batch-size 32 \
    --eval-batch-size 16 \
    --subbatch-max-tokens 500 \
    --predict-tags

