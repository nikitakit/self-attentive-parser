
#python src/main.py train --use-words \
#    --use-chars-lstm \
#    --model-path-base models/en_charlstm \
#    --d-char-emb 64

CUDA_VISIBLE_DEVICES=6 python src/main.py train --lang "en" --use-bert \
    --model-path-base models/en_genia_bert_uncased_aux \
    --bert-model "bert-base-uncased" \
    --train-path "data/ptb_genia_train.trees" \
    --dev-path "data/ptb_genia_dev.trees" \
    --num-layers 2 \
    --learning-rate 0.00005 \
    --batch-size 32 \
    --eval-batch-size 16 \
    --subbatch-max-tokens 500 \
    --predict-tags

