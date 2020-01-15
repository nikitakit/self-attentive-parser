
#python src/main.py train --use-words \
#    --use-chars-lstm \
#    --model-path-base models/en_charlstm \
#    --d-char-emb 64

CUDA_VISIBLE_DEVICES=3 python src/main.py train --lang "zh" --use-xlm \
    --model-path-base models/cn_xlmxnli15_lr25 \
    --xlm-model "xlm-mlm-tlm-xnli15-1024" \
    --train-path "data/ctb51_train_berkeley.clean" \
    --dev-path "data/ctb51_dev_berkeley.clean" \
    --num-layers 2 \
    --learning-rate 0.000025 \
    --batch-size 32 \
    --eval-batch-size 16 \
    --subbatch-max-tokens 500

