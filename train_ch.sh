
#python src/main.py train --use-words \
#    --use-chars-lstm \
#    --model-path-base models/en_charlstm \
#    --d-char-emb 64

CUDA_VISIBLE_DEVICES=4 python src/main.py train --lang "zh" --use-chars-lstm \
    --model-path-base models/cn_charlstm_l4_aux \
    --train-path "data/ctb51_train_berkeley.clean" \
    --dev-path "data/ctb51_dev_berkeley.clean" \
    --num-layers 4 \
    --learning-rate 0.0008 \
    --batch-size 250 \
    --eval-batch-size 100 \
    --predict-tags

