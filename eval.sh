
CUDA_VISIBLE_DEVICES=7 python src/main.py test \
    --model-path-base $1 \
    --eval-batch-size 32 \
    --test-path $2
