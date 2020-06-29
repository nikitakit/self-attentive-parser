
time CUDA_VISIBLE_DEVICES=-1 python src/main.py test \
    --model-path-base $1 \
    --eval-batch-size 1 \
    --test-path $2

