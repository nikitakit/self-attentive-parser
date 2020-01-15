
CUDA_VISIBLE_DEVICES=0 python src/main.py test \
    --model-path-base $1 \
    --test-path "data/ctb51_test_berkeley.clean"
