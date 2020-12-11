#!/bin/bash
set -e

RAW_DATA_DIR="../raw"
SCRIPT_DIR="../common"

ENSURE_TOP=${SCRIPT_DIR}/ensure_top.py
STRIP_FUNCTIONAL=${SCRIPT_DIR}/strip_functional.py

LDC99T42_ROOT=${RAW_DATA_DIR}/treebank_3/parsed/mrg/wsj
LDC2015T13_ROOT=${RAW_DATA_DIR}/eng_news_txt_tbnk-ptb_revised/data/penntree

python get_wsj.py --orig_root ${LDC99T42_ROOT} --revised_root ${LDC2015T13_ROOT}

TRAIN_ORIG="train_02-21.LDC99T42.withtraces"
DEV_ORIG="dev_22.LDC99T42.withtraces"
TEST_ORIG="test_23.LDC99T42.withtraces"

python $STRIP_FUNCTIONAL < $TRAIN_ORIG | python $ENSURE_TOP | sed 's/PRT|ADVP/PRT/g' > train_02-21.LDC99T42
python $STRIP_FUNCTIONAL < $DEV_ORIG | python $ENSURE_TOP > dev_22.LDC99T42
python $STRIP_FUNCTIONAL < $TEST_ORIG | python $ENSURE_TOP > test_23.LDC99T42

TRAIN="train_02-21.LDC2015T13.withtraces"
DEV="dev_22.LDC2015T13.withtraces"
TEST="test_23.LDC2015T13.withtraces"

python fixup_data_errors.py < $TRAIN | python $STRIP_FUNCTIONAL | python $ENSURE_TOP | sed 's/PRT|ADVP/PRT/g' > train_02-21.LDC2015T13
python $STRIP_FUNCTIONAL < $DEV | python $ENSURE_TOP > dev_22.LDC2015T13
python $STRIP_FUNCTIONAL < $TEST | python $ENSURE_TOP > test_23.LDC2015T13

python recover_whitespace.py --treebank3_root ${RAW_DATA_DIR}/treebank_3 --revised_root ${LDC2015T13_ROOT}

python convert_to_revised_tokenization.py --orig train_02-21.LDC99T42 --revised train_02-21.LDC2015T13 --out train_02-21.LDC99T42.retokenized
python convert_to_revised_tokenization.py --orig dev_22.LDC99T42 --revised dev_22.LDC2015T13 --out dev_22.LDC99T42.retokenized
python convert_to_revised_tokenization.py --orig test_23.LDC99T42 --revised test_23.LDC2015T13 --out test_23.LDC99T42.retokenized
