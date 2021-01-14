#!/bin/bash
set -e

RAW_DATA_DIR="../raw"
SCRIPT_DIR="../common"

ENSURE_TOP=${SCRIPT_DIR}/ensure_top.py
STRIP_FUNCTIONAL=${SCRIPT_DIR}/strip_functional.py

CTB_ROOT=${RAW_DATA_DIR}/ctb5.1_507K


python process_ctb.py --ctb ${CTB_ROOT}/data

for SPLIT in train dev test
do
  STRIPPED=ctb.${SPLIT}
  python $STRIP_FUNCTIONAL < ctb.${SPLIT}.withtraces | python $ENSURE_TOP > $STRIPPED
done
