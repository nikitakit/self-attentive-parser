#!/bin/bash
set -e

RAW_DATA_DIR="../raw"
SCRIPT_DIR="../common"

STRIP_FUNCTIONAL=${SCRIPT_DIR}/strip_functional.py
SPMRL_ENSURE_TOP=spmrl_ensure_top.py
STRIP_SPMRL_FEATURES=strip_spmrl_features.py

ATB1_ROOT=${RAW_DATA_DIR}/atb1_v4_1
ATB2_ROOT=${RAW_DATA_DIR}/atb_2_3.1
ATB3_ROOT=${RAW_DATA_DIR}/atb3_v3_2

SPMRL_BASE_PATH=${RAW_DATA_DIR}
for SPMRL_LANG in Arabic Basque French German Hebrew Hungarian Korean Polish Swedish
do
    SPMRL_LANG_UPPERCASE=$(echo ${SPMRL_LANG} | tr '[:lower:]' '[:upper:]')
    SPMRL_PATH=${SPMRL_BASE_PATH}/${SPMRL_LANG_UPPERCASE}_SPMRL/gold/ptb
    TRAIN_PATH=${SPMRL_PATH}/train/train.${SPMRL_LANG}.gold.ptb
    DEV_PATH=${SPMRL_PATH}/dev/dev.${SPMRL_LANG}.gold.ptb
    TEST_PATH=${SPMRL_PATH}/test/test.${SPMRL_LANG}.gold.ptb

    if [ ! -e "${TRAIN_PATH}" ]; then
        # Some languages only have the train5k data condition
        TRAIN_PATH=${SPMRL_PATH}/train5k/train5k.${SPMRL_LANG}.gold.ptb
    fi

    if [ ! -e "${TRAIN_PATH}" ]; then
        if [[ "${SPMRL_LANG}" == "Swedish" ]]; then
            # At least one variant of the SPMRL data exists in which the
            # Swedish paths have 'swedish' in lowercase
            TRAIN_PATH=${SPMRL_PATH}/train5k/train5k.swedish.gold.ptb
            DEV_PATH=${SPMRL_PATH}/dev/dev.swedish.gold.ptb
            TEST_PATH=${SPMRL_PATH}/test/test.swedish.gold.ptb
        fi
    fi

    if [ -e "${TRAIN_PATH}" ]; then
        echo "Processing data for language: ${SPMRL_LANG}"
        python $SPMRL_ENSURE_TOP < "${TRAIN_PATH}" | python $STRIP_FUNCTIONAL | python $STRIP_SPMRL_FEATURES > ${SPMRL_LANG}.train
        python $SPMRL_ENSURE_TOP < "${DEV_PATH}" | python $STRIP_FUNCTIONAL | python $STRIP_SPMRL_FEATURES > ${SPMRL_LANG}.dev
        python $SPMRL_ENSURE_TOP < "${TEST_PATH}" | python $STRIP_FUNCTIONAL | python $STRIP_SPMRL_FEATURES > ${SPMRL_LANG}.test
    elif [[ "${SPMRL_LANG}" == "Arabic" ]]; then
        if [ -d "${ATB1_ROOT}" -a -d "${ATB2_ROOT}" -a -d "${ATB3_ROOT}" ]; then
            echo "SPMRL-compatible Arabic data will be re-created from ATB parts 1-3 as distributed by LDC"
        else
            echo "Skipping ${SPMRL_LANG}: Neither SPMRL data nor LDC Arabic Treebank parts 1-3 were found"
        fi    
    else
        echo "Skipping ${SPMRL_LANG}: SPMRL data not found for this language"
    fi
done

if [ -e "${SPMRL_BASE_PATH}/ARABIC_SPMRL/gold/ptb" ]; then
    # Arabic data was already copied over from the SPMRL distribution
    :
elif [ -d "${ATB1_ROOT}" -a -d "${ATB2_ROOT}" -a -d "${ATB3_ROOT}" ]; then
    cat << EOF

Re-creating SPMRL-compatible Arabic data from ATB parts 1-3
This data should be interchangeable for the purposes of running EVALB, but uses
a different part-of-speech tagging convention compared to the data distributed
for the SPMRL 2013/2014 shared tasks.

EOF
    mkdir -p arabic
    pushd arabic

    # Download UD data. We use this to determine the correct splits (even though
    # it's possible to also possible to specify the splits manually)
    if [ ! -f UD_Arabic-NYUAD-r2.7.zip ]; then
        wget https://github.com/UniversalDependencies/UD_Arabic-NYUAD/archive/r2.7.zip -O UD_Arabic-NYUAD-r2.7.zip
    fi

    if [ ! -d UD_Arabic-NYUAD-r2.7 ]; then
        unzip UD_Arabic-NYUAD-r2.7.zip
    fi
    popd

    mkdir -p arabic/penntree/without-vowel
    cp ${ATB1_ROOT}/data/penntree/without-vowel/*.tree arabic/penntree/without-vowel
    cp ${ATB2_ROOT}/data/penntree/without-vowel/*.tree arabic/penntree/without-vowel
    cp ${ATB3_ROOT}/data/penntree/without-vowel/*.tree arabic/penntree/without-vowel

    UD_TRAIN=arabic/UD_Arabic-NYUAD-r2.7/ar_nyuad-ud-train.conllu
    UD_DEV=arabic/UD_Arabic-NYUAD-r2.7/ar_nyuad-ud-dev.conllu
    UD_TEST=arabic/UD_Arabic-NYUAD-r2.7/ar_nyuad-ud-test.conllu
    python get_atb_spmrlcompat.py --tree_dir arabic/penntree/without-vowel --out_dir arabic --ud_train ${UD_TRAIN} --ud_dev ${UD_DEV} --ud_test ${UD_TEST}

    python $SPMRL_ENSURE_TOP < "arabic/train.spmrlcompat.withmosttraces" | python $STRIP_FUNCTIONAL > Arabic.train
    python $SPMRL_ENSURE_TOP < "arabic/dev.spmrlcompat.withmosttraces" | python $STRIP_FUNCTIONAL > Arabic.dev
    python $SPMRL_ENSURE_TOP < "arabic/test.spmrlcompat.withmosttraces" | python $STRIP_FUNCTIONAL > Arabic.test
    echo "Done re-creating Arabic data."
fi
