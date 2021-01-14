# Parsing Data Generation

## English WSJ

1. Place a copy of the Penn Treebank
([LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)) in `data/raw/treebank_3`.
After doing this, `data/raw/treebank_3/parsed/mrg/wsj` should have folders
named `00`-`24`.
2. Place a copy of the revised Penn Treebank
([LDC2015T13](https://catalog.ldc.upenn.edu/LDC2015T13)) in
`data/raw/eng_news_txt_tbnk-ptb_revised`.
3. Ensure that the active version of Python is Python 3 and has `nltk` and
`pytokenizations` installed.
4. `cd data/wsj && ./build_corpus.sh`

Processed trees are written to the following files:


| File in `data/wsj/`                | Description                                         |
| ---------------------------------- | --------------------------------------------------- |
| `train_02-21.LDC99T42`             | The standard training split for English constituency parsing publications. |
| `train_02-21.LDC99T42.text`        | Non-destructive tokenization of the training split, meaning that words retain their original spelling (without any escape sequences), and information about whitespace between tokens is preserved. |
| `dev_22.LDC99T42`                  | The standard development/validation split for English constituency parsing publications. |
| `dev_22.LDC99T42.text`             | Non-destructive tokenization of the development split. |
| `test_23.LDC99T42`                 | The standard test split for English constituency parsing publications. |
| `test_23.LDC99T42.text`            | Non-destructive tokenization of the test split. |
| `train_02-21.LDC2015T13`           | English training sentences with revised tokenization and syntactic annotation guidelines. These revised guidelines were also used for other treebanks, including the English Web Treebank and OntoNotes. |
| `train_02-21.LDC2015T13.text`      | Non-destructive revised tokenization of the training sentences. |
| `dev_22.LDC2015T13`                | English development/validation sentences with revised tokenization and syntactic annotation guidelines. |
| `dev_22.LDC2015T13.text`           | Non-destructive revised tokenization of the development sentences. |
| `test_23.LDC2015T13`               | English test sentences with revised tokenization and syntactic annotation guidelines. |
| `test_23.LDC2015T13.text`          | Non-destructive revised tokenization of the test sentences. |
| `train_02-21.LDC99T42.retokenized` | Syntatic annotations (labeled brackets) from the standard training split, overlaid on top of the revised tokenization. |
| `dev_22.LDC99T42.retokenized`      | Syntatic annotations (labeled brackets) from the standard development/validation split, overlaid on top of the revised tokenization. |
| `test_23.LDC99T42.retokenized`     | Syntatic annotations (labeled brackets) from the standard test split, overlaid on top of the revised tokenization. |


## Chinese Treebank (CTB 5.1)

This prepares the standard Chinese constituency parsing split, following recent papers such as [Liu and Zhang (2017)](https://www.aclweb.org/anthology/Q17-1004/).

### Instructions

1. Place a copy of the Chinese Treebank 5.1
([LDC2005T01](https://catalog.ldc.upenn.edu/LDC2005T01)) in `data/raw/ctb5.1_507K`.
2. Ensure that the active version of Python is Python 3 and has `nltk` installed.
3. `cd data/ctb_5.1 && ./build_corpus.sh`

Processed trees are written to the following files:


| File in `data/ctb_5.1/` | Description                                         |
| ----------------------- | --------------------------------------------------- |
| `ctb.train`             | The standard training split for Chinese constituency parsing publications. |
| `ctb.dev`               | The standard development split for Chinese constituency parsing publications. |
| `ctb.test`              | The standard test split for Chinese constituency parsing publications. |


## Multilingual: SPMRL 2013/2014 Shared Task Data

The SPMRL shared tasks used treebanks for the following languages: Arabic, Basque, French, German, Hebrew, Hungarian, Korean, Polish, Swedish.

### Instructions

1. Copy or symlink the various SPMRL folders (`ARABIC_SPMRL`, `BASQUE_SPMRL`, etc.) into `data/raw/`.
2. If you do not have access to the SPMRL data for Arabic, we provide an alternative and compatible preprocessing pipeline that starts directly with data from the LDC. To use this pipeline, place a copy of the Arabic Treebank Parts 1-3 ([LDC2010T13](https://catalog.ldc.upenn.edu/LDC2010T13), [LDC2011T09](https://catalog.ldc.upenn.edu/LDC2011T09), and [LDC2010T08](https://catalog.ldc.upenn.edu/LDC2010T08)) in `data/raw/atb1_v4_1`, `data/raw/atb_2_3.1`, and `data/raw/atb3_v3_2`. The LDC data will be used automatically if the `data/raw/ARABIC_SPMRL` folder is not found.
2. Ensure that the active version of Python is Python 3 and has `nltk` installed.
3. `cd data/spmrl && ./build_corpus.sh`

| Files in `data/spmrl/`       | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| `Arabic.{train,dev,test}`    | Arabic data for the SPMRL 2013/2014 Shared Tasks    |
| `Basque.{train,dev,test}`    | Basque data for the SPMRL 2013/2014 Shared Tasks    |
| `French.{train,dev,test}`    | French data for the SPMRL 2013/2014 Shared Tasks    |
| `German.{train,dev,test}`    | German data for the SPMRL 2013/2014 Shared Tasks    |
| `Hebrew.{train,dev,test}`    | Hebrew data for the SPMRL 2013/2014 Shared Tasks    |
| `Hungarian.{train,dev,test}` | Hungarian data for the SPMRL 2013/2014 Shared Tasks |
| `Korean.{train,dev,test}`    | Korean data for the SPMRL 2013/2014 Shared Tasks    |
| `Polish.{train,dev,test}`    | Polish data for the SPMRL 2013/2014 Shared Tasks    |
| `Swedish.{train,dev,test}`   | Swedish data for the SPMRL 2013/2014 Shared Tasks   |

### Preprocessing notes for Arabic

The Arabic data used for the SPMRL shared tasks is licensed from the LDC, and it has at times been less available from other sources in SPMRL-preprocessed form. Fortunately, it is possible to use scripts that run SPMRL-compatible processing starting with the LDC sources. Our `build_corpus.sh` script will attempt this if the `ARABIC_SPMRL` folder is not found. The resulting tree files should be interchangeable with the original shared task data for the purposes of running EVALB.

Arabic constituency trees are from the Penn Arabic Treebank (PATB), parts 1, 2, and 3. The dataset splits follow [Diab et al. (2013)](https://arxiv.org/abs/1309.5652). The [NYUAD Arabic UD treebank](https://github.com/UniversalDependencies/UD_Arabic-NYUAD) has dependency trees over the exact same sentences with the exact same splits, and the dataset preparation scripts will download the UD data to get the right splits.

The SPMRL-compatible preprocessing differs from the LDC version of ATB in the following ways:
1. 27 training sentences, 1 development sentence, and 4 test sentences are removed.
2. All sentences must have a syntactic label for the constituent that spans the full sentence. The ATB tree files are formatted to have one sentence per line, but there are lines that contain a series of sub-trees with no root node spanning them. In these cases, the SPMRL pre-processing inserted an `XP` node spanning the full sentence.
3. Unary chains with X-over-X productions are collapsed (e.g. the two `NP`s in `(NP (NP ...))` are combined). Nodes are only merged if they have identical labels including functional annotations.
4. In the Buckwalter transliteration, all uses of the symbol "{" in the LDC version are replaced with the symbol "A".
5. Note: the order of the sentences is not the same as in the NYUAD Arabic UD treebank, but the splits are the same excepting the deletion of some sentences per bullet point 1.

Our SPMRL-compatible preprocessing has a few minor differences from the actual data distributed as part of the SPMRL shared tasks. These discrepancies have no effect on the behavior of EVALB, so we have not gone to the effort of resolving them.
1. The files include only the original part-of-speech tags from ATB, and do not additionally convert to any other tagset
2. Our pre-processing scripts directly generate tree files without functional annotations, which are not used for evaluation. In contrast, the trees in the `ARABIC_SPMRL` include functional annotations -- but our parser strips these during preprocessing, anyway.
