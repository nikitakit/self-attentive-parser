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
