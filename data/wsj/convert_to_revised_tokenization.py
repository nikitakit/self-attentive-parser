"""
Converts English parse trees from treebank_3 (LDC99T42) tokenization to revised
(LDC2015T13) tokenization.

The description of LDC2015T13 states that it aims to "bring the full WSJ
treebank section into compliance with the agreed-upon policies and updates
implemented for current English treebank annotation specifications at LDC.
Examples include English Web Treebank (LDC2012T13), OntoNotes (LDC2013T19), and
English translation treebanks such as English Translation Treebank: An-Nahar
Newswire (LDC2012T02)." (Source: https://catalog.ldc.upenn.edu/LDC2015T13)

In practice, the revised tokenizatation mostly breaks up hyphenated words into
multiple tokens.

This script takes labelled brackets from trees over the original Penn Treebank
and overlays them on top of token sequences from the revised treebank. The goal
is to maintain the invariant that EVALB(gold, guess) gives the same output as
EVALB(convert_to_revised_tokenization(gold),
        convert_to_revised_tokenization(guess)).
We observe this to be true with the following minor exceptions:
    (a) Sentence lengths (as measured by EVALB) are different depending on the
        tokenization, so only the bracketing statistics for full-length trees
        should remain unchanged. Statistics for short sentences (the last set
        of lines printed by EVALB) will not match, and neither will the tagging
        accuracy.
    (b) There is one tree in the training data that is present in the original
        treebank but omitted from the revised treebank. This script skips
        converting that sentence if it detects it.
"""

from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
import nltk
import tokenizations  # pip install pytokenizations==0.7.2


TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    ''': "'",
    ''': "'",
    '"': '"',
    '"': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}


def standardize_form(word):
    word = word.replace('\\/', '/').replace('\\*', '*')
    # Mid-token punctuation occurs in biomedical text
    word = word.replace('-LSB-', '[').replace('-RSB-', ']')
    word = word.replace('-LRB-', '(').replace('-RRB-', ')')
    word = word.replace('-LCB-', '{').replace('-RCB-', '}')
    word = TOKEN_MAPPING.get(word, word)
    return word


# Rather than writing tree traversal code to delete nodes when needed during
# conversion, it's simpler to output a node label that EVALB is known to
# ignore. I (Nikita) originally tried using -NONE-, but this didn't seem to
# work while TOP does. Deletion only affects punctuation, which EVALB ignores,
# so I did not observe this causing bracketing changes during conversion.
DUMMY_LABEL = 'TOP'
DUMMY_WORD = '*'

def convert_to_revised_tokenization(orig_trees, revised_trees):
    for orig_tree, revised_tree in zip(orig_trees, revised_trees):
        orig_words = [standardize_form(word) for word in orig_tree.leaves()]
        revised_words = [standardize_form(word) for word in revised_tree.leaves()]
        o2r, r2o = tokenizations.get_alignments(orig_words, revised_words)
        assert all(len(x) >= 1 for x in o2r)
        
        converted_tree = orig_tree.copy(deep=True)
        for j in range(len(revised_words)):
            if len(r2o[j]) > 1:
                for i in r2o[j][1:]:
                    orig_treeposition = orig_tree.leaf_treeposition(i)
                    if len(orig_treeposition) > 1 and len(
                        orig_tree[orig_treeposition[:-1]]) == 1:
                        converted_tree[orig_treeposition[:-1]] = nltk.Tree(
                            DUMMY_LABEL, [DUMMY_WORD])
                    else:
                        converted_tree[orig_treeposition] = DUMMY_LABEL

        for i in range(len(orig_words)):
            if converted_tree[orig_tree.leaf_treeposition(i)] == DUMMY_LABEL:
                continue
            elif len(o2r[i]) == 1:
                j = o2r[i][0]
                converted_tree[orig_tree.leaf_treeposition(i)] = revised_tree[
                    revised_tree.leaf_treeposition(j)]
            else:
                orig_treeposition = orig_tree.leaf_treeposition(i)
                if len(orig_treeposition) > 1 and len(
                        orig_tree[orig_treeposition[:-1]]) == 1:
                    orig_treeposition = orig_treeposition[:-1]
                    revised_leaves = [revised_tree[revised_tree.leaf_treeposition(j)[:-1]] for j in o2r[i]]
                    assert all(len(x) == 1 for x in revised_leaves)
                    converted_tree[orig_treeposition] = nltk.Tree(
                        DUMMY_LABEL,
                        revised_leaves
                        )
                else:
                    converted_tree[orig_treeposition] = nltk.Tree(
                        DUMMY_LABEL,
                        [revised_tree[revised_tree.leaf_treeposition(j)] for j in o2r[i]])

        yield converted_tree


def write_to_file(orig_file, revised_file, outfile=None, no_heuristic_mismatch_fix=False):
    reader = BracketParseCorpusReader('.', [])
    orig_trees = reader.parsed_sents(orig_file)
    revised_trees = reader.parsed_sents(revised_file)

    # The revised PTB parses have one less tree in the training split.
    # This attempts to patch the problem by skipping this tree.
    if not no_heuristic_mismatch_fix:
        orig_trees = list(orig_trees)
        revised_trees = list(revised_trees)
        if len(orig_trees) == 39832 and len(revised_trees) == 39831:
            del orig_trees[4906]

    converted_trees = convert_to_revised_tokenization(
        orig_trees, revised_trees)

    if outfile is None:
        for tree in converted_trees:
            print(tree.pformat(margin=1e100))
    else:
        with open(outfile, 'w') as f:
            for tree in converted_trees:
                tree_rep = tree.pformat(margin=1e100)
                assert('\n' not in tree_rep)
                f.write(tree_rep)
                f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True)
    parser.add_argument("--revised", required=True)
    parser.add_argument("--out")
    parser.add_argument("--no-heuristic-mismatch-fix", action='store_true')

    args = parser.parse_args()

    write_to_file(args.orig, args.revised, args.out, args.no_heuristic_mismatch_fix)
