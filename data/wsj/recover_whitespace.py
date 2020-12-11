import glob
import os

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


train_splits = ["0" + str(i) for i in range(2, 10)] + [str(i) for i in range(10, 22)]
test_splits = ["23"]
dev_22_splits = ["22"]
dev_24_splits = ["24"]


def glob_raw_files(treebank_root, splits):
    # Get all applicable raw files
    results = [fname for split in splits for fname in sorted(
        glob.glob(os.path.join(treebank_root, 'raw', 'wsj', split, "wsj_????")))]
    # Exclude raw files with no corresponding parse
    mrg_results = [fname.replace('parsed/mrg/wsj', 'raw/wsj').replace('.mrg', '')
        for split in splits for fname in sorted(
        glob.glob(os.path.join(treebank_root, 'parsed', 'mrg', 'wsj', split, "wsj_????.mrg")))]
    return [fname for fname in results if fname in mrg_results]


def glob_tree_files(target_root, splits):
    return [fname for split in splits for fname in sorted(
        glob.glob(os.path.join(target_root, split, "wsj_????.tree"))
        + glob.glob(os.path.join(target_root, 'parsed', 'mrg', 'wsj', split, "wsj_????.mrg")))]


def standardize_form(word):
    word = word.replace('\\/', '/').replace('\\*', '*')
    # Mid-token punctuation occurs in biomedical text
    word = word.replace('-LSB-', '[').replace('-RSB-', ']')
    word = word.replace('-LRB-', '(').replace('-RRB-', ')')
    word = word.replace('-LCB-', '{').replace('-RCB-', '}')
    word = TOKEN_MAPPING.get(word, word)
    return word


def get_raw_text_for_trees(treebank_root, splits, tree_files):
    lines = []
    for fname in glob_raw_files(treebank_root, splits):
        with open(fname, 'r', encoding="windows-1252") as f:
            for line in f:
                if line.strip() and not line.startswith('.START'):
                    # Delete invalid gcharacters caused by encoding issues
                    line = line.replace("Õ", "").replace("å", "")
                    lines.append(line)
    
    reader = BracketParseCorpusReader('.', tree_files)
    target_sents = reader.sents()

    line_iter = iter(lines)
    line = ""
    pairs = []
    for target_sent in target_sents:
        if not line.strip():
            line = next(line_iter)

        # Handle PTB-style escaping mismatches
        target_sent = [standardize_form(word) for word in target_sent]

        # Handle transpositions: sometimes the raw text transposes punctuation,
        # while the parsed version cleans up this transposition
        if 'U.S..' in ''.join(target_sent):
            target_sent = [x.replace('U.S.', 'U.S') for x in target_sent]
        if 'Co.,' in ''.join(target_sent) and 'Co,.' in line:
            target_sent = [x.replace('Co.', 'Co') for x in target_sent]
        if "But that 's" in ' '.join(target_sent) and "But's that" in line:
            target_sent = [x.replace("that", "tha") for x in target_sent]
            target_sent = [x.replace("'s", "t") for x in target_sent]
        if ('-- Freshman football player' in line
            or '-- Sophomore football player' in line
            or '-- Junior football player' in line
            or '-- Senior football player' in line
            or '-- Graduate-student football player' in line
            or '-- Football player' in line
            or '-- Freshman basketball player' in line
            or '-- Sophomore basketball player' in line
            or '-- Junior basketball player' in line
            or '-- Senior basketball player' in line
            or '-- Basketball player' in line) and (
                '" .' in ' '.join(target_sent)
                and target_sent[-1] == '.'):
            target_sent = target_sent[:-1]

        # Attempt to align raw and parsed text
        r2p, p2r = tokenizations.get_alignments(line.replace("`", "'"), target_sent)

        # Handle skips: some lines in the raw data are not parsed
        while not all(p2r):
            go_next = False
            if line.startswith('(See') and '-- WSJ' in line:
                go_next = True
            elif line == 'San Diego ':
                go_next = True
            elif line == '" ':
                go_next = True
            if go_next:
                line = next(line_iter)
                r2p, p2r = tokenizations.get_alignments(line.replace("`", "'"), target_sent)
            else:
                break

        # Handle line breaks in raw format that come in the middle of the sentence
        # (such as mid-sentence line breaks in poems)
        for _ in range(12):  # Loop limit is to aid in debugging
            if not all(p2r):
                line = line + next(line_iter)
                r2p, p2r = tokenizations.get_alignments(line.replace("`", "'"), target_sent)

        assert all(p2r)
        end = max([max(x) for x in p2r]) + 1

        # Trim excess raw text at the start
        line_to_save = line[:end]
        r2p, p2r = tokenizations.get_alignments(line_to_save.replace("`", "'"), target_sent)
        while True:
            _, alt_p2r = tokenizations.get_alignments(
                '\n'.join(line_to_save.replace("`", "'").splitlines()[1:]), target_sent)
            if sum([len(x) for x in p2r]) == sum([len(x) for x in alt_p2r]):
                line_to_save = '\n'.join(line_to_save.splitlines()[1:])
            else:
                break

        pairs.append((line_to_save, target_sent))
        line = line[end:]
    
    assert len(pairs) == len(target_sents)
    return [line for (line, target_sent) in pairs]


def get_words_and_whitespace(treebank_root, splits, tree_files):
    reader = BracketParseCorpusReader('.', tree_files)
    target_sents = reader.sents()
    raw_sents = get_raw_text_for_trees(treebank_root, splits, tree_files)

    pairs = []
    for line, target_sent in zip(raw_sents, target_sents):
        # Fix some errors in the raw text that are also fixed in the parsed trees
        if "But's that just" in line:
            line = line.replace("But's that just", "But that's just")
        if 'Co,.' in line:
            line = line.replace('Co,.', 'Co.,')
        if 'U.S..' in ''.join(target_sent):
            # Address cases where underlying "U.S." got tokenized as "U.S." ".""
            # This is expected in the sentence-final position, but it seems to
            # occur in other places, too.
            line = line.replace('U.S.', 'U.S..').replace(
                'U.S.. market', 'U.S. market').replace(
                'U.S.. agenda', 'U.S. agenda').replace(
                'U.S.. even', 'U.S. even').replace(
                'U.S.. counterpart', 'U.S. counterpart').replace(
                'U.S.. unit', 'U.S. unit').replace(
                'U.S..,', 'U.S.,')
        words = target_sent[:]
        target_sent = [standardize_form(word).replace("``", '"') for word in target_sent]

        r2p, p2r = tokenizations.get_alignments(line.replace("`", "'"), target_sent)

        last_char_for_parsed = [max(x) if x else None for x in p2r]
        have_space_after = [None] * len(words)
        for i, word in enumerate(target_sent):
            if last_char_for_parsed[i] is None:
                continue
            char_after_word = line[last_char_for_parsed[i]+1:last_char_for_parsed[i]+2]
            have_space_after[i] = (char_after_word != char_after_word.lstrip())

            # Fix the few cases where the word form in the parsed data is incorrect
            if word == "'T-" and target_sent[i+1] == 'is':
                target_sent[i] = "'T"
            if word == "16" and target_sent[i+1:i+5] == ['64', '-', 'inch', 'opening']:
                # This error occurs in the test set, and moreover would affect
                # tokenization by introducing an extra '/', so we don't fix it.
                # target_sent[i] = "16/"
                have_space_after[i] = True
            if word == "Gaming"and target_sent[i-1:i+2] == ['and', 'Gaming', 'company']:
                target_sent[i] = "gaming"
        pairs.append((target_sent, have_space_after))

        # For each token in the treebank, we have now queried the raw string to
        # determine if the token should have whitespace following it. The lines
        # below are a sanity check that the reconstructed text matches the raw
        # version as closely as possible.
        to_delete = set()
        for indices in p2r:
            if not indices:
                continue
            to_delete |= set(range(min(indices), max(indices)+1)) - set(indices)
        raw = list(line)
        for i in sorted(to_delete, reverse=True):
            del raw[i]
        raw = "".join(raw)
        raw = " ".join(x.strip() for x in raw.split())

        guess = "".join(
            [w + (" " if sp else "") for (w, sp) in zip(target_sent, have_space_after)])

        if "filings policy-making" in guess:
            # The parsed version of this sentence drops an entire span from the raw
            # text. Maybe we shouldn't be training on this bad example, but for now
            # we'll just skip validating it.
            continue

        # Fix some issues with the raw text that are corrected in the parsed version
        raw = raw.replace("`", "'")
        raw = raw.replace("and <Tourism", "and Tourism")
        raw = raw.replace("staf reporter", "staff reporter")
        if " S$" in raw and " S$" not in guess:
            raw = raw.replace(" S$", " US$")
        raw = raw.replace("16/ 64-inch opening", "16 64-inch opening")
        if raw != guess and raw.replace('."', '".') == guess:
            raw = raw.replace('."', '".')

        # assert raw == guess
        if raw != guess:
            print(raw)
            print(guess)
            print()
    
    return pairs


def get_id_list(target_root, splits):
    res = []
    for fname in glob_tree_files(target_root, splits):
        reader = BracketParseCorpusReader('.', [fname])
        num_sents = len(reader.parsed_sents())
        doc_id = os.path.splitext(os.path.split(fname)[-1])[0]
        for sent_id in range(num_sents):
            sent_id = "{}_{:03}".format(doc_id, sent_id)
            res.append((doc_id, sent_id))
    return res


def write_to_file(treebank3_root, target_root, splits, tree_file, outfile):
    words_and_whitespace = get_words_and_whitespace(treebank3_root, splits, [tree_file])
    doc_and_sent_ids = get_id_list(target_root, splits)
    # print(len(words_and_whitespace), len(doc_and_sent_ids))
    assert len(words_and_whitespace) == len(doc_and_sent_ids)

    with open(outfile, 'w') as f:
        old_doc_id = None
        for (doc_id, sent_id), (words, have_space_after) in zip(
                doc_and_sent_ids, words_and_whitespace):
            if doc_id != old_doc_id:
                old_doc_id = doc_id
                f.write("# newdoc_id = {}\n".format(doc_id))
            f.write("# sent_id = {}\n".format(sent_id))
            text = "".join(
                [w + (" " if sp else "") for w, sp in zip(words, have_space_after)])
            f.write("# text = {}\n".format(text))
            for word_id, (w, sp) in enumerate(zip(words, have_space_after), start=1):
                if sp:
                    misc = "_"
                else:
                    misc = "SpaceAfter=No"
                f.write("{}\t{}\t{}\n".format(word_id, w, misc))
            f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--treebank3_root", required=True)
    parser.add_argument("--revised_root")
    args = parser.parse_args()

    write_to_file(args.treebank3_root, args.treebank3_root, train_splits, 'train_02-21.LDC99T42', 'train_02-21.LDC99T42.text')
    write_to_file(args.treebank3_root, args.treebank3_root, test_splits, 'test_23.LDC99T42', 'test_23.LDC99T42.text')
    write_to_file(args.treebank3_root, args.treebank3_root, dev_22_splits, 'dev_22.LDC99T42', 'dev_22.LDC99T42.text')
    if args.revised_root is not None:
        write_to_file(args.treebank3_root, args.revised_root, train_splits, 'train_02-21.LDC2015T13', 'train_02-21.LDC2015T13.text')
        write_to_file(args.treebank3_root, args.revised_root, test_splits, 'test_23.LDC2015T13', 'test_23.LDC2015T13.text')
        write_to_file(args.treebank3_root, args.revised_root, dev_22_splits, 'dev_22.LDC2015T13', 'dev_22.LDC2015T13.text')
