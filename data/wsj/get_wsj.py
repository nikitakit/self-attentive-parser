from os.path import join
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
import nltk
import glob

train_splits = ["0" + str(i) for i in range(2, 10)] + [str(i) for i in range(10, 22)]
test_splits = ["23"]
dev_22_splits = ["22"]
dev_24_splits = ["24"]

def glob_files(data_root, splits):
    return [
        fname for split in splits for fname in sorted(
            glob.glob(join(data_root, split, "*.mrg"))
            + glob.glob(join(data_root, split, "*.tree"))
            )]

def write_to_file(data_root, splits, outfile, add_top=False):
    reader = BracketParseCorpusReader('.', glob_files(data_root, splits))
    with open(outfile, 'w') as f:
        for tree in reader.parsed_sents():
            tree_rep = tree.pformat(margin=1e100)
            if add_top:
                tree_rep = "(TOP %s)" % tree_rep
            assert('\n' not in tree_rep)
            f.write(tree_rep)
            f.write("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_root")
    parser.add_argument("--revised_root")
    parser.add_argument("--add_top", action='store_true')
    args = parser.parse_args()

    for split in range(25):
        split = "{:02d}".format(24)

    if args.orig_root:
        write_to_file(args.orig_root, train_splits, 'train_02-21.LDC99T42.withtraces', args.add_top)
        write_to_file(args.orig_root, test_splits, 'test_23.LDC99T42.withtraces', args.add_top)
        write_to_file(args.orig_root, dev_22_splits, 'dev_22.LDC99T42.withtraces', args.add_top)
        write_to_file(args.orig_root, dev_24_splits, 'dev_24.LDC99T42.withtraces', args.add_top)
    
    if args.revised_root:
        write_to_file(args.revised_root, train_splits, 'train_02-21.LDC2015T13.withtraces', args.add_top)
        write_to_file(args.revised_root, test_splits, 'test_23.LDC2015T13.withtraces', args.add_top)
        write_to_file(args.revised_root, dev_22_splits, 'dev_22.LDC2015T13.withtraces', args.add_top)
        write_to_file(args.revised_root, dev_24_splits, 'dev_24.LDC2015T13.withtraces', args.add_top)
