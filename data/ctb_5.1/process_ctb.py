# -*- coding:utf-8 -*-
# Filename: make_ctb.py
# Author：hankcs
# Date: 2017-11-03 21:23

# modified from https://github.com/hankcs/TreebankPreprocessing
import argparse
from os import listdir
from os.path import isfile, join, isdir

import nltk
import glob

import errno
from os import makedirs

import sys

def make_sure_path_exists(path):
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def combine_files(fids, out, tb):
    print('%d files...' % len(fids))
    total_sentence = 0
    for n, file in enumerate(fids):
        if n % 10 == 0 or n == len(fids) - 1:
            print("%c%.2f%%" % (13, (n + 1) / float(len(fids)) * 100), end='')
        sents = tb.parsed_sents(file)
        for s in sents:
            out.write(s.pformat(margin=sys.maxsize))
            out.write('\n')
            total_sentence += 1
    print()
    print('%d sentences.' % total_sentence)
    print()


def convert(ctb_root,
            out_root,
            encoding='GB2312',
            filter_fn=lambda f: f.endswith('.fid')):
    ctb_root = join(ctb_root, 'bracketed')
    fids = [f for f in listdir(ctb_root) if isfile(join(ctb_root, f)) and filter_fn(f)]
    make_sure_path_exists(out_root)
    for f in fids:
        fname = join(ctb_root, f)
        with open(fname, encoding=encoding) as src, open(join(out_root, f), 'w') as out:
            try:
                for line in src:
                    if not (line.strip().startswith('<') and line.strip().endswith('>')):
                        out.write(line)
                        # if any(tok == 'CP' for tok in line.split()):
                        #     raise Exception("{}: {}".format(f, line))
                # in_s_tag = False
                # for line in src:
                #     if line.startswith('<S ID='):
                #         in_s_tag = True
                #     elif line.startswith('</S>'):
                #         in_s_tag = False
                #     elif in_s_tag:
                #         out.write(line)
            except Exception as e:
                print("error for file {}".format(fname))
                print(e)
                pass


def combine_fids(treebank, root_path, fids, out_path, suffix_glob='fid', pad_hundreds=False):
    print('Generating ' + out_path)
    files = []
    for fid in fids:
        if not pad_hundreds and fid < 1000:
            f = 'chtb_%03d.%s' % (fid, suffix_glob)
        else:
            f = 'chtb_%04d.%s' % (fid, suffix_glob)
        matching_files = glob.glob(join(root_path, f))
        assert len(matching_files) <= 1
        if len(matching_files) == 0:
            print("no file found for id %s" % fid)
            continue
        files.append(matching_files[0])
    with open(out_path, 'w') as out:
        combine_files(files, out, treebank)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine Chinese Treebank 5.1 fid files into train/dev/test set')
    parser.add_argument("--ctb", required=True,
                        help='The root path to Chinese Treebank 5.1')
    # parser.add_argument("--output", required=True,
    #                     help='The folder where to store the output train.txt/dev.txt/test.txt')

    args = parser.parse_args()

    ctb_in_nltk = None
    for root in nltk.data.path:
        if isdir(root):
            ctb_in_nltk = root

    if ctb_in_nltk is None:
        eprint('You should run nltk.download(\'ptb\') to fetch some data first!')
        exit(1)

    ctb_in_nltk = join(ctb_in_nltk, 'corpora')
    ctb_in_nltk = join(ctb_in_nltk, 'ctb')

    print('Converting CTB: removing xml tags...')
    convert(args.ctb, ctb_in_nltk)
    print('Importing to nltk...\n')
    from nltk.corpus import BracketParseCorpusReader, LazyCorpusLoader

    ctb = LazyCorpusLoader(
        'ctb', BracketParseCorpusReader, r'chtb_.*\.fid',
        tagset='unknown')

    # commented out splits are typically for dependency parsing, e.g. Zhang and Clark 2008
    # training = list(range(1, 815 + 1)) + list(range(1001, 1136 + 1))
    # development = list(range(886, 931 + 1)) + list(range(1148, 1151 + 1))
    # test = list(range(816, 885 + 1)) + list(range(1137, 1147 + 1))

    # splits for constituency parsing, see Petrov and Klein 2007, Liu and Zhang 2017
    training = list(range(1,270+1)) + list(range(440,1151+1))
    development = list(range(301, 325+1))
    test = list(range(271,300+1))

    # make sure there's no overlap
    assert not (set(training) & set(development))
    assert not (set(training) & set(test))
    assert not (set(development) & set(test))

    print("total num files: %d" % len(training + development + test))

    combine_fids(ctb, ctb_in_nltk, training, 'ctb.train.withtraces')
    combine_fids(ctb, ctb_in_nltk, development, 'ctb.dev.withtraces')
    combine_fids(ctb, ctb_in_nltk, test, 'ctb.test.withtraces')
