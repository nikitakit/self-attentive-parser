import re
import fileinput
import argparse

#!/usr/bin/env python

from collections import defaultdict

def remove_symbol_functionals(symbol):
    if symbol[0] == '-' and symbol[-1] == '-':
        return symbol
    morph_split = symbol.split('##')
    morph_split[0] = morph_split[0].split('-')[0]
    morph_split[0] = morph_split[0].split('=')[0]
    return '##'.join(morph_split)

def _strip_tag_suffix(tag, sep='#'):
    assert len(sep) == 1
    ix = tag.find('#')
    return tag[:ix]

class PhraseTree(object):
# adapted from https://github.com/jhcross/span-parser/blob/master/src/phrase_tree.py

    puncs = [",", ".", ":", "``", "''", "PU"] ## (COLLINS.prm)


    def __init__(
        self,
        symbol=None,
        children=[],
        sentence=[],
        leaf=None,
    ):
        self.symbol = symbol        # label at top node
        self.children = children    # list of PhraseTree objects
        self.sentence = sentence
        self.leaf = leaf            # word at bottom level else None

        self._str = None
        self._left_span = None
        self._right_span = None

    def left_span(self):
        if self.leaf is not None:
            return self.leaf
        elif self._left_span is not None:
            return self._left_span
        else:
            assert self.children
            self._left_span = self.children[0].left_span()
            return self._left_span

    def right_span(self):
        if self.leaf is not None:
            return self.leaf + 1
        elif self._right_span is not None:
            return self._right_span
        else:
            assert self.children
            self._right_span = self.children[-1].right_span()
            return self._right_span

    def remove_nodes(self, symbol_list):
        children = []
        for child in self.children:
            children.extend(child.remove_nodes(symbol_list))
        if self.symbol in symbol_list:
            return children
        else:
            return [PhraseTree(self.symbol, children, self.sentence, leaf=self.leaf)]


    def _zpar_contraction_spans(self):
        if '#' in self.symbol:
            return [(self.left_span(), self.right_span())]
        else:
            spans = []
            for c in self.children:
                spans += c._zpar_contraction_spans()
            return spans

    def _zpar_contract(self, ix, is_root, sentence):
        if '#' in self.symbol:
            assert not is_root
            assert self.symbol.endswith('#t')
            stripped = _strip_tag_suffix(self.symbol)
            rep_node = PhraseTree(stripped, [], sentence, ix)
            ix += 1
        else:
            children = []
            for child in self.children:
                rep_child, ix = child._zpar_contract(ix, False, sentence)
                children.append(rep_child)
            rep_node = PhraseTree(self.symbol, children, sentence, None)
        return rep_node, ix

    def zpar_contract(self):
        contraction_spans = self._zpar_contraction_spans()
        contracted = []
        for (start, end) in contraction_spans:
            words = [self.sentence[i][0] for i in range(start, end)]
            tags = [self.sentence[i][1] for i in range(start, end)]
            assert tags[0].endswith("#b")
            stripped_tags = [_strip_tag_suffix(tag) for tag in tags]
            assert all(st == stripped_tags[0] for st in stripped_tags)
            contracted.append((''.join(words), stripped_tags[0]))
        node, ix = self._zpar_contract(0, True, contracted)
        assert ix == len(contracted)
        return node

    def remove_tag_tokens(self, tok_tag_pred):
        # this doesn't remove tokens from sentence; just drops them from the tree. but so long as sentence is refered to by indices stored in PhraseTree.leaf, should be ok
        children = []
        for child in self.children:
            if child.leaf is not None and tok_tag_pred(self.sentence[child.leaf]):
                continue
            else:
                children.append(child.remove_tag_tokens(tok_tag_pred))
        return PhraseTree(self.symbol, children, self.sentence, leaf=self.leaf)

    def __str__(self):
        if self._str is None:
            if len(self.children) != 0:
                childstr = ' '.join(str(c) for c in self.children)
                self._str = '({} {})'.format(self.symbol, childstr)
            else:
                self._str = '({} {})'.format(
                    self.sentence[self.leaf][1],
                    self.sentence[self.leaf][0],
                )
        return self._str

    def pretty(self, level=0, marker='  '):
        pad = marker * level

        if self.leaf is not None:
            leaf_string = '({} {})'.format(
                    self.symbol,
                    self.sentence[self.leaf][0],
            )
            return pad + leaf_string

        else:
            result = pad + '(' + self.symbol
            for child in self.children:
                result += '\n' + child.pretty(level + 1)
            result += ')'
            return result


    @staticmethod
    def parse(line):
        """
        Loads a tree from a tree in PTB parenthetical format.
        """
        line += " "
        sentence = []
        ix, t = PhraseTree._parse(line, 0, sentence)
        assert not line[ix:].strip(), "suffix remaining: {}".format(line[ix:].strip())

        return t


    @staticmethod
    def _parse(line, index, sentence):

        "((...) (...) w/t (...)). returns pos and tree, and carries sent out."

        assert line[index] == '(', "Invalid tree string {} at {}".format(line, index)
        index += 1
        symbol = None
        children = []
        leaf = None
        while line[index] != ')':
            if line[index] == '(':
                index, t = PhraseTree._parse(line, index, sentence)
                if t is not None:
                    children.append(t)
            else:
                if symbol is None:
                    # symbol is here!
                    rpos = min(line.find(' ', index), line.find(')', index))
                    # see above N.B. (find could return -1)

                    symbol = line[index:rpos] # (word, tag) string pair

                    index = rpos
                else:
                    rpos = line.find(')', index)
                    word = line[index:rpos]
                    if symbol != '-NONE-':
                        sentence.append((word, remove_symbol_functionals(symbol)))
                        leaf = len(sentence) - 1
                    index = rpos

            if line[index] == " ":
                index += 1

        assert line[index] == ')', "Invalid tree string %s at %d" % (line, index)

        if symbol == '-NONE-' or (children == [] and leaf is None):
            t = None
        else:
            t = PhraseTree(
                symbol=remove_symbol_functionals(symbol),
                children=children,
                sentence=sentence,
                leaf=leaf,
            )

        return (index + 1), t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_symbols', nargs='*', help="remove these NT symbols from anywhere in the tree")
    parser.add_argument("--remove_root", help="remove this symbol from the root of the tree, if it's there")
    parser.add_argument("--remove_root_must_have", help="remove this symbol from the root of the tree, and throw an error if it's not present")
    parser.add_argument('--root_removed_replacement')
    parser.add_argument('--dedup_punct_symbols', nargs='*')
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    args = parser.parse_args()

    dedup_punct_symbols = set(args.dedup_punct_symbols) if args.dedup_punct_symbols else set()

    for line in fileinput.input(files=args.files if len(args.files) > 0 else ('-', )):
        line = line.strip()
        # line = re.sub('\)', ') ', line)

        # #line = re.sub(r'-[^\s^\)]* |##[^\s^\)]*## ', ' ', line)
        # #line = re.sub(r'##[^\s^\)]*## ', ' ', line)
        # tokens = line.split(' ')
        # proc_tokens = []
        # last_was_none = False
        # for tok in tokens:
        #     if last_was_none:
        #         # follows '(-NONE-', should just be a terminal that looks like *op*, drop it
        #         assert not '(' in tok
        #         assert '*' in tok
        #         assert tok.count(')') == 1 and tok[-1] == ')'
        #         last_was_none = False
        #         continue

        #     if tok.startswith('('):
        #         if '-NONE-' in tok:
        #             last_was_none = True
        #             continue
        #         # remove functional tags -- anything after a - but not preceeded by ##
        #         morph_split = tok.split('##')
        #         morph_split[0] = re.sub('-.*', '', morph_split[0])
        #         tok = '##'.join(morph_split)
        #     proc_tokens.append(tok)

        # linearized = ' '.join(proc_tokens)
        # linearized = re.sub('\) ', ')', linearized)
        # print(linearized)
        tree = PhraseTree.parse(line)
        if args.remove_root is not None or args.remove_root_must_have is not None:
            assert not (args.remove_root is not None and args.remove_root_must_have is not None)
            if args.remove_root_must_have is not None:
                assert tree.symbol == args.remove_root_must_have
            symbol_to_remove = args.remove_root_must_have if args.remove_root_must_have is not None else args.remove_root
            if tree.symbol == symbol_to_remove:
                trees = tree.children
            else:
                trees = [tree]
        else:
            trees = [tree]

        if args.remove_symbols:
            trees = [
                t for tree in trees
                for t in tree.remove_nodes(set(args.remove_symbols))
            ]

        if len(trees) == 1:
            tree = trees[0]
        else:
            if args.root_removed_replacement:
                assert all(t.sentence == trees[0].sentence for t in trees[1:])
                tree = PhraseTree(symbol=args.root_removed_replacement,
                                    children=trees,
                                    sentence=trees[0].sentence,
                                    leaf=None)
            else:
                assert len(trees) == 1, "can't remove a root symbol with multiple children without passing --root_removed_replacement!"

        if dedup_punct_symbols:
            for ix in range(len(tree.sentence)):
                tok, tag = tree.sentence[ix]
                if tag in dedup_punct_symbols:
                    if all(x == tok[0] for x in tok[1:]):
                        tok = tok[0]
                        tree.sentence[ix] = tok, tag
        print(tree)
