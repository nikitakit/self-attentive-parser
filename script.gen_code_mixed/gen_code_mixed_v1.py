
import os, sys, json, random
from collections import defaultdict
from nltk.tree import Tree


def recur_get_phrase(root, phrase_dict, threshold):
    if isinstance(root, Tree) == False:
        return 1
    width = 0
    for child in root:
        width += recur_get_phrase(child, phrase_dict, threshold)
    if width <= threshold:
        phrase_dict[root.label()].add(str(root))
    return width


def read_treebank(path):
    treebank_str = []
    phrase_dict = defaultdict(set)
    for line in open(path, 'r'):
        line = line.strip()
        treebank_str.append(line)
        recur_get_phrase(Tree.fromstring(line), phrase_dict, 5)
    phrase_dict = {k:[Tree.fromstring(x) for x in v] for k, v in phrase_dict.items()}
    return treebank_str, phrase_dict


def recur_replace_phrase(parent, cur, cid, phrase_dict, threshold):
    if isinstance(cur, Tree) == False:
        return
    width = len(cur.leaves())
    if parent != None and width <= threshold: # root or too large
        phrases = phrase_dict.get(cur.label(), [])
        if random.random() < 0.5 and len(phrases) > 0:
            #print(tree_to_str(parent[cid]))
            parent[cid] = phrases[random.randint(0, len(phrases) - 1)]
            #print(tree_to_str(parent[cid]))
            #print('=====')
    else:
        for i in range(len(cur)):
            recur_replace_phrase(cur, cur[i], i, phrase_dict, threshold)


def tree_to_str(tree):
    tree_str = str(tree).replace('\n', ' ')
    tree_str =  ' '.join(tree_str.split())
    return tree_str


ptb_treebank_str, ptb_phrase_dict = read_treebank('../data.ptb/train.gold.stripped')
print('PTB: {} trees'.format(len(ptb_treebank_str)))
genia_treebank_str, genia_phrase_dict = read_treebank('../data.genia/train.trees')
print('GENIA: {} trees'.format(len(genia_treebank_str)))

new_ptb_treebank_str = []
for tree_str in ptb_treebank_str:
    tree = Tree.fromstring(tree_str)
    recur_replace_phrase(None, tree, -1, genia_phrase_dict, 5)
    new_ptb_treebank_str.append(tree_to_str(tree))

new_genia_treebank_str = []
for tree_str in genia_treebank_str:
    tree = Tree.fromstring(tree_str)
    recur_replace_phrase(None, tree, -1, ptb_phrase_dict, 5)
    new_genia_treebank_str.append(tree_to_str(tree))

final_treebank_str = ptb_treebank_str + genia_treebank_str + new_ptb_treebank_str + new_genia_treebank_str
random.shuffle(final_treebank_str)
f = open('train_orig_plus_code_mix.trees', 'w')
for tree_str in final_treebank_str:
    f.write(tree_str+'\n')
f.close()

