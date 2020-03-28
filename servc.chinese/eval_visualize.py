
import os, sys, json, time
import benepar
import svgling
from nltk.tree import Tree


def extract_spans_recur(root, offset, chunks, labels):
    width = 0
    for child in root:
        if isinstance(child, Tree):
            sub_width = extract_spans_recur(child, offset+width, chunks, labels)
            width += sub_width
        else:
            width += 1

    if root.label() in labels:
        assert width == len(root.leaves())
        chunks.append((offset, offset+width))

    return width


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

print('Loading model...')
parser = benepar.Parser("benepar_zh")

line = "刘德华 偶尔 玩 下 由 腾讯 开发 的 王者荣耀 这 款 游戏 。 平时 ， 经常 会 跟 小伙伴 通过 微信 聊聊天 。"
line = line.strip().split()
tree = parser.parse(line)
print(str(tree)+'\n========')

chunks = []
labels = ['NP',]
line_len = extract_spans_recur(tree, 0, chunks, labels)
assert len(line) == line_len
for st, ed in chunks:
    print('\t'+' '.join(line[st:ed]))

#tree_str = ' '.join(str(tree).split())
#print(tree_str)
#tree = Tree.fromstring(tree_str)
#t = svgling.draw_tree(tree)
#t.get_svg().saveas("demo.svg")

