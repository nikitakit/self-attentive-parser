
import os, sys, json, time
import benepar

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('Loading model...')
parser = benepar.Parser("cn_roberta_aux")
#parser = benepar.Parser("/data2/lfsong/exp.parsing/servc.chinese/cn_roberta_aux")

print('Decoding...')
fout = open('test_pred.txt', 'w')
count = 0
st = time.time()
for line in open(sys.argv[1], 'r'):
    tree = parser.parse(line.strip().split())
    fout.write(str(tree)+'\n')
    count += 1

print('Decoding time for {} sentences: {}'.format(count, time.time() - st))
fout.close()

