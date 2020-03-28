
import requests
import sys, time, json

url = 'http://localhost:6060/'

print('Decoding...')
fout = open('test_pred.txt', 'w')
count = 0
count_token = 0
st = time.time()
for line in open('test.txt', 'r'):
    response = requests.put(url, data={'segment':line.strip()})
    tree_string = response.json()['tree_string']
    fout.write(tree_string+'\n')
    count += 1
    count_token += len(line.strip().split())

print('Decoding time: {}, sentences: {}, tokens: {}'.format(time.time() - st, count, count_token))
fout.close()

