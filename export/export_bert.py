"""
This file documents the process that was used to convert our model from a saved
PyTorch checkpoint to a TensorFlow graph. The code here was run one cell at a
time inside an IPython/Jupyter session.

This version of the export script is for models that use BERT.

BERT code for both tensorflow and pytorch is required:
- pytorch_pretrained_bert 0.4.0 (available via pip)
- https://github.com/google-research/bert/commit/f39e881b169b9d53bea03d2d341b31707a6c052b
"""

%cd ~/dev/self-attentive-parser
import sys
sys.path.insert(0, "/Users/kitaev/dev/self-attentive-parser/src")

sys.path.append("/Users/kitaev/dev/bert")
import bert
import bert.modeling, bert.tokenization

import pytorch_pretrained_bert

import argparse
import itertools
import os.path
import time
import shutil
import re
import json

import torch
import torch.optim.lr_scheduler

import numpy as np

import evaluate
import trees
import vocabulary
import nkutil
import parse_nk
tokens = parse_nk
#%%

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

# %%

class args:
    model_path_base="models/nk_base9_scale=5_dev=95.40.pt"
    test_path="data/22.goldtags" # dev set with gold tag (not distributed in this repo)
    eval_batch_size=100
    evalb_dir="EVALB/"

# %%
if True:
    if parse_nk.use_cuda:
        info = torch.load(args.model_path_base)
    else:
        info = torch.load(args.model_path_base, map_location=lambda storage, location: storage)
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])
    bert_model = info['spec']['hparams']['bert_model']
    bert_do_lower_case = info['spec']['hparams']['bert_do_lower_case']

#%%

print("Loading test trees from {}...".format(args.test_path))
test_treebank = trees.load_trees(args.test_path)
print("Loaded {:,} test examples.".format(len(test_treebank)))

#%%

import tensorflow as tf

sess = tf.InteractiveSession()

sd = parser.state_dict()

LABEL_VOCAB = [x[0] for x in sorted(parser.label_vocab.indices.items(), key=lambda x: x[1])]
TAG_VOCAB = [x[0] for x in sorted(parser.tag_vocab.indices.items(), key=lambda x: x[1])]

# %%

def make_bert_instance(input_ids, input_mask, token_type_ids):
    # Transfer BERT config into tensorflow implementation
    config = bert.modeling.BertConfig.from_dict(parser.bert.config.to_dict())
    model = bert.modeling.BertModel(config=config, is_training=False,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    # Next, transfer learned weights (after fine-tuning)
    bert_variables = [v for v in tf.get_collection('variables') if 'bert' in v.name]
    tf.variables_initializer(bert_variables).run()

    # Based on: convert_tf_checkpoint_to_pytorch.py from pytorch-pretrained-BERT
    for variable in bert_variables:
        name = variable.name.split(':')[0]
        name = name.split('/')
        array = variable.eval()
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pytorch_var = parser
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pytorch_var = getattr(pytorch_var, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pytorch_var = getattr(pytorch_var, 'bias')
            elif l[0] == 'output_weights':
                pytorch_var = getattr(pytorch_var, 'weight')
            else:
                pytorch_var = getattr(pytorch_var, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pytorch_var = pytorch_var[num]
        if m_name[-11:] == '_embeddings':
            pytorch_var = getattr(pytorch_var, 'weight')
        elif m_name == 'kernel':
            pytorch_var = pytorch_var.t()
        try:
            assert pytorch_var.shape == array.shape
        except AssertionError as e:
            e.args += (pytorch_var.shape, array.shape)
            raise

        # print("Extracting PyTorch weight {}".format(name))
        variable.load(pytorch_var.detach().cpu().numpy())
    return model

#%%

def make_bert(input_ids, word_end_mask):
    # We can derive input_mask from either input_ids or word_end_mask
    input_mask = (1 - tf.cumprod(1 - word_end_mask, axis=-1, reverse=True))
    token_type_ids = tf.zeros_like(input_ids)
    bert_model = make_bert_instance(input_ids, input_mask, token_type_ids)

    bert_features = bert_model.get_sequence_output()
    bert_features_packed = tf.gather(
        tf.reshape(bert_features, [-1, int(bert_features.shape[-1])]),
        tf.to_int32(tf.where(tf.reshape(word_end_mask, (-1,))))[:,0])
    projected_annotations = tf.matmul(
        bert_features_packed,
        tf.constant(sd['project_bert.weight'].numpy().transpose()))

    # input_mask is over subwords, whereas valid_mask is over words
    sentence_lengths = tf.reduce_sum(word_end_mask, -1)
    valid_mask = (tf.range(tf.reduce_max(sentence_lengths))[None,:] < sentence_lengths[:, None])
    dim_padded = tf.shape(valid_mask)[:2]
    mask_flat = tf.reshape(valid_mask, (-1,))
    dim_flat = tf.shape(mask_flat)[:1]
    nonpad_ids = tf.to_int32(tf.where(mask_flat)[:,0])

    return projected_annotations, nonpad_ids, dim_flat, dim_padded, valid_mask, sentence_lengths

#%%

position_table = tf.constant(sd['embedding.position_table'], name="position_table")

# %%

def make_layer_norm(input, torch_name, name):
    # TODO(nikita): The epsilon here isn't quite the same as in pytorch
    # The pytorch code adds eps=1e-3 to the standard deviation, while this
    # tensorflow code adds eps=1e-6 to the variance.
    # However, the resulting mismatch in floating-point values does not seem to
    # translate to any noticable changes in the parser's tree output
    mean, variance = tf.nn.moments(input, [1], keep_dims=True)
    return tf.nn.batch_normalization(
        input,
        mean, variance,
        offset=tf.constant(sd[f'{torch_name}.b_2'], name=f"{name}/offset"),
        scale=tf.constant(sd[f'{torch_name}.a_2'], name=f"{name}/scale"),
        variance_epsilon=1e-6)


def make_heads(input, shape_bthf, shape_xtf, torch_name, name):
    res = tf.matmul(input,
        tf.constant(sd[torch_name].numpy().transpose((1,0,2)).reshape((512, -1)), name=f"{name}/W"))
    res = tf.reshape(res, shape_bthf)
    res = tf.transpose(res, (0,2,1,3)) # batch x num_heads x time x feat
    res = tf.reshape(res, shape_xtf) # _ x time x feat
    return res

def make_attention(input, nonpad_ids, dim_flat, dim_padded, valid_mask, torch_name, name):
    input_flat = tf.scatter_nd(indices=nonpad_ids[:, None], updates=input, shape=tf.concat([dim_flat, tf.shape(input)[1:]], axis=0))
    input_flat_dat, input_flat_pos = tf.split(input_flat, 2, axis=-1)

    shape_bthf = tf.concat([dim_padded, [8, -1]], axis=0)
    shape_bhtf = tf.convert_to_tensor([dim_padded[0], 8, dim_padded[1], -1])
    shape_xtf = tf.convert_to_tensor([dim_padded[0] * 8, dim_padded[1], -1])
    shape_xf = tf.concat([dim_flat, [-1]], axis=0)

    qs1 = make_heads(input_flat_dat, shape_bthf, shape_xtf, f'{torch_name}.w_qs1', f'{name}/q_dat')
    ks1 = make_heads(input_flat_dat, shape_bthf, shape_xtf, f'{torch_name}.w_ks1', f'{name}/k_dat')
    vs1 = make_heads(input_flat_dat, shape_bthf, shape_xtf, f'{torch_name}.w_vs1', f'{name}/v_dat')
    qs2 = make_heads(input_flat_pos, shape_bthf, shape_xtf, f'{torch_name}.w_qs2', f'{name}/q_pos')
    ks2 = make_heads(input_flat_pos, shape_bthf, shape_xtf, f'{torch_name}.w_ks2', f'{name}/k_pos')
    vs2 = make_heads(input_flat_pos, shape_bthf, shape_xtf, f'{torch_name}.w_vs2', f'{name}/v_pos')

    qs = tf.concat([qs1, qs2], axis=-1)
    ks = tf.concat([ks1, ks2], axis=-1)
    attn_logits = tf.matmul(qs, ks, transpose_b=True) / (1024 ** 0.5)

    attn_mask = tf.reshape(tf.tile(valid_mask, [1,8*dim_padded[1]]), tf.shape(attn_logits))
    # TODO(nikita): use tf.where and -float('inf') here?
    attn_logits -= 1e10 * tf.to_float(~attn_mask)

    attn = tf.nn.softmax(attn_logits)

    attended_dat_raw = tf.matmul(attn, vs1)
    attended_dat_flat = tf.reshape(tf.transpose(tf.reshape(attended_dat_raw, shape_bhtf), (0,2,1,3)), shape_xf)
    attended_dat = tf.gather(attended_dat_flat, nonpad_ids)
    attended_pos_raw = tf.matmul(attn, vs2)
    attended_pos_flat = tf.reshape(tf.transpose(tf.reshape(attended_pos_raw, shape_bhtf), (0,2,1,3)), shape_xf)
    attended_pos = tf.gather(attended_pos_flat, nonpad_ids)

    out_dat = tf.matmul(attended_dat, tf.constant(sd[f'{torch_name}.proj1.weight'].numpy().transpose()))
    out_pos = tf.matmul(attended_pos, tf.constant(sd[f'{torch_name}.proj2.weight'].numpy().transpose()))

    out = tf.concat([out_dat, out_pos], -1)
    return make_layer_norm(input + out, f'{torch_name}.layer_norm', f'{name}/layer_norm')

def make_dense_relu_dense(input, torch_name, torch_type, name):
    # TODO: use name
    mul1 = tf.matmul(input, tf.constant(sd[f'{torch_name}.w_1{torch_type}.weight'].numpy().transpose()))
    mul1b = tf.nn.bias_add(mul1, tf.constant(sd[f'{torch_name}.w_1{torch_type}.bias']))
    mul1b = tf.nn.relu(mul1b)
    mul2 = tf.matmul(mul1b, tf.constant(sd[f'{torch_name}.w_2{torch_type}.weight'].numpy().transpose()))
    mul2b = tf.nn.bias_add(mul2, tf.constant(sd[f'{torch_name}.w_2{torch_type}.bias']))
    return mul2b

def make_ff(input, torch_name, name):
    # TODO: use name
    input_dat, input_pos = tf.split(input, 2, axis=-1)
    out_dat = make_dense_relu_dense(input_dat, torch_name, 'c', name="TODO_dat")
    out_pos = make_dense_relu_dense(input_pos, torch_name, 'p', name="TODO_pos")
    out = tf.concat([out_dat, out_pos], -1)
    return make_layer_norm(input + out, f'{torch_name}.layer_norm', f'{name}/layer_norm')

def make_stacks(input, nonpad_ids, dim_flat, dim_padded, valid_mask, num_stacks):
    res = input
    for i in range(num_stacks):
        res = make_attention(res, nonpad_ids, dim_flat, dim_padded, valid_mask, f'encoder.attn_{i}', name=f'attn_{i}')
        res = make_ff(res, f'encoder.ff_{i}', name=f'ff_{i}')
    return res

def make_layer_norm_with_constants(input, constants):
    # TODO(nikita): The epsilon here isn't quite the same as in pytorch
    # The pytorch code adds eps=1e-3 to the standard deviation, while this
    # tensorflow code adds eps=1e-6 to the variance.
    # However, the resulting mismatch in floating-point values does not seem to
    # translate to any noticable changes in the parser's tree output
    mean, variance = tf.nn.moments(input, [1], keep_dims=True)
    return tf.nn.batch_normalization(
        input,
        mean, variance,
        offset=constants[0],
        scale=constants[1],
        variance_epsilon=1e-6)

def make_flabel_with_constants(input, constants):
    mul1 = tf.matmul(input, constants[0])
    mul1b = tf.nn.bias_add(mul1, constants[1])
    mul1b = make_layer_norm_with_constants(mul1b, constants[2:4])
    mul1b = tf.nn.relu(mul1b)
    mul2 = tf.matmul(mul1b, constants[4])
    mul2b = tf.nn.bias_add(mul2, constants[5], name='flabel')
    return mul2b

def make_ftag(input):
    constants = (
        tf.constant(sd['f_tag.0.weight'].numpy().transpose()),
        tf.constant(sd['f_tag.0.bias']),
        tf.constant(sd['f_tag.1.b_2'], name="tag/layer_norm/offset"),
        tf.constant(sd['f_tag.1.a_2'], name="tag/layer_norm/scale"),
        tf.constant(sd['f_tag.3.weight'].numpy().transpose()),
        tf.constant(sd['f_tag.3.bias']),
    )
    mul1 = tf.matmul(input, constants[0])
    mul1b = tf.nn.bias_add(mul1, constants[1])
    mul1b = make_layer_norm_with_constants(mul1b, constants[2:4])
    mul1b = tf.nn.relu(mul1b)
    mul2 = tf.matmul(mul1b, constants[4])
    mul2b = tf.nn.bias_add(mul2, constants[5], name='ftag')
    return mul2b

def make_flabel_constants():
    return (
        tf.constant(sd['f_label.0.weight'].numpy().transpose()),
        tf.constant(sd['f_label.0.bias']),
        tf.constant(sd['f_label.1.b_2'], name="label/layer_norm/offset"),
        tf.constant(sd['f_label.1.a_2'], name="label/layer_norm/scale"),
        tf.constant(sd['f_label.3.weight'].numpy().transpose()),
        tf.constant(sd['f_label.3.bias']),
    )

def make_network():
    # batch x num_subwords
    input_ids = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_ids')
    word_end_mask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_end_mask')
    input_dat, nonpad_ids, dim_flat, dim_padded, valid_mask, lengths = make_bert(input_ids, word_end_mask)
    input_pos_flat = tf.tile(position_table[:dim_padded[1]], [dim_padded[0], 1])
    input_pos = tf.gather(input_pos_flat, nonpad_ids)

    input_joint = tf.concat([input_dat, input_pos], -1)
    input_joint = make_layer_norm(input_joint, 'embedding.layer_norm', 'embedding/layer_norm')

    word_out = make_stacks(input_joint, nonpad_ids, dim_flat, dim_padded, valid_mask, num_stacks=parser.spec['hparams']['num_layers'])
    word_out = tf.concat([word_out[:, 0::2], word_out[:, 1::2]], -1)

    # part-of-speech predictions
    ftag = make_ftag(word_out)
    tags_packed = tf.argmax(ftag, axis=-1)
    tags = tf.reshape(
        tf.scatter_nd(indices=nonpad_ids[:, None], updates=tags_packed, shape=dim_flat),
        dim_padded
        )
    tags = tf.identity(tags, name="tags")

    fp_out = tf.concat([word_out[:-1,:512], word_out[1:,512:]], -1)

    fp_start_idxs = tf.cumsum(lengths, exclusive=True)
    fp_end_idxs = tf.cumsum(lengths) - 1 # the number of fenceposts is 1 less than the number of words

    fp_end_idxs_uneven = fp_end_idxs - tf.convert_to_tensor([1, 0])

    # Have to make these outside tf.map_fn for model compression to work
    constants = make_flabel_constants()

    def to_map(start_and_end):
        start, end = start_and_end
        fp = fp_out[start:end]
        flabel = make_flabel_with_constants(tf.reshape(fp[None,:,:] - fp[:,None,:], (-1, 1024)), constants)
        actual_chart_size = end-start
        flabel = tf.reshape(flabel, [actual_chart_size, actual_chart_size, -1])
        amount_to_pad = dim_padded[1] - actual_chart_size
        # extra padding on the label dimension is for the not-a-constituent label,
        # which always has a score of 0
        flabel = tf.pad(flabel, [[0, amount_to_pad], [0, amount_to_pad], [1, 0]])
        return flabel

    charts = tf.map_fn(to_map, (fp_start_idxs, fp_end_idxs), dtype=(tf.float32))
    charts = tf.identity(charts, name="charts")

    return input_ids, word_end_mask, charts, tags

# %%

from parse_nk import PTB_TOKEN_UNESCAPE
def bertify_batch(sentences):
    all_input_ids = np.zeros((len(sentences), parser.bert_max_len), dtype=int)
    all_word_end_mask = np.zeros((len(sentences), parser.bert_max_len), dtype=int)

    subword_max_len = 0
    for snum, sentence in enumerate(sentences):
        tokens = []
        word_end_mask = []

        tokens.append("[CLS]")
        word_end_mask.append(1)

        cleaned_words = []
        for word in sentence:
            word = PTB_TOKEN_UNESCAPE.get(word, word)
            if word == "n't" and cleaned_words:
                cleaned_words[-1] = cleaned_words[-1] + "n"
                word = "'t"
            cleaned_words.append(word)

        for word in cleaned_words:
            word_tokens = parser.bert_tokenizer.tokenize(word)
            for _ in range(len(word_tokens)):
                word_end_mask.append(0)
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)
        tokens.append("[SEP]")
        word_end_mask.append(1)

        input_ids = parser.bert_tokenizer.convert_tokens_to_ids(tokens)

        subword_max_len = max(subword_max_len, len(input_ids))

        all_input_ids[snum, :len(input_ids)] = input_ids
        all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

    all_input_ids = all_input_ids[:, :subword_max_len]
    all_word_end_mask = all_word_end_mask[:, :subword_max_len]
    return all_input_ids, all_word_end_mask

# %%

the_inp_tokens, the_inp_mask, the_out_chart, the_out_tags = make_network()

# %%

def tf_parse_batch(sentences):
    inp_val_tokens, inp_val_mask = bertify_batch([[word for (tag, word) in sentence] for sentence in sentences])
    out_val_chart, out_val_tags = sess.run((the_out_chart, the_out_tags), {the_inp_tokens: inp_val_tokens, the_inp_mask: inp_val_mask})

    trees = []
    scores = []
    for snum, sentence in enumerate(sentences):
        chart_size = len(sentence) + 1
        tf_chart = out_val_chart[snum,:chart_size,:chart_size,:]
        sentence = list(zip([TAG_VOCAB[idx] for idx in out_val_tags[snum,1:chart_size]], [x[1] for x in sentence]))
        tree, score = parser.decode_from_chart(sentence, tf_chart)
        trees.append(tree)
        scores.append(score)
    return trees, scores

#%%

print("Parsing test sentences using tensorflow...")
start_time = time.time()

test_predicted = []
for start_index in range(0, len(test_treebank), args.eval_batch_size):
# for start_index in range(0, 2, 2):
    print(start_index, format_elapsed(start_time))
    subbatch_trees = test_treebank[start_index:start_index+args.eval_batch_size]
    subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]
    predicted, _ = tf_parse_batch(subbatch_sentences)

    del _
    test_predicted.extend([p.convert() for p in predicted])

test_fscore = evaluate.evalb(args.evalb_dir, test_treebank[:len(test_predicted)], test_predicted)

print('Done', format_elapsed(start_time))
str(test_fscore)

#%%

input_node_names = [the_inp_tokens.name.split(':')[0], the_inp_mask.name.split(':')[0]]
output_node_names = [the_out_chart.name.split(':')[0], the_out_tags.name.split(':')[0]]

print("Input node names:", input_node_names)
print("Output node names:", output_node_names)

graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)

#%%
from tensorflow.tools.graph_transforms import TransformGraph
graph_def = TransformGraph(graph_def, input_node_names, output_node_names, [
'strip_unused_nodes()',
'remove_nodes(op=Identity, op=CheckNumerics)',
'fold_constants()',
'fold_old_batch_norms',
'fold_batch_norms',
'round_weights(num_steps=128)',
])

#%%

with open('export/model.pb', 'wb') as f:
    f.write(graph_def.SerializeToString())

vocab_path = pytorch_pretrained_bert.file_utils.cached_path(
    pytorch_pretrained_bert.tokenization.PRETRAINED_VOCAB_ARCHIVE_MAP[bert_model])
target_file = "export/vocab.txt"
if not os.path.exists(target_file):
    shutil.copyfile(vocab_path, target_file)

META = {
    'tag_vocab': TAG_VOCAB,
    'label_vocab': LABEL_VOCAB,
    'language_code': 'en',
    'provides_tags': True,
    'bert_do_lower_case': bert_do_lower_case,
    }

with open("export/meta.json", "w") as f:
    json.dump(META, f)

#%%
newg = tf.Graph()

with newg.as_default():
    tf.import_graph_def(graph_def)

new_inp_tokens = newg.get_tensor_by_name('import/input_ids:0')
new_inp_mask = newg.get_tensor_by_name('import/word_end_mask:0')
new_out_chart = newg.get_tensor_by_name('import/charts:0')
new_out_tags = newg.get_tensor_by_name('import/tags:0')

new_sess = tf.InteractiveSession(graph=newg)
#%%

def tf_parse_batch_new(sentences):
    inp_val_tokens, inp_val_mask = bertify_batch([[word for (tag, word) in sentence] for sentence in sentences])
    out_val_chart, out_val_tags = new_sess.run((new_out_chart, new_out_tags), {new_inp_tokens: inp_val_tokens, new_inp_mask: inp_val_mask})

    trees = []
    scores = []
    for snum, sentence in enumerate(sentences):
        chart_size = len(sentence) + 1
        tf_chart = out_val_chart[snum,:chart_size,:chart_size,:]
        sentence = list(zip([TAG_VOCAB[idx] for idx in out_val_tags[snum,1:chart_size]], [x[1] for x in sentence]))
        tree, score = parser.decode_from_chart(sentence, tf_chart)
        trees.append(tree)
        scores.append(score)
    return trees, scores

print("Parsing test sentences using tensorflow...")
start_time = time.time()

test_predicted = []
for start_index in range(0, len(test_treebank), args.eval_batch_size):
    print(start_index, format_elapsed(start_time))
    subbatch_trees = test_treebank[start_index:start_index+args.eval_batch_size]
    subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]
    predicted, _ = tf_parse_batch_new(subbatch_sentences)
    del _
    test_predicted.extend([p.convert() for p in predicted])

test_fscore = evaluate.evalb(args.evalb_dir, test_treebank[:len(test_predicted)], test_predicted)

print('Done', format_elapsed(start_time))
str(test_fscore)

#%%
#%%
#%%
#%%
