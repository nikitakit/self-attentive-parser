"""
This file documents the process that was used to convert our model from a saved
PyTorch checkpoint to a TensorFlow graph. The code here was run one cell at a
time inside an IPython/Jupyter session.

The ELMo code for TensorFlow can be found at:
https://github.com/allenai/bilm-tf/commit/7cd16b0c1487587cadd4d5cffbb662e9013e990f
with additional changes applied:
0001-bilm-tf-changes-for-use-with-benepar.patch
"""

%cd ~/dev/self-attentive-parser
import sys
sys.path.insert(0, "/Users/kitaev/dev/self-attentive-parser/src")

from bilm import Batcher, BidirectionalLanguageModel, weight_layers

import argparse
import itertools
import os.path
import time

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
    model_path_base="models/en_elmo_dev=95.21.pt"
    test_path="data/22.auto.clean" # dev set
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

#%%

print("Loading test trees from {}...".format(args.test_path))
test_treebank = trees.load_trees(args.test_path)
print("Loaded {:,} test examples.".format(len(test_treebank)))

#%%

import tensorflow as tf

sess = tf.InteractiveSession()

sd = parser.state_dict()

LABEL_VOCAB = [x[0] for x in sorted(parser.label_vocab.indices.items(), key=lambda x: x[1])]
LABEL_VOCAB

#%%

def make_elmo(chars_batched):
    bilm = BidirectionalLanguageModel(
                    options_file="data/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    weight_file="data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    max_batch_size=128)

    lm = bilm(chars_batched)
    word_representations_padded = weight_layers('scalar_mix', lm, l2_coef=0.0)['weighted_op']

    # Strip off multiplication by gamma. Our parser has gamma=1 because there is a
    # projection matrix right after
    word_representations_padded = word_representations_padded.op.inputs[0]

    with tf.variable_scope('', reuse=True):
        elmo_scalar_mix_matrix = tf.get_variable('scalar_mix_ELMo_W')

    tf.global_variables_initializer().run()
    tf.assign(elmo_scalar_mix_matrix, [
        float(sd['elmo.scalar_mix_0.scalar_parameters.0']),
        float(sd['elmo.scalar_mix_0.scalar_parameters.1']),
        float(sd['elmo.scalar_mix_0.scalar_parameters.2'])]).eval()

    # Switch from padded to packed representation
    valid_mask = lm['mask']
    dim_padded = tf.shape(lm['mask'])[:2]
    mask_flat = tf.reshape(lm['mask'], (-1,))
    dim_flat = tf.shape(mask_flat)[:1]
    nonpad_ids = tf.to_int32(tf.where(mask_flat)[:,0])
    word_reps_shape = tf.shape(word_representations_padded)
    word_representations_flat = tf.reshape(word_representations_padded, [-1, int(word_representations_padded.shape[-1])])
    word_representations = tf.gather(word_representations_flat, nonpad_ids)

    projected_annotations = tf.matmul(
        word_representations,
        tf.constant(sd['project_elmo.weight'].numpy().transpose()))

    return projected_annotations, nonpad_ids, dim_flat, dim_padded, valid_mask, lm['lengths']

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
    # batch x num_words x 50
    chars = tf.placeholder(shape=(None, None, 50), dtype=tf.int32, name='chars')

    input_dat, nonpad_ids, dim_flat, dim_padded, valid_mask, lengths = make_elmo(chars)
    chars_shape = tf.shape(chars)
    input_pos_flat = tf.tile(position_table[:chars_shape[1]], [chars_shape[0], 1])
    input_pos = tf.gather(input_pos_flat, nonpad_ids)

    input_joint = tf.concat([input_dat, input_pos], -1)
    input_joint = make_layer_norm(input_joint, 'embedding.layer_norm', 'embedding/layer_norm')

    word_out = make_stacks(input_joint, nonpad_ids, dim_flat, dim_padded, valid_mask, num_stacks=4)
    word_out = tf.concat([word_out[:, 0::2], word_out[:, 1::2]], -1)


    fp_out = tf.concat([word_out[:-1,:512], word_out[1:,512:]], -1)

    fp_start_idxs = tf.cumsum(lengths, exclusive=True)
    fp_end_idxs = tf.cumsum(lengths) - 1 # the number of fenceposts is 1 less than the number of words

    fp_end_idxs_uneven = fp_end_idxs - tf.convert_to_tensor([1, 0])

    # Have to make these outside tf.map_fn for model compression to work
    constants = make_flabel_constants()

    def to_map(start_and_end):
        start, end = start_and_end
        fp = fp_out[start:end]
        # flabel = make_flabel(tf.reshape(fp[None,:,:] - fp[:,None,:], (-1, 1024)))
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

    return chars, charts

# %%

def charify_batch(sentences):
    ELMO_START_SENTENCE = 256
    ELMO_STOP_SENTENCE = 257
    ELMO_START_WORD = 258
    ELMO_STOP_WORD = 259
    ELMO_CHAR_PAD = 260

    padded_len = max([len(sentence) + 2 for sentence in sentences])

    all_chars = np.zeros((len(sentences), padded_len, 50), dtype=np.int32)

    for snum, sentence in enumerate(sentences):
        all_chars[snum, :len(sentence)+2,:] = ELMO_CHAR_PAD

        all_chars[snum, 0, 0] = ELMO_START_WORD
        all_chars[snum, 0, 1] = ELMO_START_SENTENCE
        all_chars[snum, 0, 2] = ELMO_STOP_WORD

        for i, word in enumerate(sentence):
            chars = [ELMO_START_WORD] + list(word.encode('utf-8', 'ignore')[:(50-2)]) + [ELMO_STOP_WORD]
            all_chars[snum, i+1, :len(chars)] = chars

        all_chars[snum, len(sentence)+1, 0] = ELMO_START_WORD
        all_chars[snum, len(sentence)+1, 1] = ELMO_STOP_SENTENCE
        all_chars[snum, len(sentence)+1, 2] = ELMO_STOP_WORD

        # Add 1; 0 is a reserved value for signaling words past the end of the
        # sentence, which we don't have because batch_size=1
        all_chars[snum, :len(sentence)+2,:] += 1

    return all_chars

# %%

the_inp, the_out = make_network()

# %%


def tf_parse_batch(sentences):
    inp_val = charify_batch([[word for (tag, word) in sentence] for sentence in sentences])
    out_val = sess.run(the_out, {the_inp: inp_val})

    trees = []
    scores = []
    for snum, sentence in enumerate(sentences):
        chart_size = len(sentence) + 1
        tf_chart = out_val[snum,:chart_size,:chart_size,:]
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

input_node_names = [the_inp.name.split(':')[0]]
output_node_names = [the_out.name.split(':')[0]]

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

with open('batched_elmo128.pb', 'wb') as f:
    f.write(graph_def.SerializeToString())

#%%
newg = tf.Graph()

with newg.as_default():
    tf.import_graph_def(graph_def)

new_inp = newg.get_tensor_by_name('import/chars:0')
new_out = newg.get_tensor_by_name('import/charts:0')

new_sess = tf.InteractiveSession(graph=newg)
#%%

def tf_parse_batch_new(sentences):
    inp_val = charify_batch([[word for (tag, word) in sentence] for sentence in sentences])
    out_val = new_sess.run(new_out, {new_inp: inp_val})

    trees = []
    scores = []
    for snum, sentence in enumerate(sentences):
        chart_size = len(sentence) + 1
        tf_chart = out_val[snum,:chart_size,:chart_size,:]
        tree, score = parser.decode_from_chart(sentence, tf_chart)
        trees.append(tree)
        scores.append(score)
    return trees, scores

print("Parsing test sentences using tensorflow...")
start_time = time.time()

test_predicted = []
for start_index in range(0, len(test_treebank), args.eval_batch_size):
    subbatch_trees = test_treebank[start_index:start_index+args.eval_batch_size]
    subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in subbatch_trees]
    predicted, _ = tf_parse_batch_new(subbatch_sentences)
    del _
    test_predicted.extend([p.convert() for p in predicted])

test_fscore = evaluate.evalb(args.evalb_dir, test_treebank[:len(test_predicted)], test_predicted)

str(test_fscore)

#%%
#%%
#%%
#%%
#%%
#%%
