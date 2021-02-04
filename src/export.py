import argparse
import functools
import itertools
import os.path
import time

import torch
import torch.nn as nn

import numpy as np

import evaluate
import treebanks

from benepar import Parser, InputSentence
from benepar.partitioned_transformer import PartitionedMultiHeadAttention

import json


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def inputs_from_treebank(treebank, predict_tags):
    return [
        InputSentence(
            words=example.words,
            space_after=example.space_after,
            tags=None if predict_tags else [tag for _, tag in example.pos()],
            escaped_words=list(example.leaves()),
        )
        for example in treebank
    ]


def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = treebanks.load_trees(
        args.test_path, args.test_path_text, args.text_processing
    )
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path))
    parser = Parser(args.model_path, batch_size=args.batch_size)

    print("Parsing test sentences...")
    start_time = time.time()

    if args.output_path == "-":
        output_file = sys.stdout
    elif args.output_path:
        output_file = open(args.output_path, "w")
    else:
        output_file = None

    test_predicted = []
    for predicted_tree in parser.parse_sents(
        inputs_from_treebank(test_treebank, predict_tags=args.predict_tags)
    ):
        test_predicted.append(predicted_tree)
        if output_file is not None:
            print(tree.pformat(margin=1e100), file=output_file)

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank.trees, test_predicted)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )


def get_compressed_state_dict(model):
    state_dict = model.state_dict()
    for module_name, module in model.named_modules():
        if not isinstance(
            module, (nn.Linear, nn.Embedding, PartitionedMultiHeadAttention)
        ):
            continue
        elif "token_type_embeddings" in module_name:
            continue
        elif "position_embeddings" in module_name:
            continue
        elif "f_tag" in module_name or "f_label" in module_name:
            continue
        elif "project_pretrained" in module_name:
            continue

        if isinstance(module, PartitionedMultiHeadAttention):
            weight_names = [
                module_name + "." + param
                for param in ("w_qkv_c", "w_qkv_p", "w_o_c", "w_o_p")
            ]
        else:
            weight_names = [module_name + ".weight"]
        for weight_name in weight_names:
            weight = state_dict[weight_name]
            if weight.shape.numel() <= 2048:
                continue
            print(weight_name, ":", weight.shape.numel(), "parameters")

            if isinstance(module, nn.Embedding) or "word_embeddings" in module_name or "shared.weight" in weight_name:
                is_embedding = True
            else:
                is_embedding = False

            num_steps = 64
            use_histogram = True
            if "pooler.dense.weight" in weight_name:
                weight.data.zero_()
                continue
            elif "pretrained_model" in weight_name and not is_embedding:
                num_steps = 128
                if not model.retokenizer.is_t5:
                    use_histogram = False
            elif isinstance(module, PartitionedMultiHeadAttention):
                num_steps = 128

            if use_histogram:
                observer = torch.quantization.HistogramObserver()
                observer.dst_nbins = num_steps
                observer(weight)
                scale, zero_point = observer.calculate_qparams()
                scale = scale.item()
                zero_point = zero_point.item()
                cluster_centers = (
                    scale * (np.arange(0, 256, 256 / num_steps) - zero_point)[:, None]
                )
                cluster_centers = np.asarray(cluster_centers, dtype=np.float32)
            else:
                weight_np = weight.cpu().detach().numpy()
                min_val = weight_np.min()
                max_val = weight_np.max()
                bucket_width = (max_val - min_val) / num_steps
                cluster_centers = (
                    min_val
                    + (np.arange(num_steps, dtype=np.float32) + 0.5) * bucket_width
                )
                cluster_centers = cluster_centers.reshape((-1, 1))

            codebook = torch.tensor(
                cluster_centers, dtype=weight.dtype, device=weight.device
            )
            distances = weight.data.reshape((-1, 1)) - codebook.t()
            codes = torch.argmin(distances ** 2, dim=-1)
            weight_rounded = codebook[codes].reshape(weight.shape)
            weight.data.copy_(weight_rounded)

    return state_dict


def run_export(args):
    if args.test_path is not None:
        print("Loading test trees from {}...".format(args.test_path))
        test_treebank = treebanks.load_trees(
            args.test_path, args.test_path_text, args.text_processing
        )
        print("Loaded {:,} test examples.".format(len(test_treebank)))
    else:
        test_treebank = None

    print("Loading model from {}...".format(args.model_path))
    parser = Parser(args.model_path, batch_size=args.batch_size)
    model = parser._parser
    if model.pretrained_model is None:
        raise ValueError(
            "Exporting is only defined when using a pre-trained transformer "
            "encoder. For CharLSTM-based model, just distribute the pytorch "
            "checkpoint directly. You may manually delete the 'optimizer' "
            "field to reduce file size by discarding the optimizer state."
        )

    if test_treebank is not None:
        print("Parsing test sentences (predicting tags)...")
        start_time = time.time()
        test_inputs = inputs_from_treebank(test_treebank, predict_tags=True)
        test_predicted = list(parser.parse_sents(test_inputs))
        test_fscore = evaluate.evalb(args.evalb_dir, test_treebank.trees, test_predicted)
        test_elapsed = format_elapsed(start_time)
        print("test-fscore {} test-elapsed {}".format(test_fscore, test_elapsed))

        print("Parsing test sentences (not predicting tags)...")
        start_time = time.time()
        test_inputs = inputs_from_treebank(test_treebank, predict_tags=False)
        notags_test_predicted = list(parser.parse_sents(test_inputs))
        notags_test_fscore = evaluate.evalb(
            args.evalb_dir, test_treebank.trees, notags_test_predicted
        )
        notags_test_elapsed = format_elapsed(start_time)
        print(
            "test-fscore {} test-elapsed {}".format(notags_test_fscore, notags_test_elapsed)
        )

    print("Exporting tokenizer...")
    model.retokenizer.tokenizer.save_pretrained(args.output_dir)

    print("Exporting config...")
    config = model.pretrained_model.config
    config.benepar = model.config
    config.save_pretrained(args.output_dir)

    if args.compress:
        print("Compressing weights...")
        state_dict = get_compressed_state_dict(model.cpu())
        print("Saving weights...")
    else:
        print("Exporting weights...")
        state_dict = model.cpu().state_dict()
    torch.save(state_dict, os.path.join(args.output_dir, "benepar_model.bin"))

    del model, parser, state_dict

    print("Loading exported model from {}...".format(args.output_dir))
    exported_parser = Parser(args.output_dir, batch_size=args.batch_size)

    if test_treebank is None:
        print()
        print("Export complete.")
        print("Did not verify model accuracy because no treebank was provided.")
        return

    print("Parsing test sentences (predicting tags)...")
    start_time = time.time()
    test_inputs = inputs_from_treebank(test_treebank, predict_tags=True)
    exported_predicted = list(exported_parser.parse_sents(test_inputs))
    exported_fscore = evaluate.evalb(
        args.evalb_dir, test_treebank.trees, exported_predicted
    )
    exported_elapsed = format_elapsed(start_time)
    print(
        "exported-fscore {} exported-elapsed {}".format(
            exported_fscore, exported_elapsed
        )
    )

    print("Parsing test sentences (not predicting tags)...")
    start_time = time.time()
    test_inputs = inputs_from_treebank(test_treebank, predict_tags=False)
    notags_exported_predicted = list(exported_parser.parse_sents(test_inputs))
    notags_exported_fscore = evaluate.evalb(
        args.evalb_dir, test_treebank.trees, notags_exported_predicted
    )
    notags_exported_elapsed = format_elapsed(start_time)
    print(
        "exported-fscore {} exported-elapsed {}".format(
            notags_exported_fscore, notags_exported_elapsed
        )
    )

    print()
    print("Export and verification complete.")
    fscore_delta = evaluate.FScore(
        recall=notags_exported_fscore.recall - notags_test_fscore.recall,
        precision=notags_exported_fscore.precision - notags_test_fscore.precision,
        fscore=notags_exported_fscore.fscore - notags_test_fscore.fscore,
        complete_match=(
            notags_exported_fscore.complete_match - notags_test_fscore.complete_match
        ),
        tagging_accuracy=(
            exported_fscore.tagging_accuracy - test_fscore.tagging_accuracy
        ),
    )
    print("delta-fscore {}".format(fscore_delta))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path", type=str, required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", type=str, required=True)
    subparser.add_argument("--test-path-text", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--predict-tags", action="store_true")
    subparser.add_argument("--output-path", default="")
    subparser.add_argument("--batch-size", type=int, default=8)

    subparser = subparsers.add_parser("export")
    subparser.set_defaults(callback=run_export)
    subparser.add_argument("--model-path", type=str, required=True)
    subparser.add_argument("--output-dir", type=str, required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", type=str, default=None)
    subparser.add_argument("--test-path-text", type=str)
    subparser.add_argument("--text-processing", default="default")
    subparser.add_argument("--compress", action="store_true")
    subparser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
