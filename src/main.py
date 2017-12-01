import argparse
import itertools
import os.path
import time

import dynet as dy
import numpy as np

import evaluate
import parse
import parse_pytorch
import torch
import trees
import vocabulary

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    train_parse = [tree.convert() for tree in train_treebank]

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(parse.START)
    tag_vocab.index(parse.STOP)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {parse.START, parse.STOP, parse.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)

    print("Initializing model...")
    if not args.use_pytorch:
        model = dy.ParameterCollection()
    if args.use_pytorch and args.parser_type == "chart":
        parser = parse_pytorch.ChartParser(
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.dropout,
        )
    elif args.use_pytorch:
        raise NotImplementedError("This parser type unsupported with pytorch")
    elif args.parser_type == "top-down":
        parser = parse.TopDownParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.split_hidden_dim,
            args.dropout,
        )
    else:
        parser = parse.ChartParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.dropout,
        )
    if not args.use_pytorch:
        trainer = dy.AdamTrainer(model)
    else:
        trainer = torch.optim.Adam(parser.parameters())

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        for tree in dev_treebank:
            if not args.use_pytorch:
                dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
            predicted, _ = parser.parse(sentence)
            dev_predicted.append(predicted.convert())

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            if not args.use_pytorch:
                dy.save(best_dev_model_path, [parser])
            else:
                print("WARNING: saving not implemented when using pytorch")

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            if not args.use_pytorch:
                dy.renew_cg()
            else:
                trainer.zero_grad()
            batch_losses = []
            for tree in train_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                if args.parser_type == "top-down":
                    _, loss = parser.parse(sentence, tree, args.explore)
                else:
                    _, loss = parser.parse(sentence, tree)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            if not args.use_pytorch:
                batch_loss = dy.average(batch_losses)
                batch_loss_value = batch_loss.scalar_value()
                batch_loss.backward()
                trainer.update()
            else:
                batch_loss = sum(batch_losses) / len(batch_losses)
                batch_loss_value = float(batch_loss.data.numpy())
                batch_loss.backward()
                trainer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()

def dynet_to_pytoch(dynet_parser):
    spec = dynet_parser.spec

    def map_name(k):
        if k.startswith("/Parser/_"):
            num = int(k.split("_")[1])
            if num == 0:
                return "tag_embeddings.weight"
            else:
                return "word_embeddings.weight"
        elif k.startswith("/Parser/Feedforward"):
            num = int(k.split("_")[1])
            base_idx = (num // 2) * 2
            role = "weight" if (num % 2 == 0) else "bias"
            return f"f_label.{base_idx}.{role}"
        elif k.startswith("/Parser/birnn/vanilla-lstm-builder"):
            key = k.split("/Parser/birnn/vanilla-lstm-builder")[1]
            key = key[1:]
            birnn_num, param_num = key.split("_")
            birnn_num = 0 if not birnn_num else int(birnn_num[:-1])
            param_num = int(param_num)

            layer = birnn_num // 2
            reverse = "_reverse" if (birnn_num % 2 == 1) else ""


            if param_num == 0:
                return f"lstm.weight_ih_l{layer}{reverse}"
            elif param_num == 1:
                return f"lstm.weight_hh_l{layer}{reverse}"
            elif param_num == 2:
                return f"lstm.bias_ih_l{layer}{reverse}"
            else:
                raise NotImplementedError(f"Key {k} cannot be mapped to pytorch")

            return birnn_num, param_num

        raise NotImplementedError(f"Key {k} cannot be mapped to pytorch")

    state_dict = {}
    for param in (dynet_parser.model.lookup_parameters_list()
                    + dynet_parser.model.parameters_list()):
        name = param.name()
        torch_name = map_name(name)
        value = param.as_array()
        if 'lstm' in torch_name:
            # reorder the array
            observed_lstm_size = value.shape[0] // 4
            value = np.concatenate([
                value[0*observed_lstm_size:1*observed_lstm_size],
                value[1*observed_lstm_size:2*observed_lstm_size],
                value[3*observed_lstm_size:4*observed_lstm_size],
                value[2*observed_lstm_size:3*observed_lstm_size],
                ])

        torch_value = torch.from_numpy(value)
        state_dict[torch_name] = torch_value

        if "bias_ih" in torch_name:
            second_torch_name = torch_name.replace("bias_ih", "bias_hh")
            second_torch_value = torch.zeros(torch_value.size())
            second_torch_value[observed_lstm_size:2*observed_lstm_size] = 1.0
            state_dict[second_torch_name] = second_torch_value

    return parse_pytorch.ChartParser.from_spec(spec, state_dict)

def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    if args.model_path_base.endswith(".pt"):
        if not args.use_pytorch:
            raise NotImplementedError("Can't use pytorch parameter savefiles with dynet backend")

        raise NotImplementedError("Pytorch savefiles not implemented yet")
    else:
        model = dy.ParameterCollection()
        [parser] = dy.load(args.model_path_base, model)

        if args.use_pytorch:
            parser = dynet_to_pytoch(parser)

    print("Parsing test sentences...")
    start_time = time.time()

    test_predicted = []
    for tree in test_treebank:
        if not args.use_pytorch:
            dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
        predicted, _ = parser.parse(sentence)
        test_predicted.append(predicted.convert())

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )

def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--parser-type", choices=["top-down", "chart"], required=True)
    subparser.add_argument("--tag-embedding-dim", type=int, default=50)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=250)
    subparser.add_argument("--label-hidden-dim", type=int, default=250)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--explore", action="store_true")
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", default="data/02-21.10way.clean")
    subparser.add_argument("--dev-path", default="data/22.auto.clean")
    subparser.add_argument("--batch-size", type=int, default=10)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--use-pytorch", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="data/23.auto.clean")
    subparser.add_argument("--use-pytorch", action="store_true")

    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()
