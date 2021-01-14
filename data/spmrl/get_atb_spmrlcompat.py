import os
import nltk

TO_DELETE = set(
    [
        # Train
        "20000815_AFP_ARB.0110:5",
        "20000915_AFP_ARB.0016:3",
        "20000915_AFP_ARB.0041:5",
        "20000915_AFP_ARB.0049:8",
        "20001015_AFP_ARB.0005:3",
        "20001015_AFP_ARB.0119:2",
        "20001015_AFP_ARB.0141:4",
        "20001015_AFP_ARB.0148:3",
        "20001115_AFP_ARB.0013:4",
        "20001115_AFP_ARB.0111:10",
        "UMAAH_UM.ARB_20020602-a.0002:1",
        "UMAAH_UM.ARB_backissue_12-a.0006:3",
        "UMAAH_UM.ARB_backissue_19-a.0004:7",
        "UMAAH_UM.ARB_backissue_20-a.0005:2",
        "UMAAH_UM.ARB_backissue_20-a.0014:1",
        "UMAAH_UM.ARB_backissue_23-a.0018:1",
        "ANN20020215.0031:8",
        "ANN20020715.0003:8",
        "ANN20020715.0006:30",
        "ANN20020815.0061:8",
        "ANN20020915.0001:21",
        "ANN20020915.0002:21",
        "ANN20020915.0011:3",
        "ANN20021015.0082:10",
        "ANN20021015.0083:14",
        "ANN20021015.0096:9",
        "ANN20021015.0096:10",

        # Dev
        "ANN20020115.0004:29",

        # Test
        "20001115_AFP_ARB.0147:4",
        "20001115_AFP_ARB.0194:6",
        "ANN20021115.0083:4",
        "ANN20021215.0029:4",
    ]
)


def delete_x_over_x(tree, ignore_lst=False):
    """Deletes certain cases of unary chains that repeat the same label.

    This is ad-hoc code that re-creates the output of the SPMRL processing, but not the
    actual process. It is not the original SPMRL code. Do not use this function for any
    purpose other than re-creating the SPMRL Arabic constituent data.
    """
    treepositions = tree.treepositions()
    for treeposition in treepositions:
        if len(treeposition) <= 1:
            continue
        subtree = tree[treeposition]
        if isinstance(subtree, str):
            continue
        label = subtree.label()
        parent = tree[treeposition[:-1]]
        parent_label = parent.label()
        if label != parent_label:
            continue
        if len(parent) == 1:
            if label == "LST" and ignore_lst:
                continue
            elif (
                label == "LST"
                and len(subtree) == 1
                and not isinstance(subtree[0], str)
                and subtree[0].label() == "LST"
            ):
                ignore_lst = True
            tree[treeposition[:-1]] = subtree
            return delete_x_over_x(tree, ignore_lst)

        try:
            siblings = []
            for sib_pos in range(len(parent)):
                if sib_pos == treeposition[-1]:
                    continue
                sibling = tree[treeposition[:-1] + (sib_pos,)]
                siblings.append(sibling)
        except:
            continue

        if all(
            len(sibling) == 1
            and not isinstance(sibling[0], str)
            and sibling[0].label() == "-NONE-"
            for sibling in siblings
        ):
            tree[treeposition[:-1]] = subtree
            return delete_x_over_x(tree, ignore_lst)

    return tree


def write_to_file(tree_dir, conll_file, outfile, to_delete=()):
    sent_ids = []
    file_ids = set()

    sents = []
    with open(conll_file) as f:
        sent_lines = []
        sent_id = None
        for line in f:
            if not line.strip() and sent_lines:
                assert sent_id is not None
                sents.append("".join(sent_lines))
                sent_lines = []
                sent_id = None

            sent_lines.append(line)

            if line.startswith("# sent_id"):
                sent_id = line.split("=")[1].strip()
                sent_ids.append(sent_id)
                file_id = sent_id.split(":")[0]
                file_ids.add(file_id)

    assert len(sents) == len(sent_ids)

    trees_by_file_id = {}
    for file_id in file_ids:
        with open(os.path.join(tree_dir, file_id + ".tree")) as f:
            trees = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        nltk.Tree.fromstring(line)
                        tree = nltk.Tree.fromstring("(ROOT {})".format(line))
                    except:
                        tree = nltk.Tree.fromstring("(ROOT (XP {}))".format(line))
                    tree = delete_x_over_x(tree)  # Specific to SPMRL
                    trees.append(tree)
            trees_by_file_id[file_id] = trees

    with open(outfile, "w") as f:
        for i, (sent, sent_id) in enumerate(
            sorted(
                zip(sents, sent_ids),
                # Re-order to match SPMRL data: first ATB1, then ATB2, then ATB3
                key=lambda x: (
                    "2000" not in x[1],
                    "UMAAH" not in x[1],
                    "ANN2002" not in x[1],
                ),
            )
        ):
            if sent_id in to_delete:
                continue
            file_id, sent_num = sent_id.split(":")
            tree = trees_by_file_id[file_id][int(sent_num) - 1]
            tree_rep = tree.pformat(margin=1e100)
            assert "\n" not in tree_rep
            tree_rep = tree_rep.replace("{", "A")  # Specific to SPMRL
            f.write(tree_rep)
            f.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--ud_train")
    parser.add_argument("--ud_dev")
    parser.add_argument("--ud_test")
    args = parser.parse_args()

    write_to_file(
        args.tree_dir,
        args.ud_train,
        os.path.join(args.out_dir, "train.spmrlcompat.withmosttraces"),
        TO_DELETE,
    )
    write_to_file(
        args.tree_dir,
        args.ud_dev,
        os.path.join(args.out_dir, "dev.spmrlcompat.withmosttraces"),
        TO_DELETE,
    )
    write_to_file(
        args.tree_dir,
        args.ud_test,
        os.path.join(args.out_dir, "test.spmrlcompat.withmosttraces"),
        TO_DELETE,
    )
