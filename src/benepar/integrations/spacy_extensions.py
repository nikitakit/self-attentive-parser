NOT_PARSED_SENTINEL = object()


class NonConstituentException(Exception):
    pass


class ConstituentData:
    def __init__(self, starts, ends, labels, loc_to_constituent, label_vocab):
        self.starts = starts
        self.ends = ends
        self.labels = labels
        self.loc_to_constituent = loc_to_constituent
        self.label_vocab = label_vocab


def get_constituent(span):
    constituent_data = span.doc._._constituent_data
    if constituent_data is NOT_PARSED_SENTINEL:
        raise Exception(
            "No constituency parse is available for this document."
            " Consider adding a BeneparComponent to the pipeline."
        )

    search_start = constituent_data.loc_to_constituent[span.start]
    if span.start + 1 < len(constituent_data.loc_to_constituent):
        search_end = constituent_data.loc_to_constituent[span.start + 1]
    else:
        search_end = len(constituent_data.ends)
    found_position = None
    for position in range(search_start, search_end):
        if constituent_data.ends[position] <= span.end:
            if constituent_data.ends[position] == span.end:
                found_position = position
            break

    if found_position is None:
        raise NonConstituentException("Span is not a constituent: {}".format(span))
    return constituent_data, found_position


def get_labels(span):
    constituent_data, position = get_constituent(span)
    label_num = constituent_data.labels[position]
    return constituent_data.label_vocab[label_num]


def parse_string(span):
    constituent_data, position = get_constituent(span)
    label_vocab = constituent_data.label_vocab
    doc = span.doc

    idx = position - 1

    def make_str():
        nonlocal idx
        idx += 1
        i, j, label_idx = (
            constituent_data.starts[idx],
            constituent_data.ends[idx],
            constituent_data.labels[idx],
        )
        label = label_vocab[label_idx]
        if (i + 1) >= j:
            token = doc[i]
            s = (
                "("
                + u"{} {}".format(token.tag_, token.text)
                .replace("(", "-LRB-")
                .replace(")", "-RRB-")
                .replace("{", "-LCB-")
                .replace("}", "-RCB-")
                .replace("[", "-LSB-")
                .replace("]", "-RSB-")
                + ")"
            )
        else:
            children = []
            while (
                (idx + 1) < len(constituent_data.starts)
                and i <= constituent_data.starts[idx + 1]
                and constituent_data.ends[idx + 1] <= j
            ):
                children.append(make_str())

            s = u" ".join(children)

        for sublabel in reversed(label):
            s = u"({} {})".format(sublabel, s)
        return s

    return make_str()


def get_subconstituents(span):
    constituent_data, position = get_constituent(span)
    label_vocab = constituent_data.label_vocab
    doc = span.doc

    while position < len(constituent_data.starts):
        start = constituent_data.starts[position]
        end = constituent_data.ends[position]

        if span.end <= start or span.end < end:
            break

        yield doc[start:end]
        position += 1


def get_child_spans(span):
    constituent_data, position = get_constituent(span)
    label_vocab = constituent_data.label_vocab
    doc = span.doc

    child_start_expected = span.start
    position += 1
    while position < len(constituent_data.starts):
        start = constituent_data.starts[position]
        end = constituent_data.ends[position]

        if span.end <= start or span.end < end:
            break

        if start == child_start_expected:
            yield doc[start:end]
            child_start_expected = end

        position += 1


def get_parent_span(span):
    constituent_data, position = get_constituent(span)
    label_vocab = constituent_data.label_vocab
    doc = span.doc
    sent = span.sent

    position -= 1
    while position >= 0:
        start = constituent_data.starts[position]
        end = constituent_data.ends[position]

        if start <= span.start and span.end <= end:
            return doc[start:end]
        if end < span.sent.start:
            break
        position -= 1

    return None


def install_spacy_extensions():
    from spacy.tokens import Doc, Span, Token

    # None is not allowed as a default extension value!
    Doc.set_extension("_constituent_data", default=NOT_PARSED_SENTINEL)

    Span.set_extension("labels", getter=get_labels)
    Span.set_extension("parse_string", getter=parse_string)
    Span.set_extension("constituents", getter=get_subconstituents)
    Span.set_extension("parent", getter=get_parent_span)
    Span.set_extension("children", getter=get_child_spans)

    Token.set_extension(
        "labels", getter=lambda token: get_labels(token.doc[token.i : token.i + 1])
    )
    Token.set_extension(
        "parse_string",
        getter=lambda token: parse_string(token.doc[token.i : token.i + 1]),
    )
    Token.set_extension(
        "parent", getter=lambda token: get_parent_span(token.doc[token.i : token.i + 1])
    )


try:
    install_spacy_extensions()
except ImportError:
    pass
