PTB_UNESCAPE_MAPPING = {
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}

NO_SPACE_BEFORE = {"-RRB-", "-RCB-", "-RSB-", "''"} | set("%.,!?:;")
NO_SPACE_AFTER = {"-LRB-", "-LCB-", "-LSB-", "``", "`"} | set("$#")
NO_SPACE_BEFORE_TOKENS_ENGLISH = {"'", "'s", "'ll", "'re", "'d", "'m", "'ve"}
PTB_DASH_ESCAPED = {"-RRB-", "-RCB-", "-RSB-", "-LRB-", "-LCB-", "-LSB-", "--"}


def ptb_unescape(words):
    cleaned_words = []
    for word in words:
        word = PTB_UNESCAPE_MAPPING.get(word, word)
        # This un-escaping for / and * was not yet added for the
        # parser version in https://arxiv.org/abs/1812.11760v1
        # and related model releases (e.g. benepar_en2)
        word = word.replace("\\/", "/").replace("\\*", "*")
        # Mid-token punctuation occurs in biomedical text
        word = word.replace("-LSB-", "[").replace("-RSB-", "]")
        word = word.replace("-LRB-", "(").replace("-RRB-", ")")
        word = word.replace("-LCB-", "{").replace("-RCB-", "}")
        word = word.replace("``", '"').replace("`", "'").replace("''", '"')
        cleaned_words.append(word)
    return cleaned_words


def guess_space_after_non_english(escaped_words):
    sp_after = [True for _ in escaped_words]
    for i, word in enumerate(escaped_words):
        if i > 0 and (
            (
                word.startswith("-")
                and not any(word.startswith(x) for x in PTB_DASH_ESCAPED)
            )
            or any(word.startswith(x) for x in NO_SPACE_BEFORE)
            or word == "'"
        ):
            sp_after[i - 1] = False
        if (
            word.endswith("-") and not any(word.endswith(x) for x in PTB_DASH_ESCAPED)
        ) or any(word.endswith(x) for x in NO_SPACE_AFTER):
            sp_after[i] = False

    return sp_after


def guess_space_after(escaped_words, for_english=True):
    if not for_english:
        return guess_space_after_non_english(escaped_words)

    sp_after = [True for _ in escaped_words]
    for i, word in enumerate(escaped_words):
        if word.lower() == "n't" and i > 0:
            sp_after[i - 1] = False
        elif word.lower() == "not" and i > 0 and escaped_words[i - 1].lower() == "can":
            sp_after[i - 1] = False

        if i > 0 and (
            (
                word.startswith("-")
                and not any(word.startswith(x) for x in PTB_DASH_ESCAPED)
            )
            or any(word.startswith(x) for x in NO_SPACE_BEFORE)
            or word.lower() in NO_SPACE_BEFORE_TOKENS_ENGLISH
        ):
            sp_after[i - 1] = False
        if (
            word.endswith("-") and not any(word.endswith(x) for x in PTB_DASH_ESCAPED)
        ) or any(word.endswith(x) for x in NO_SPACE_AFTER):
            sp_after[i] = False

    return sp_after
