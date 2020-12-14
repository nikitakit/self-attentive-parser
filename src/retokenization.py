"""
Converts from linguistically motivated word-based tokenization to subword
tokenization used by pre-trained models.
"""

import numpy as np
import torch
import transformers


def retokenize(
    tokenizer,
    words,
    space_after,
    return_token_type_ids=False,
    return_attention_mask=False,
    return_offsets_mapping=False,
    return_tensors=None,
    **kwargs
):
    """Re-tokenize into subwords.

    Args:
        tokenizer: An instance of transformers.PreTrainedTokenizerFast
        words: List of words
        space_after: A list of the same length as `words`, indicating whether
            whitespace follows each word.
        **kwargs: all remaining arguments are passed on to tokenizer.__call__

    Returns:
        The output of tokenizer.__call__, with one additional dictionary field:
        - **words_from_tokens** -- List of the same length as `words`, where
          each entry is the index of the *last* subword that overlaps the
          corresponding word.
    """
    s = "".join([w + (" " if sp else "") for w, sp in zip(words, space_after)])
    word_offset_starts = np.cumsum(
        [0] + [len(w) + (1 if sp else 0) for w, sp in zip(words, space_after)]
    )[:-1]
    word_offset_ends = word_offset_starts + np.asarray([len(w) for w in words])

    tokenized = tokenizer(
        s,
        return_token_type_ids=return_token_type_ids,
        return_attention_mask=return_attention_mask,
        return_offsets_mapping=True,
        return_tensors=return_tensors,
        **kwargs
    )
    if return_offsets_mapping:
        token_offset_mapping = tokenized["offset_mapping"]
    else:
        token_offset_mapping = tokenized.pop("offset_mapping")
    if return_tensors is not None:
        token_offset_mapping = np.asarray(token_offset_mapping)[0].tolist()

    offset_mapping_iter = iter(
        [
            (i, (start, end))
            for (i, (start, end)) in enumerate(token_offset_mapping)
            if start != end
        ]
    )
    token_idx, (token_start, token_end) = next(offset_mapping_iter)
    words_from_tokens = [-1] * len(words)
    for word_idx, (word_start, word_end) in enumerate(
        zip(word_offset_starts, word_offset_ends)
    ):
        while token_end <= word_start:
            token_idx, (token_start, token_end) = next(offset_mapping_iter)
        if token_end > word_end:
            words_from_tokens[word_idx] = token_idx
        while token_end <= word_end:
            words_from_tokens[word_idx] = token_idx
            try:
                token_idx, (token_start, token_end) = next(offset_mapping_iter)
            except StopIteration:
                assert word_idx == len(words) - 1
                break
    if return_tensors == "np":
        words_from_tokens = np.asarray(words_from_tokens, dtype=int)
    elif return_tensors == "pt":
        words_from_tokens = torch.tensor(words_from_tokens, dtype=torch.long)
    elif return_tensors == "tf":
        raise NotImplementedError("Returning tf tensors is not implemented")
    tokenized["words_from_tokens"] = words_from_tokens
    return tokenized


class Retokenizer:
    def __init__(self, pretrained_model_name_or_path, retain_start_stop=False):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, fast=True
        )
        self.retain_start_stop = retain_start_stop

    def __call__(self, words, space_after, **kwargs):
        example = retokenize(self.tokenizer, words, space_after, **kwargs)
        if self.retain_start_stop:
            # Include CLS/SEP tokens
            if kwargs.get("return_tensors") == "pt":
                example["words_from_tokens"] = torch.cat(
                    [
                        torch.tensor([0]),
                        example["words_from_tokens"],
                        torch.tensor([example["input_ids"].shape[-1] - 1]),
                    ]
                )
            else:
                example["words_from_tokens"] = (
                    [0] + example["words_from_tokens"] + [len(example["input_ids"]) - 1]
                )
        return example

    def pad(self, encoded_inputs, return_tensors=None, **kwargs):
        if return_tensors != "pt":
            raise NotImplementedError("Only return_tensors='pt' is supported.")
        res = self.tokenizer.pad(
            [
                {k: v for k, v in example.items() if k != "words_from_tokens"}
                for example in encoded_inputs
            ],
            return_tensors=return_tensors,
            **kwargs
        )
        res["words_from_tokens"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(example["words_from_tokens"]) for example in encoded_inputs],
            batch_first=True,
            padding_value=-1,
        )
        res["valid_token_mask"] = res["words_from_tokens"] != -1
        return res
