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
    return_attention_mask=True,
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
    words_from_tokens = [-100] * len(words)
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
        if not self.tokenizer.is_fast:
            raise NotImplementedError(
                "Converting from treebank tokenization to tokenization used by a "
                "pre-trained model requires a 'fast' tokenizer, which appears to not "
                "be available for this pre-trained model type."
            )
        self.retain_start_stop = retain_start_stop
        self.is_t5 = "T5Tokenizer" in str(type(self.tokenizer))
        self.is_gpt2 = "GPT2Tokenizer" in str(type(self.tokenizer))

        if self.is_gpt2:
            # The provided GPT-2 tokenizer does not specify a padding token by default
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.retain_start_stop:
            # When retain_start_stop is set, the next layer after the pre-trained model
            # expects start and stop token embeddings. For BERT these can naturally be
            # the feature vectors for CLS and SEP, but pre-trained models differ in the
            # special tokens that they use. This code attempts to find special token
            # positions for each pre-trained model.
            dummy_ids = self.tokenizer.build_inputs_with_special_tokens([-100])
            if self.is_t5:
                # For T5 we use the output from the decoder, which accepts inputs that
                # are shifted relative to the encoder.
                dummy_ids = [self.tokenizer.pad_token_id] + dummy_ids
            if self.is_gpt2:
                # For GPT-2, we append an eos token if special tokens are needed
                dummy_ids = dummy_ids + [self.tokenizer.eos_token_id]
            try:
                input_idx = dummy_ids.index(-100)
            except ValueError:
                raise NotImplementedError(
                    "Could not automatically infer how to extract start/stop tokens "
                    "from this pre-trained model"
                )
            num_prefix_tokens = input_idx
            num_suffix_tokens = len(dummy_ids) - input_idx - 1
            self.start_token_idx = None
            self.stop_token_idx = None
            if num_prefix_tokens > 0:
                self.start_token_idx = num_prefix_tokens - 1
            if num_suffix_tokens > 0:
                self.stop_token_idx = -num_suffix_tokens
            if self.start_token_idx is None and num_suffix_tokens > 0:
                self.start_token_idx = -1
            if self.stop_token_idx is None and num_prefix_tokens > 0:
                self.stop_token_idx = 0
            if self.start_token_idx is None or self.stop_token_idx is None:
                assert num_prefix_tokens == 0 and num_suffix_tokens == 0
                raise NotImplementedError(
                    "Could not automatically infer how to extract start/stop tokens "
                    "from this pre-trained model because the associated tokenizer "
                    "appears not to add any special start/stop/cls/sep/etc. tokens "
                    "to the sequence."
                )

    def __call__(self, words, space_after, **kwargs):
        example = retokenize(self.tokenizer, words, space_after, **kwargs)
        if self.is_t5:
            # decoder_input_ids (which are shifted wrt input_ids) will be created after
            # padding, but we adjust words_from_tokens now, in anticipation.
            if isinstance(example["words_from_tokens"], list):
                example["words_from_tokens"] = [
                    x + 1 for x in example["words_from_tokens"]
                ]
            else:
                example["words_from_tokens"] += 1
        if self.retain_start_stop:
            num_tokens = len(example["input_ids"])
            if self.is_t5:
                num_tokens += 1
            if self.is_gpt2:
                num_tokens += 1
                if kwargs.get("return_tensors") == "pt":
                    example["input_ids"] = torch.cat(
                        example["input_ids"],
                        torch.tensor([self.tokenizer.eos_token_id]),
                    )
                    example["attention_mask"] = torch.cat(
                        example["attention_mask"], torch.tensor([1])
                    )
                else:
                    example["input_ids"].append(self.tokenizer.eos_token_id)
                    example["attention_mask"].append(1)
            if num_tokens > self.tokenizer.model_max_length:
                raise ValueError(
                    f"Sentence of length {num_tokens} (in sub-word tokens) exceeds the "
                    f"maximum supported length of {self.tokenizer.model_max_length}"
                )
            start_token_idx = (
                self.start_token_idx
                if self.start_token_idx >= 0
                else num_tokens + self.start_token_idx
            )
            stop_token_idx = (
                self.stop_token_idx
                if self.stop_token_idx >= 0
                else num_tokens + self.stop_token_idx
            )
            if kwargs.get("return_tensors") == "pt":
                example["words_from_tokens"] = torch.cat(
                    [
                        torch.tensor([start_token_idx]),
                        example["words_from_tokens"],
                        torch.tensor([stop_token_idx]),
                    ]
                )
            else:
                example["words_from_tokens"] = (
                    [start_token_idx] + example["words_from_tokens"] + [stop_token_idx]
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
        if self.tokenizer.padding_side == "right":
            res["words_from_tokens"] = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.tensor(example["words_from_tokens"])
                    for example in encoded_inputs
                ],
                batch_first=True,
                padding_value=-100,
            )
        else:
            # XLNet adds padding tokens on the left of the sequence, so
            # words_from_tokens must be adjusted to skip the added padding tokens.
            assert self.tokenizer.padding_side == "left"
            res["words_from_tokens"] = torch.nn.utils.rnn.pad_sequence(
                [
                    torch.tensor(example["words_from_tokens"])
                    + (res["input_ids"].shape[-1] - len(example["input_ids"]))
                    for example in encoded_inputs
                ],
                batch_first=True,
                padding_value=-100,
            )

        if self.is_t5:
            res["decoder_input_ids"] = torch.cat(
                [
                    torch.full_like(
                        res["input_ids"][:, :1], self.tokenizer.pad_token_id
                    ),
                    res["input_ids"],
                ],
                1,
            )
            res["decoder_attention_mask"] = torch.cat(
                [
                    torch.ones_like(res["attention_mask"][:, :1]),
                    res["attention_mask"],
                ],
                1,
            )
        res["valid_token_mask"] = res["words_from_tokens"] != -100
        return res
