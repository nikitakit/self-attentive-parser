"""
Utilities for splitting batches of examples into smaller sub-batches.

This is useful during training when the batch size is too large to fit on GPU,
meaning that gradient accumulation across multiple sub-batches must be used.
It is also useful for batching examples during evaluation. Unlike a naive
approach, this code groups examples with similar lengths to reduce the amount
of wasted computation due to padding. 
"""

import numpy as np


def split(*data, costs, max_cost):
    """Splits a batch of input items into sub-batches.

    Args:
        *data: One or more lists of input items, all of the same length
        costs: A list of costs for each item
        max_cost: Maximum total cost for each sub-batch

    Yields:
        (example_ids, *subbatch_data) tuples.
    """
    costs = np.asarray(costs, dtype=int)
    costs_argsort = np.argsort(costs).tolist()

    subbatch_size = 1
    while costs_argsort:
        if subbatch_size == len(costs_argsort) or (
            subbatch_size * costs[costs_argsort[subbatch_size]] > max_cost
        ):
            subbatch_item_ids = costs_argsort[:subbatch_size]
            subbatch_data = [[items[i] for i in subbatch_item_ids] for items in data]
            yield (subbatch_item_ids,) + tuple(subbatch_data)
            costs_argsort = costs_argsort[subbatch_size:]
            subbatch_size = 1
        else:
            subbatch_size += 1


def map(func, *data, costs, max_cost, **common_kwargs):
    """Maps a function over subbatches of input items.

    Args:
        func: Function to map over the data
        *data: One or more lists of input items, all of the same length.
        costs: A list of costs for each item
        max_cost: Maximum total cost for each sub-batch
        **common_kwargs: Keyword arguments to pass to all calls of func

    Returns:
        A list of outputs from calling func(*subbatch_data, **kwargs) for each
        subbatch, and then rearranging the outputs from func into the original
        item order.
    """
    res = [None] * len(data[0])
    for item_ids, *subbatch_items in split(*data, costs=costs, max_cost=max_cost):
        subbatch_out = func(*subbatch_items, **common_kwargs)
        for item_id, item_out in zip(item_ids, subbatch_out):
            res[item_id] = item_out
    return res
