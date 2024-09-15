"""
methods to get token masks for activation steering or direction extraction

all of them take the arguments:
input_ids of shape batch x max_n_seq
seq_lens of shape batch - the actual sequence lengths of the input_ids (right-padding is assumed)
(optional) search_tokens - tokens to search for in the input_ids
    - if search_tokens is a tensor of shape n_seq, we search for the exact sequence of tokens in each sequence
    - if search_tokens is a tuple of two tensors of shape n_seq, we mask between the two sequences of tokens
in_sampling_mode - a boolean flag indicating whether we are in sampling mode or not
    - in sampling mode, we are sampling one extra token at a time
    - so we have different logic for the token masks in this case

and return a tensor of shape batch x n_seq with 1s corresponding to the token positions of interest and 0s elsewhere
"""

from typing import Callable
import torch


TokenLocationFn = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None, bool],
    torch.Tensor,
]


def validate_shapes(location_func: TokenLocationFn) -> TokenLocationFn:

    def wrapper(input_ids, seq_lens, search_tokens, in_sampling_mode=False):
        assert len(input_ids.shape) == 2
        if seq_lens is not None:
            assert len(seq_lens.shape) == 1
            assert input_ids.shape[0] == seq_lens.shape[0]
        if search_tokens is not None:
            if isinstance(search_tokens, tuple):
                assert len(search_tokens) == 2
                assert len(search_tokens[0].shape) == 1
                assert len(search_tokens[1].shape) == 1
            else:
                assert len(search_tokens.shape) == 1
        return location_func(input_ids, seq_lens, search_tokens, in_sampling_mode)

    wrapper.__name__ = location_func.__name__

    return wrapper


def get_search_token_positions(input_ids: torch.Tensor, search_tokens: torch.Tensor) -> list[list[tuple[int, int]]]:
    """
    Returns a list of lists containing (start, end) tuples for each occurrence of search_tokens in each sequence of input_ids.

    input_ids (torch.Tensor): Tensor of shape (batch_size, seq_len)
    search_tokens (torch.Tensor): Tensor of shape (search_len,)

    returns: list[list[tuple[int, int]]]: A list where each element corresponds to a sequence in input_ids and contains a list of (start, end) tuples for each match.
    """
    batch_size, seq_len = input_ids.shape
    search_len = search_tokens.shape[0]
    positions = seq_len - search_len + 1

    if positions <= 0:
        raise ValueError("Sequence length must be longer than search_tokens length")

    # Use as_strided to get all subsequences of length search_len
    subsequences = input_ids.as_strided(
        size=(batch_size, positions, search_len),
        stride=(input_ids.stride(0), input_ids.stride(1), input_ids.stride(1))
    )

    # Compare subsequences with search_tokens
    matches = (subsequences == search_tokens).all(dim=2)  # shape (batch_size, positions)

    # For each sequence, get the indices where matches is True
    positions_list = []
    for i in range(batch_size):
        matched_positions = matches[i].nonzero(as_tuple=False).squeeze(1)
        if matched_positions.numel() == 0:
            raise ValueError(f"search_tokens {search_tokens.tolist()} not found in input_ids {input_ids[i].tolist()}")
        start_positions = matched_positions.tolist()
        end_positions = (matched_positions + search_len).tolist()
        positions_list.append(list(zip(start_positions, end_positions)))
    return positions_list


@validate_shapes
def last_token(input_ids, seq_lens, _, in_sampling_mode=False):
    if in_sampling_mode:
        return torch.zeros_like(input_ids)
    mask = torch.zeros_like(input_ids)
    for i, seq_len in enumerate(seq_lens):
        mask[i, seq_len - 1] = 1
    return mask


@validate_shapes
def all_tokens(input_ids, seq_lens, _, in_sampling_mode=False):
    if in_sampling_mode:
        return torch.ones_like(input_ids)
    mask = torch.zeros_like(input_ids)
    for i, seq_len in enumerate(seq_lens):
        mask[i, :seq_len] = 1
    return mask


@validate_shapes
def at_search_tokens(input_ids, _, search_tokens, in_sampling_mode=False):
    assert not isinstance(search_tokens, tuple)
    if in_sampling_mode:
        return torch.zeros_like(input_ids)
    search_positions = get_search_token_positions(input_ids, search_tokens)
    mask = torch.zeros_like(input_ids)
    for i, positions in enumerate(search_positions):
        for start, end in positions:
            mask[i, start:end] = 1
    return mask


@validate_shapes
def before_search_tokens(input_ids, _, search_tokens, in_sampling_mode=False):
    assert not isinstance(search_tokens, tuple)
    if in_sampling_mode:
        return torch.zeros_like(input_ids)
    search_positions = get_search_token_positions(input_ids, search_tokens)
    mask = torch.zeros_like(input_ids)
    for i, positions in enumerate(search_positions):
        start, _ = positions[0]
        mask[i, :start] = 1
    return mask


@validate_shapes
def from_search_tokens(input_ids, _, search_tokens, in_sampling_mode=False):
    assert not isinstance(search_tokens, tuple)
    if in_sampling_mode:
        return torch.ones_like(input_ids)
    search_positions = get_search_token_positions(input_ids, search_tokens)
    mask = torch.zeros_like(input_ids)
    for i, positions in enumerate(search_positions):
        start, _ = positions[0]
        mask[i, start:] = 1
    return mask


@validate_shapes
def after_search_tokens(input_ids, _, search_tokens, in_sampling_mode=False):
    assert not isinstance(search_tokens, tuple)
    if in_sampling_mode:
        return torch.ones_like(input_ids)
    search_positions = get_search_token_positions(input_ids, search_tokens)
    mask = torch.zeros_like(input_ids)
    for i, positions in enumerate(search_positions):
        _, end = positions[0]
        mask[i, end:] = 1
    return mask


@validate_shapes
def between_search_tokens(input_ids, _, search_tokens, in_sampling_mode=False):
    assert isinstance(search_tokens, tuple)
    if in_sampling_mode:
        return torch.zeros_like(input_ids)
    start_search_positions = get_search_token_positions(input_ids, search_tokens[0])
    end_search_positions = get_search_token_positions(input_ids, search_tokens[1])
    mask = torch.zeros_like(input_ids)
    for i, (start_positions, end_positions) in enumerate(zip(start_search_positions, end_search_positions)):
        for (start, _), (_, end) in zip(start_positions, end_positions):
            mask[i, start:end] = 1
    return mask