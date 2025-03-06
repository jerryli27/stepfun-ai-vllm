# SPDX-License-Identifier: Apache-2.0
# Modified from https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flash_attn.py

import math
from typing import List, Optional, Tuple

import pytest
import torch

from vllm.platforms import current_platform
from vllm.vllm_flash_attn import (fa_version_unsupported_reason,
                                  flash_attn_varlen_func,
                                  flash_attn_with_kvcache,
                                  is_fa_version_supported)


try:
    import os
    OPTIMUS_LIB_PATH = os.environ.get('OPTIMUS_LIB_PATH')
    if torch.__version__ >= "2.5":
        torch.ops.load_library(os.path.join(OPTIMUS_LIB_PATH, "liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so"))
    elif torch.__version__ >= "2.3":
        torch.ops.load_library(os.path.join(OPTIMUS_LIB_PATH, 'liboptimus_ths-torch2.3-cu121.cpython-310-x86_64-linux-gnu.so'))
    elif torch.__version__ >= "2.2":
        torch.ops.load_library(os.path.join(OPTIMUS_LIB_PATH, 'liboptimus_ths-torch2.2-cu121.cpython-310-x86_64-linux-gnu.so'))
    else:
        raise ImportError("Failed to load optimus library for flash attn ops for step1 model")
except:
    raise ImportError("Failed to load optimus library for flash attn ops for step1 model")


NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.float16, torch.bfloat16]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]


def get_alibi_slopes(n_heads, device, dtype=torch.float32):
    # alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
    #     (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
    #     is added to the attention score of query i and key j.
    n = 2 ** math.floor(math.log2(n_heads))  # nearest 2**n to n_heads
    m0 = 2.0 ** (-8.0 / n)
    # 2^(-8/n), 2^(-8*2/n), 2^(-8*3/n), ...
    slopes = torch.pow(m0, torch.arange(1, n + 1))
    if n < n_heads:
        m1 = 2.0 ** (-4.0 / n)
        mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        slopes = torch.cat([slopes, mm])
    slopes = slopes.to(dtype=dtype).to(device)
    return slopes


def build_alibi_cache(query_len, kv_len, n_heads, alibi_sqrt: bool, device, dtype=torch.float32):
    # get slopes
    slopes = get_alibi_slopes(n_heads, device, dtype)

    tril = torch.tril(torch.ones(1, query_len, kv_len, device=device), diagonal=kv_len - query_len)

    bias_rows = torch.arange(query_len, device=device).view(1, -1)
    bias_cols = torch.arange(kv_len, device=device).view(-1, 1)
    if alibi_sqrt:
        bias = -torch.sqrt(bias_rows - bias_cols + kv_len - query_len)
    else:
        bias = -torch.abs(bias_rows - bias_cols + kv_len - query_len)
    bias = bias.view(1, query_len, kv_len) * slopes.view(-1, 1, 1)
    bias = bias.masked_fill(tril == 0, float("-inf"))

    return bias.type(dtype)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    do_alibi: bool = False,
    alibi_sqrt: bool = False,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    _, num_query_heads, _ = query.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len -
                                             (query_len + sliding_window) +
                                             1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if do_alibi:
            bias = build_alibi_cache(query_len, kv_len, num_query_heads, alibi_sqrt, q.device, q.dtype)
            attn += bias
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("use_out", [True, False])
@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("sliding_window", [None, 256])
# @pytest.mark.parametrize("fa_version", [2, 3])
@pytest.mark.parametrize("fa_version", [2,])
@pytest.mark.parametrize("do_alibi", [False, True])
@pytest.mark.parametrize("alibi_sqrt", [False, True])
@torch.inference_mode()
def test_flash_attn_with_paged_kv(
    use_out: bool,
    kv_lens: List[int],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    sliding_window: Optional[int],
    fa_version: int,
    do_alibi: bool,
    alibi_sqrt: bool,
) -> None:
    torch.set_default_device("cuda")
    if not is_fa_version_supported(fa_version):
        pytest.skip(f"Flash attention version {fa_version} not supported due "
                    f"to: \"{fa_version_unsupported_reason(fa_version)}\"")

    current_platform.seed_everything(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    window_size = ((sliding_window - 1, 0) if sliding_window is not None else
                   (-1, -1))

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    if do_alibi:
        alibi_slopes = get_alibi_slopes(num_query_heads, query.device)
    else:
        alibi_slopes = None

    q = query.unsqueeze(1)
    out = torch.empty_like(q) if use_out else None

    if do_alibi and alibi_sqrt:
        output = torch.ops.Optimus.vllm_fwd_kvcache(
            q,
            key_cache,
            value_cache,
            None,
            None,
            kv_lens_tensor,
            None,
            None,
            None,
            block_tables,
            alibi_slopes,
            out,
            scale,
            True,
            window_size[0],
            window_size[1],
            True,
            0
        )
    else:
        output = flash_attn_with_kvcache(
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            out=out,
            softmax_scale=scale,
            causal=True,
            block_table=block_tables,
            cache_seqlens=kv_lens_tensor,
            softcap=soft_cap if soft_cap is not None else 0,
            window_size=window_size,
            fa_version=fa_version,
            alibi_slopes=alibi_slopes,
        )
    output = output if not use_out else out
    output = output.squeeze(1)

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=[1] * num_seqs,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                soft_cap=soft_cap,
                                sliding_window=sliding_window,
                                do_alibi=do_alibi,
                                alibi_sqrt=alibi_sqrt)
    torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


# @pytest.mark.parametrize("use_out", [True, False])
# @pytest.mark.parametrize("seq_lens",
#                          [[(1, 1328), (5, 18),
#                            (129, 463)], [(1, 523), (1, 37), (1, 2011)]])
# @pytest.mark.parametrize("num_heads", NUM_HEADS)
# @pytest.mark.parametrize("head_size", HEAD_SIZES)
# @pytest.mark.parametrize("block_size", BLOCK_SIZES)
# @pytest.mark.parametrize("sliding_window", [None, 256])
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
# @pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
# @pytest.mark.parametrize("fa_version", [2, 3])
# @torch.inference_mode()
# def test_varlen_with_paged_kv(
#     use_out: bool,
#     seq_lens: List[Tuple[int, int]],
#     num_heads: Tuple[int, int],
#     head_size: int,
#     sliding_window: Optional[int],
#     dtype: torch.dtype,
#     block_size: int,
#     soft_cap: Optional[float],
#     num_blocks: int,
#     fa_version: int,
# ) -> None:
#     torch.set_default_device("cuda")
#     if not is_fa_version_supported(fa_version):
#         pytest.skip(f"Flash attention version {fa_version} not supported due "
#                     f"to: \"{fa_version_unsupported_reason(fa_version)}\"")
#     current_platform.seed_everything(0)
#     num_seqs = len(seq_lens)
#     query_lens = [x[0] for x in seq_lens]
#     kv_lens = [x[1] for x in seq_lens]
#     num_query_heads = num_heads[0]
#     num_kv_heads = num_heads[1]
#     assert num_query_heads % num_kv_heads == 0
#     max_query_len = max(query_lens)
#     max_kv_len = max(kv_lens)
#     window_size = ((sliding_window - 1, 0) if sliding_window is not None else
#                    (-1, -1))
#     scale = head_size**-0.5

#     query = torch.randn(sum(query_lens),
#                         num_query_heads,
#                         head_size,
#                         dtype=dtype)
#     key_cache = torch.randn(num_blocks,
#                             block_size,
#                             num_kv_heads,
#                             head_size,
#                             dtype=dtype)
#     value_cache = torch.randn_like(key_cache)
#     cu_query_lens = torch.tensor([0] + query_lens,
#                                  dtype=torch.int32).cumsum(dim=0,
#                                                            dtype=torch.int32)
#     kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

#     max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
#     block_tables = torch.randint(0,
#                                  num_blocks,
#                                  (num_seqs, max_num_blocks_per_seq),
#                                  dtype=torch.int32)

#     out = torch.empty_like(query) if use_out else None
#     output = flash_attn_varlen_func(
#         q=query,
#         k=key_cache,
#         v=value_cache,
#         out=out,
#         cu_seqlens_q=cu_query_lens,
#         seqused_k=kv_lens,
#         max_seqlen_q=max_query_len,
#         max_seqlen_k=max_kv_len,
#         softmax_scale=scale,
#         causal=True,
#         window_size=window_size,
#         block_table=block_tables,
#         softcap=soft_cap if soft_cap is not None else 0,
#         fa_version=fa_version,
#     )
#     output = output if not use_out else out

#     ref_output = ref_paged_attn(
#         query=query,
#         key_cache=key_cache,
#         value_cache=value_cache,
#         query_lens=query_lens,
#         kv_lens=kv_lens,
#         block_tables=block_tables,
#         scale=scale,
#         sliding_window=sliding_window,
#         soft_cap=soft_cap,
#     )
#     torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2), \
#         f"{torch.max(torch.abs(output - ref_output))}"
