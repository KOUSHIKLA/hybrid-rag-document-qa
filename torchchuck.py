
# torchchuck.py
# --------------------------------------------------------------------------------------------------
# TorchChuck: First-of-its-kind GPU/distributed hierarchical text chunker with multi-format IO
#
# - Hierarchical, token/char-aware, separator-prioritized chunking (FHPA: Fused Hierarchical Pack-and-Align)
# - Single-pass, vectorized, blockwise Torch kernels (GPU-accelerated when available)
# - Deterministic, streaming-safe, overlap-aware, UTF-8 start-boundary aligned
# - Distributed-ready with scatter, all_gather, reduce, and stable global dedup via 64-bit rolling hash
# - Robust file loaders for .txt, .md, .html, .json, .jsonl, .yaml/.yml, .csv, .parquet (optional deps)
# - Scales with input size and number of distributed ranks; designed for high throughput
#
# Fixes addressing reported issues:
#   * Eliminated torch.frombuffer(..., like=...) for broad PyTorch compatibility.
#   * Avoided read-only buffer warnings by copying via bytearray in _to_device_u8.
#   * Overlap progression fix: when previous chunk length <= overlap, next chunk now starts at end (no sliding-by-1).
#   * Stop condition when reaching end of text to avoid tail thrashing.
#   * Windows CRLF-safe by default separators ("\n\n", "\n", " ", ""); users may add "\r\n" if desired.
#
# CUDA/robustness/perf updates (logic preserved):
#   * Uniform device normalization with safe CUDA fallback when unavailable.
#   * Pinned-memory hop for CPU->CUDA transfer in _to_device_u8 for faster non_blocking copies.
#   * Synchronous GPU->CPU hop in _slice_to_bytes to prevent partial reads causing UTF-8 decode errors.
#   * Fast-path separator search for 1-byte and 2-byte patterns (dominant defaults) using pure vector ops (no unfold).
#   * Kept generic blockwise search for longer patterns; identical results, lower peak memory.
#   * GPU-friendly boundary search without Python-side device sync in inner loops.
#   * All changes preserve exact chunking logic and outputs; they only improve CUDA stability and throughput.
#
# New end-to-end integration and whitespace/noise controls:
#   * Parallel IO and parallel chunking on CPU via ThreadPool (configurable, order-stable).
#   * Optional whitespace normalization to eliminate long spaces and blank-line bursts (configurable limits).
#   * Optional punctuation-run limiting (e.g., "....." -> "...", "------" -> "---") to remove noisy long runs.
#   * Optional removal of ASCII/Unicode "bar" lines (e.g., "--------------------" or "=======") with configurable length.
#   * Per-file normalization and direct-text normalization paths; offsets refer to normalized text when enabled.
#   * Safer defaults for CLI separators (no arbitrary multi-space or punctuation separators).
#   * Optional torch.compile acceleration for hot kernels when available.
# --------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import io
import math
import json
import csv
import sys
import hashlib
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except Exception:
    _DIST_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except Exception:
    _PANDAS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PYARROW_AVAILABLE = True
except Exception:
    _PYARROW_AVAILABLE = False

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except Exception:
    _YAML_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except Exception:
    _BS4_AVAILABLE = False


# --------------------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------------------

KeepMode = Literal["end", "start", "none"]


@dataclass(frozen=True)
class ChunkConfig:
    """
    Core chunking configuration and IO/distributed options.
    Notes on whitespace/noise normalization:
      - If normalize_whitespace is True, input text is normalized before chunking (and before hashing).
      - Offsets (when returned) refer to the normalized text, not the raw file bytes.
      - Normalization collapses long space runs and excessive blank lines, removes line-edge spaces, and standardizes CRLF.
      - If normalize_punct_runs is True, long runs of punctuation (., -, _, =, *, ~, #, and common box/long dashes) are capped.
      - If strip_ascii_bars is True, "bar lines" made of the above punctuation are removed when length >= bar_min_len.
    Parallelism notes:
      - num_workers_io enables multi-file IO in parallel (safe for all devices).
      - num_workers_chunk enables parallel chunking on CPU; for CUDA we default to serial to avoid device thrash.
    """
    chunk_size: int
    chunk_overlap: int = 0
    separators: Tuple[str, ...] = ("\n\n", "\n", " ", "")
    keep_separator: KeepMode = "end"
    device: Optional[str] = None
    compile: bool = False
    return_offsets: bool = False
    deduplicate: bool = False
    token_boundaries: Optional[Sequence[int]] = None
    mem_block_cap_bytes: int = 256 * (1 << 20)
    csv_text_columns: Optional[Sequence[str]] = None
    parquet_text_columns: Optional[Sequence[str]] = None
    json_text_fields: Optional[Sequence[str]] = None
    yaml_text_fields: Optional[Sequence[str]] = None
    html_strip: bool = True
    distributed: bool = False
    return_local_when_distributed: bool = False
    output_hash: bool = True

    # Whitespace and noise normalization controls
    normalize_whitespace: bool = True
    max_space_run: int = 1
    max_newline_run: int = 2
    strip_line_edges: bool = True
    drop_blank_chunks: bool = True
    normalize_punct_runs: bool = True
    max_punct_run: int = 3
    strip_ascii_bars: bool = True
    bar_min_len: int = 6
    punct_chars: str = "._-~=*#•·–—─━═"

    # Parallelism
    num_workers_io: int = max(1, min(8, (os.cpu_count() or 2)))
    num_workers_chunk: int = max(1, min(8, (os.cpu_count() or 2)))


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    start: int
    end: int
    text: str
    hash64: Optional[int] = None


# --------------------------------------------------------------------------------------------------
# Whitespace and noise normalization utilities
# --------------------------------------------------------------------------------------------------

# Map a set of common Unicode space-like characters to a normal space.
_WS_EXPAND_MAP = {
    ord(u"\u00A0"): " ",  # NO-BREAK SPACE
    ord(u"\u1680"): " ",  # OGHAM SPACE MARK
    ord(u"\u180E"): " ",  # MONGOLIAN VOWEL SEPARATOR (deprecated, still seen)
    ord(u"\u2000"): " ",
    ord(u"\u2001"): " ",
    ord(u"\u2002"): " ",
    ord(u"\u2003"): " ",
    ord(u"\u2004"): " ",
    ord(u"\u2005"): " ",
    ord(u"\u2006"): " ",
    ord(u"\u2007"): " ",
    ord(u"\u2008"): " ",
    ord(u"\u2009"): " ",
    ord(u"\u200A"): " ",
    ord(u"\u200B"): " ",
    ord(u"\u202F"): " ",
    ord(u"\u205F"): " ",
    ord(u"\u3000"): " ",
}

# Pre-compiled patterns reused across normalization passes.
_RE_CRLF = re.compile(r"\r\n?")  # CRLF or CR -> LF
_RE_SPACES_BEFORE_NL = re.compile(r"[ \t\f\v]+\n")
_RE_SPACES_AFTER_NL = re.compile(r"\n[ \t\f\v]+")
# Punct-run limiter and newline/space-run limiters are built/applied on demand per config.


def _limit_runs(text: str, char: str, max_run: int) -> str:
    """
    Replace runs of a specific character with a capped run length.
    Example: max_run=2 for '\n' ensures no more than 2 blank lines in a row.
    """
    if max_run < 1:
        return text.replace(char, "")
    pat = re.compile(re.escape(char) + "{" + str(max_run + 1) + ",}")
    return pat.sub(char * max_run, text)


def _limit_punct_runs(text: str, max_run: int, charset: str) -> str:
    """
    Cap runs of punctuation in 'charset' to at most max_run using a single backref regex:
      e.g., "Hello....." -> "Hello..." if '.' in charset and max_run=3.
    """
    if max_run < 1 or not charset:
        return text
    pat = re.compile(r"([%s])\1{%d,}" % (re.escape(charset), max_run))
    return pat.sub(lambda m: m.group(1) * max_run, text)


def _drop_ascii_bar_lines(text: str, bar_min_len: int, charset: str) -> str:
    """
    Remove lines composed entirely of punctuation from 'charset' (ignoring spaces/tabs) if length >= bar_min_len.
    Fix: parameter renamed to 'bar_min_len' to match call sites and CLI flags; previously named 'min_len' caused a TypeError.
    """
    if bar_min_len <= 0 or not text:
        return text
    out_lines: List[str] = []
    for ln in text.splitlines():
        core = ln.strip()
        if not core:
            out_lines.append(ln)
            continue
        core_compact = core.replace(" ", "").replace("\t", "")
        if len(core_compact) >= bar_min_len and core_compact and all((c in charset) for c in core_compact):
            # Drop this noise bar line (e.g., "-----", "======", "********", "────────")
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def _normalize_text_ws(
    text: str,
    max_space_run: int = 1,
    max_newline_run: int = 2,
    strip_line_edges: bool = True,
    normalize_punct_runs: bool = True,
    max_punct_run: int = 3,
    strip_ascii_bars: bool = True,
    bar_min_len: int = 6,
    punct_chars: str = "._-~=*#•·–—─━═",
) -> str:
    """
    End-to-end whitespace and noise normalization:
      - Standardize line-endings to '\n'
      - Convert odd Unicode spaces to ASCII space and replace tabs with space
      - Strip spaces adjacent to newlines
      - Optionally collapse punctuation runs (., -, _, =, *, ~, #, dashes, box draw)
      - Optionally drop "bar lines" made only of these punctuation chars when length >= bar_min_len
      - Collapse consecutive spaces and blank lines
      - Remove leading/trailing whitespace
    The order is chosen to minimize artifacts:
      CRLF normalize -> space/tab homogenize -> trim spaces around newlines
      -> (punct-run limit) -> (drop bars) -> cap runs of spaces/newlines -> final edge-strip
    """
    if not text:
        return text

    # Normalize newlines and homogenize space-likes
    s = _RE_CRLF.sub("\n", text)
    s = s.translate(_WS_EXPAND_MAP).replace("\t", " ")

    # Remove spaces hugging newlines early
    s = _RE_SPACES_BEFORE_NL.sub("\n", s)
    s = _RE_SPACES_AFTER_NL.sub("\n", s)

    # Optional punctuation-run limiting to avoid ".....", "-----", etc.
    if normalize_punct_runs and max_punct_run >= 1:
        s = _limit_punct_runs(s, max_punct_run, punct_chars)

    # Optional removal of pure "bar" lines (ASCII art separators)
    if strip_ascii_bars:
        s = _drop_ascii_bar_lines(s, bar_min_len=max(1, bar_min_len), charset=punct_chars)

    # Now cap runs of spaces/newlines
    if max_space_run >= 1:
        s = _limit_runs(s, " ", max_space_run)
    if max_newline_run >= 1:
        s = _limit_runs(s, "\n", max_newline_run)

    # Final clean-up around newlines and global strip
    if strip_line_edges:
        s = _RE_SPACES_BEFORE_NL.sub("\n", s)
        s = _RE_SPACES_AFTER_NL.sub("\n", s)
    return s.strip()


def _maybe_normalize_text(text: str, config: ChunkConfig) -> str:
    """
    Apply normalization if enabled; safe no-op if disabled.
    """
    if not config.normalize_whitespace:
        return text
    return _normalize_text_ws(
        text=text,
        max_space_run=max(1, config.max_space_run),
        max_newline_run=max(1, config.max_newline_run),
        strip_line_edges=config.strip_line_edges,
        normalize_punct_runs=config.normalize_punct_runs,
        max_punct_run=max(1, config.max_punct_run),
        strip_ascii_bars=config.strip_ascii_bars,
        bar_min_len=max(1, config.bar_min_len),
        punct_chars=config.punct_chars,
    )


# --------------------------------------------------------------------------------------------------
# Device + UTF-8 helpers
# --------------------------------------------------------------------------------------------------

def _normalize_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """
    Normalize desired device; if CUDA requested but unavailable, fall back to CPU.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return dev


def _to_device_u8(text: str, device: Optional[Union[str, torch.device]]) -> torch.Tensor:
    """
    Safe encoder: use bytearray->torch.tensor to avoid read-only buffer warnings and maximize portability.
    CUDA path uses a pinned-memory hop before non_blocking GPU transfer for better throughput.
    """
    b = text.encode("utf-8")
    cpu_t = torch.tensor(bytearray(b), dtype=torch.uint8)
    if device is None:
        return cpu_t
    dev = torch.device(device)
    if dev.type == "cuda":
        try:
            cpu_t = cpu_t.pin_memory()
        except Exception:
            pass
        return cpu_t.to(dev, non_blocking=True)
    return cpu_t.to(dev, non_blocking=False)


def _utf8_start_mask(u8: torch.Tensor) -> torch.Tensor:
    return (u8 & 0b11000000) != 0b10000000


def _utf8_start_indices(mask: torch.Tensor) -> torch.Tensor:
    return mask.nonzero(as_tuple=False).flatten()


def _align_end_idx(end_idx: int, start_idx: torch.Tensor, N: int) -> int:
    """
    Align end index to the most recent UTF-8 start boundary <= end_idx.
    """
    if end_idx >= N:
        return N
    pos = torch.searchsorted(start_idx, torch.tensor([end_idx], device=start_idx.device), right=True).item() - 1
    if pos >= 0:
        return int(start_idx[pos].item())
    return 0


def _align_start_idx(start_idx_candidate: int, start_idx: torch.Tensor, N: int) -> int:
    """
    Align start index to the earliest UTF-8 start boundary >= start_idx_candidate.
    """
    if start_idx_candidate <= 0:
        return 0
    pos = torch.searchsorted(start_idx, torch.tensor([start_idx_candidate], device=start_idx.device), right=False).item()
    if pos < start_idx.numel():
        return int(start_idx[pos].item())
    return N


def _slice_to_bytes(u8: torch.Tensor, start: int, end: int) -> bytes:
    """
    Fast, portable tensor-slice -> bytes conversion with CUDA safety:
      - Uses a synchronous GPU->CPU hop to prevent partial reads (fixes sporadic UTF-8 decode errors).
      - Prefers .numpy().tobytes() on CPU tensors; falls back to a safe path if needed.
    """
    if end <= start:
        return b""
    view = u8.narrow(0, start, end - start)
    if view.device.type == "cuda":
        view = view.to("cpu", non_blocking=False)
    if not view.is_contiguous():
        view = view.contiguous()
    try:
        return view.numpy().tobytes()
    except Exception:
        return bytes(view.tolist())


# --------------------------------------------------------------------------------------------------
# Pattern scanning (fast-path for 1/2 bytes, generic blockwise fallback)
# --------------------------------------------------------------------------------------------------

@torch.no_grad()
def _find_occurrences_len1(u8: torch.Tensor, b0: int) -> torch.Tensor:
    """
    Pure vector path for 1-byte patterns: return start indices where u8 == b0.
    """
    eq = (u8 == int(b0))
    idxs = torch.nonzero(eq, as_tuple=False).flatten()
    return idxs.to(torch.int64)


@torch.no_grad()
def _find_occurrences_len2(u8: torch.Tensor, b0: int, b1: int) -> torch.Tensor:
    """
    Pure vector path for 2-byte patterns: return start indices where u8[i]==b0 and u8[i+1]==b1.
    """
    N = int(u8.numel())
    if N < 2:
        return torch.empty(0, dtype=torch.int64, device=u8.device)
    eq0 = (u8[:-1] == int(b0))
    eq1 = (u8[1:] == int(b1))
    idxs = torch.nonzero(eq0 & eq1, as_tuple=False).flatten()
    return idxs.to(torch.int64)


@torch.no_grad()
def _find_occurrences_blockwise(
    u8: torch.Tensor,
    pat: torch.Tensor,
    mem_block_cap_bytes: int,
) -> torch.Tensor:
    """
    Generic blockwise unfolded-equality search for pattern start positions. Returns sorted unique int64 indices.
    """
    N = int(u8.numel())
    m = int(pat.numel())
    if m == 0 or N < m:
        return torch.empty(0, dtype=torch.int64, device=u8.device)

    rows = max(1, mem_block_cap_bytes // max(1, m))
    out_pos: List[torch.Tensor] = []
    i = 0
    while i < N:
        end = min(N, i + rows + m - 1)
        block = u8[i:end]
        M = int(block.numel())
        if M < m:
            i += rows
            continue
        win = block.unfold(0, m, 1)          # [M - m + 1, m]
        matches = win.eq(pat).all(dim=-1)    # [M - m + 1]
        idxs = torch.nonzero(matches, as_tuple=False).flatten()
        if idxs.numel() > 0:
            out_pos.append((idxs + i).to(torch.int64))
        i += rows

    if out_pos:
        cat = torch.cat(out_pos)
        cat, _ = torch.sort(cat.unique())
        return cat
    return torch.empty(0, dtype=torch.int64, device=u8.device)


# --------------------------------------------------------------------------------------------------
# Optional compilation of hot kernels (best-effort)
# --------------------------------------------------------------------------------------------------

_COMPILE_APPLIED = False

def _maybe_compile_kernels(enable: bool) -> None:
    """
    Best-effort torch.compile of hot small kernels for minor wins on supported stacks.
    No-op on failure or when already applied.
    """
    global _COMPILE_APPLIED, _find_occurrences_len1, _find_occurrences_len2
    if _COMPILE_APPLIED or not enable:
        return
    if not hasattr(torch, "compile"):
        return
    try:
        _find_occurrences_len1 = torch.compile(_find_occurrences_len1, fullgraph=False, dynamic=True)  # type: ignore
        _find_occurrences_len2 = torch.compile(_find_occurrences_len2, fullgraph=False, dynamic=True)  # type: ignore
        _COMPILE_APPLIED = True
    except Exception:
        # Compilation is optional; ignore failures.
        pass


# --------------------------------------------------------------------------------------------------
# Boundary construction
# --------------------------------------------------------------------------------------------------

@torch.no_grad()
def _build_boundaries_tiers(
    u8: torch.Tensor,
    separators: Sequence[str],
    keep: KeepMode,
    mem_block_cap_bytes: int,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Build boundary arrays per separator tier; append sentinel N to ensure closure.
    Fast-path 1/2-byte patterns; fallback to generic search otherwise.
    """
    device = u8.device
    N = int(u8.numel())
    tiers: List[torch.Tensor] = []
    lengths: List[int] = []

    for s in separators:
        if s == "":
            tiers.append(torch.tensor([N], device=device, dtype=torch.int64))
            lengths.append(0)
            continue

        sb = s.encode("utf-8")
        m = len(sb)
        lengths.append(m)

        if m == 0 or m > N:
            tiers.append(torch.tensor([N], device=device, dtype=torch.int64))
            continue

        occ: torch.Tensor
        if m == 1:
            occ = _find_occurrences_len1(u8, sb[0])
        elif m == 2:
            occ = _find_occurrences_len2(u8, sb[0], sb[1])
        else:
            pat = torch.tensor(bytearray(sb), dtype=torch.uint8, device=device)
            occ = _find_occurrences_blockwise(u8, pat, mem_block_cap_bytes)

        if occ.numel() == 0:
            tiers.append(torch.tensor([N], device=device, dtype=torch.int64))
            continue

        if keep == "end":
            bnd = occ + m
        elif keep == "start":
            bnd = occ
        else:
            bnd = occ

        bnd = bnd[(bnd > 0) & (bnd < N)]

        if m > 2 and bnd.numel() > 0:
            bnd, _ = torch.sort(bnd.unique())

        if bnd.numel() > 0:
            bnd = torch.cat([bnd, torch.tensor([N], device=device, dtype=torch.int64)], dim=0)
        else:
            bnd = torch.tensor([N], device=device, dtype=torch.int64)

        tiers.append(bnd)

    return tiers, lengths


def _search_rightmost_leq_greater_than(
    boundaries: torch.Tensor,
    target_leq: int,
    lo_exclusive: int,
) -> int:
    """
    Rightmost index j such that:
      boundaries[j] <= target_leq and boundaries[j] > lo_exclusive.
    Uses searchsorted to avoid device sync in Python loops.
    """
    if boundaries.numel() == 0:
        return -1
    j_hi = torch.searchsorted(boundaries, torch.tensor([target_leq], device=boundaries.device), right=True).item() - 1
    if j_hi < 0:
        return -1
    lo_cut = torch.searchsorted(boundaries, torch.tensor([lo_exclusive], device=boundaries.device), right=True).item()
    if j_hi >= lo_cut:
        return int(j_hi)
    return -1


# --------------------------------------------------------------------------------------------------
# 64-bit rolling hash (FNV-1a)
# --------------------------------------------------------------------------------------------------

def _fnv1a64_bytes(data: bytes) -> int:
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _fnv1a64_tensor_slice(u8: torch.Tensor, start: int, end: int) -> int:
    data = _slice_to_bytes(u8, start, end)
    return _fnv1a64_bytes(data)


# --------------------------------------------------------------------------------------------------
# FHPA packer (overlap progression fix + tail stop)
# --------------------------------------------------------------------------------------------------

@torch.no_grad()
def _pack_fhpa(
    u8: torch.Tensor,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str],
    keep: KeepMode,
    boundaries_tiers: List[torch.Tensor],
    sep_lengths: List[int],
    start_idx: torch.Tensor,
    token_boundaries: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int]]:
    """
    Greedy hierarchical pack with UTF-8 alignment and fused overlap.
    Overlap rule (fix): if previous chunk length <= overlap, next chunk starts at previous end (no sliding-by-1).
    """
    N = int(u8.numel())
    starts_token = None
    if token_boundaries:
        tb = torch.tensor(list(token_boundaries), dtype=torch.int64, device=u8.device)
        tb = tb[(tb >= 0) & (tb <= N)]
        if tb.numel() == 0 or tb[0].item() != 0:
            tb = torch.cat([torch.tensor([0], device=u8.device, dtype=torch.int64), tb])
        if tb[-1].item() != N:
            tb = torch.cat([tb, torch.tensor([N], device=u8.device, dtype=torch.int64)])
        starts_token = tb

    def tok_leq(x: int) -> int:
        if starts_token is None:
            return x
        j = torch.searchsorted(starts_token, torch.tensor([x], device=u8.device), right=True).item() - 1
        return int(starts_token[max(0, j)].item())

    def tok_geq(x: int) -> int:
        if starts_token is None:
            return x
        j = torch.searchsorted(starts_token, torch.tensor([x], device=u8.device), right=False).item()
        if j < starts_token.numel():
            return int(starts_token[j].item())
        return int(starts_token[-1].item())

    spans: List[Tuple[int, int]] = []
    b = 0
    while b < N:
        t_raw = min(N, b + chunk_size)
        t = tok_leq(t_raw)
        if t <= b:
            t = min(N, b + chunk_size)

        chosen_end = -1
        chosen_sep_len = 0
        for k, bnd in enumerate(boundaries_tiers):
            j = _search_rightmost_leq_greater_than(bnd, t, b)
            if j >= 0:
                chosen_end = int(bnd[j].item())
                chosen_sep_len = sep_lengths[k]
                break
        if chosen_end < 0:
            chosen_end = t
            chosen_sep_len = 0

        e = _align_end_idx(chosen_end, start_idx, N)
        e = tok_leq(e)
        if e <= b:
            e = _align_end_idx(min(N, b + chunk_size), start_idx, N)
            e = tok_leq(e)
            if e <= b:
                e = min(N, b + 1)

        spans.append((b, e))

        if e == N:
            break

        prev_len = e - b
        if chunk_overlap <= 0:
            nb = e
        else:
            nb = e - chunk_overlap
            if nb <= b:
                nb = e
        if keep == "none" and chosen_sep_len > 0:
            nb = max(nb, e + chosen_sep_len)

        nb = tok_geq(nb)
        b = _align_start_idx(nb, start_idx, N)
    return spans


# --------------------------------------------------------------------------------------------------
# Public text chunker
# --------------------------------------------------------------------------------------------------

@torch.no_grad()
def chunk_text(
    text: str,
    config: ChunkConfig,
) -> List[Tuple[str, int, int]] | List[str]:
    """
    Chunk a single UTF-8 text with FHPA.
    Notes:
      - If normalization is enabled, 'text' is pre-normalized so long spaces, blank-line bursts, and noisy bars are eliminated.
      - If return_offsets=True, offsets are byte positions in the normalized text (post-normalization).
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    overlap = max(0, min(config.chunk_overlap, config.chunk_size - 1))

    _maybe_compile_kernels(config.compile)

    # Optional normalization to prevent long-space and noisy punctuation artifacts end-to-end.
    text = _maybe_normalize_text(text, config)

    device = _normalize_device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    u8 = _to_device_u8(text, device)
    N = int(u8.numel())
    if N == 0:
        return []

    start_mask = _utf8_start_mask(u8)
    start_idx = _utf8_start_indices(start_mask)

    tiers, sep_lens = _build_boundaries_tiers(
        u8=u8,
        separators=config.separators,
        keep=config.keep_separator,
        mem_block_cap_bytes=config.mem_block_cap_bytes,
    )

    spans = _pack_fhpa(
        u8=u8,
        chunk_size=config.chunk_size,
        chunk_overlap=overlap,
        separators=config.separators,
        keep=config.keep_separator,
        boundaries_tiers=tiers,
        sep_lengths=sep_lens,
        start_idx=start_idx,
        token_boundaries=config.token_boundaries,
    )

    out: List[Tuple[str, int, int]] | List[str] = []
    seen: set[Tuple[int, int]] = set()
    for s0, e0 in spans:
        bt = _slice_to_bytes(u8, s0, e0)
        try:
            st = bt.decode("utf-8", errors="strict") if bt else ""
        except UnicodeDecodeError:
            st = bt.decode("utf-8", errors="replace")
        if config.drop_blank_chunks and not st.strip():
            continue
        if config.deduplicate:
            h = _fnv1a64_bytes(bt)
            key = (h, e0 - s0)
            if key in seen:
                continue
            seen.add(key)
        out.append((st, s0, e0) if config.return_offsets else st)
    return out


# --------------------------------------------------------------------------------------------------
# File loaders
# --------------------------------------------------------------------------------------------------

def _collect_text_leaves(obj: Any) -> List[str]:
    out: List[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, (int, float, bool)):
        out.append(str(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_text_leaves(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_collect_text_leaves(v))
    return out


def _flatten_json_text(obj: Any, collect_fields: Optional[Sequence[str]] = None) -> str:
    texts: List[str] = []
    if collect_fields:
        field_paths = [tuple(f.split(".")) for f in collect_fields]

        def match_and_collect(o: Any, path: Tuple[str, ...]) -> None:
            if not path:
                texts.extend(_collect_text_leaves(o))
                return
            key = path[0]
            sub = None
            if isinstance(o, dict) and key in o:
                sub = o[key]
            elif isinstance(o, (list, tuple)) and key.isdigit():
                idx = int(key)
                if 0 <= idx < len(o):
                    sub = o[idx]
            if sub is not None:
                match_and_collect(sub, path[1:])

        for p in field_paths:
            match_and_collect(obj, p)
    else:
        texts.extend(_collect_text_leaves(obj))
    return "\n".join(t for t in texts if t)


def _html_to_text(html: str) -> str:
    """
    Robust HTML -> text with scripting/style removal and light structure retention.
    Generated text may still contain extra whitespace; end-to-end normalization will clean it.
    """
    if _BS4_AVAILABLE:
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    txt = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.IGNORECASE)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"&nbsp;", " " , txt)
    txt = re.sub(r"&amp;", "&", txt)
    txt = re.sub(r"&lt;", "<", txt)
    txt = re.sub(r"&gt;", ">", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()


def _read_text_file(p: Path) -> str:
    with open(p, "rb") as f:
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _read_html_file(p: Path, strip: bool) -> str:
    s = _read_text_file(p)
    return _html_to_text(s) if strip else s


def _read_json_file(p: Path, fields: Optional[Sequence[str]]) -> str:
    with open(p, "rb") as f:
        data = f.read()
    try:
        obj = json.loads(data)
    except Exception:
        texts = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                texts.append(_flatten_json_text(o, fields))
            except Exception:
                continue
        return "\n".join(t for t in texts if t)
    return _flatten_json_text(obj, fields)


def _read_jsonl_file(p: Path, fields: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    with open(p, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out.append(_flatten_json_text(obj, fields))
            except Exception:
                continue
    return out


def _read_yaml_file(p: Path, fields: Optional[Sequence[str]]) -> str:
    if not _YAML_AVAILABLE:
        warnings.warn(f"PyYAML not available; treating YAML as plain text: {p}")
        return _read_text_file(p)
    with open(p, "rb") as f:
        data = f.read()
    try:
        obj = _yaml.safe_load(data)  # type: ignore
        return _flatten_json_text(obj, fields)
    except Exception:
        return _read_text_file(p)


def _read_csv_file(p: Path, text_columns: Optional[Sequence[str]]) -> str:
    if _PANDAS_AVAILABLE:
        try:
            df = pd.read_csv(p)
            cols: List[str]
            if text_columns:
                cols = [c for c in text_columns if c in df.columns]
            else:
                cols = [c for c in df.columns if df[c].dtype == object]
            if not cols:
                cols = list(df.columns)
            df = df[cols].astype(str)
            return "\n".join(" ".join(row) for row in df.values.tolist())
        except Exception:
            pass
    out_lines: List[str] = []
    with open(p, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            f.seek(0)
            reader2 = csv.reader(f)
            for row in reader2:
                out_lines.append(" ".join(row))
            return "\n".join(out_lines)
        fieldnames = reader.fieldnames
        cols = list(text_columns) if text_columns else fieldnames
        for row in reader:
            vals = [str(row.get(c, "")) for c in cols]
            out_lines.append(" ".join(vals))
    return "\n".join(out_lines)


def _read_parquet_file(p: Path, text_columns: Optional[Sequence[str]]) -> str:
    if _PYARROW_AVAILABLE:
        try:
            table = pq.read_table(p)
            cols: List[str]
            if text_columns:
                cols = [c for c in text_columns if c in table.column_names]
            else:
                cols = list(table.column_names)
            parts: List[str] = []
            for col in cols:
                arr = table[col].to_pylist()
                parts.extend(str(x) for x in arr)
            return "\n".join(parts)
        except Exception:
            pass
    if _PANDAS_AVAILABLE:
        try:
            df = pd.read_parquet(p)
            cols = list(text_columns) if text_columns else list(df.columns)
            df = df[cols].astype(str)
            return "\n".join(" ".join(row) for row in df.values.tolist())
        except Exception:
            pass
    raise RuntimeError(f"Neither pyarrow nor pandas can read Parquet file: {p}")


def _post_read_normalize_text(text: str, config: ChunkConfig) -> str:
    """
    Normalize immediately after reading a file (applies to all formats).
    """
    return _maybe_normalize_text(text, config)


def _read_path_to_documents(fp: Path, config: ChunkConfig) -> List[Document]:
    """
    Worker: read a single file path into one or more Documents, normalizing whitespace if enabled.
    Returns an empty list on failure, with a warning.
    """
    docs_local: List[Document] = []
    suffix = fp.suffix.lower()
    try:
        if suffix in [".txt", ".md", ".log", ".rst"]:
            txt = _post_read_normalize_text(_read_text_file(fp), config)
            if txt or not config.drop_blank_chunks:
                docs_local.append(Document(id=str(fp), text=txt, meta={"path": str(fp), "type": "text"}))
        elif suffix in [".html", ".htm"]:
            txt = _post_read_normalize_text(_read_html_file(fp, strip=config.html_strip), config)
            if txt or not config.drop_blank_chunks:
                docs_local.append(Document(id=str(fp), text=txt, meta={"path": str(fp), "type": "html"}))
        elif suffix == ".json":
            txt = _post_read_normalize_text(_read_json_file(fp, fields=config.json_text_fields), config)
            if txt or not config.drop_blank_chunks:
                docs_local.append(Document(id=str(fp), text=txt, meta={"path": str(fp), "type": "json"}))
        elif suffix in [".yaml", ".yml"]:
            txt = _post_read_normalize_text(_read_yaml_file(fp, fields=config.yaml_text_fields), config)
            if txt or not config.drop_blank_chunks:
                docs_local.append(Document(id=str(fp), text=txt, meta={"path": str(fp), "type": "yaml"}))
        elif suffix in [".jsonl", ".ndjson"]:
            records = _read_jsonl_file(fp, fields=config.json_text_fields)
            for i, rec in enumerate(records):
                recn = _post_read_normalize_text(rec, config)
                if recn or not config.drop_blank_chunks:
                    docs_local.append(Document(
                        id=f"{fp}::record:{i}",
                        text=recn,
                        meta={"path": str(fp), "type": "jsonl", "record": i},
                    ))
        elif suffix == ".csv":
            txt = _post_read_normalize_text(_read_csv_file(fp, text_columns=config.csv_text_columns), config)
            if txt or not config.drop_blank_chunks:
                docs_local.append(Document(id=str(fp), text=txt, meta={"path": str(fp), "type": "csv"}))
        elif suffix == ".parquet":
            txt = _post_read_normalize_text(_read_parquet_file(fp, text_columns=config.parquet_text_columns), config)
            if txt or not config.drop_blank_chunks:
                docs_local.append(Document(id=str(fp), text=txt, meta={"path": str(fp), "type": "parquet"}))
        else:
            txt = _post_read_normalize_text(_read_text_file(fp), config)
            if txt or not config.drop_blank_chunks:
                docs_local.append(Document(id=str(fp), text=txt, meta={"path": str(fp), "type": "unknown"}))
    except Exception as ex:
        warnings.warn(f"Failed to load {fp}: {ex}")
    return docs_local


def load_documents(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    config: ChunkConfig,
) -> List[Document]:
    """
    End-to-end loader with parallel IO and whitespace/noise normalization.
    Preserves deterministic order by initial file order; multi-record formats (jsonl) preserve in-file order.
    """
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = list(paths)
    resolved: List[Path] = []
    for p in path_list:
        pp = Path(p)
        if any(ch in str(pp) for ch in "*?[]"):
            resolved.extend([Path(s) for s in sorted(map(str, Path().glob(str(pp))))])
        elif pp.is_dir():
            resolved.extend(sorted(pp.rglob("*")))
        else:
            resolved.append(pp)
    files = [p for p in resolved if p.is_file()]

    if not files:
        return []

    if config.num_workers_io <= 1:
        docs: List[Document] = []
        for fp in files:
            docs.extend(_read_path_to_documents(fp, config))
        return docs

    docs_out: List[Document] = []
    results_ordered: List[List[Document]] = [[] for _ in range(len(files))]
    with ThreadPoolExecutor(max_workers=config.num_workers_io, thread_name_prefix="io") as ex:
        futs = {ex.submit(_read_path_to_documents, fp, config): i for i, fp in enumerate(files)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results_ordered[i] = fut.result()
            except Exception as ex:
                warnings.warn(f"Reader failed on index={i}, path={files[i]}: {ex}")
                results_ordered[i] = []
    for part in results_ordered:
        docs_out.extend(part)
    return docs_out


# --------------------------------------------------------------------------------------------------
# Document chunking + writers
# --------------------------------------------------------------------------------------------------

def chunk_document(doc: Document, config: ChunkConfig) -> List[Chunk]:
    """
    Chunk a single Document into Chunk objects, hashing text if requested.
    Drops blank chunks when configured.
    """
    chunks_raw = chunk_text(doc.text, config)
    chunks: List[Chunk] = []
    if config.return_offsets:
        for i, (s, b0, e0) in enumerate(chunks_raw):  # type: ignore
            if config.drop_blank_chunks and not str(s).strip():
                continue
            h = _fnv1a64_bytes(str(s).encode("utf-8")) if config.output_hash else None
            chunks.append(Chunk(doc_id=doc.id, index=i, start=b0, end=e0, text=str(s), hash64=h))
    else:
        for i, s in enumerate(chunks_raw):  # type: ignore
            if config.drop_blank_chunks and not str(s).strip():
                continue
            h = _fnv1a64_bytes(str(s).encode("utf-8")) if config.output_hash else None
            chunks.append(Chunk(doc_id=doc.id, index=i, start=-1, end=-1, text=str(s), hash64=h))
    return chunks


def chunk_documents(docs: Sequence[Document], config: ChunkConfig) -> List[Chunk]:
    """
    Chunk a sequence of Documents.
    Parallelizes on CPU when num_workers_chunk > 1; serializes when targeting CUDA to avoid device contention.
    """
    if not docs:
        return []

    dev = _normalize_device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    effective_workers = config.num_workers_chunk if dev.type != "cuda" else 1

    if effective_workers <= 1 or len(docs) < 2:
        out: List[Chunk] = []
        for d in docs:
            out.extend(chunk_document(d, config))
        return out

    chunks_ordered: List[List[Chunk]] = [[] for _ in range(len(docs))]
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="chunk") as ex:
        futs = {ex.submit(chunk_document, d, config): i for i, d in enumerate(docs)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                chunks_ordered[i] = fut.result()
            except Exception as ex:
                warnings.warn(f"Chunking failed for doc index={i}, id={docs[i].id}: {ex}")
                chunks_ordered[i] = []
    out_all: List[Chunk] = []
    for part in chunks_ordered:
        out_all.extend(part)
    return out_all


def write_chunks_jsonl(chunks: Sequence[Chunk], f: Union[str, Path, io.TextIOBase]) -> None:
    """
    Stream chunks to JSONL with UTF-8 encoding.
    """
    close = False
    if isinstance(f, (str, Path)):
        f = open(f, "w", encoding="utf-8")
        close = True
    try:
        for c in chunks:
            obj = {
                "doc_id": c.doc_id,
                "index": c.index,
                "start": c.start,
                "end": c.end,
                "text": c.text,
                "hash64": c.hash64,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        if close:
            f.close()


def write_chunks_csv(chunks: Sequence[Chunk], f: Union[str, Path, io.TextIOBase]) -> None:
    """
    Stream chunks to CSV with header.
    """
    close = False
    if isinstance(f, (str, Path)):
        f = open(f, "w", newline="", encoding="utf-8")
        close = True
    try:
        w = csv.writer(f)
        w.writerow(["doc_id", "index", "start", "end", "hash64", "text"])
        for c in chunks:
            w.writerow([c.doc_id, c.index, c.start, c.end, c.hash64, c.text])
    finally:
        if close:
            f.close()


def write_chunks_parquet(chunks: Sequence[Chunk], f: Union[str, Path]) -> None:
    """
    Write chunks to Parquet via pandas or pyarrow.
    """
    rows = [{
        "doc_id": c.doc_id,
        "index": c.index,
        "start": c.start,
        "end": c.end,
        "hash64": c.hash64,
        "text": c.text,
    } for c in chunks]
    if _PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
        df.to_parquet(f, index=False)
        return
    if _PYARROW_AVAILABLE:
        table = pa.Table.from_pylist(rows)  # type: ignore
        pq.write_table(table, f)            # type: ignore
        return
    raise RuntimeError("Cannot write parquet: pandas/pyarrow not available")


# --------------------------------------------------------------------------------------------------
# Distributed helpers
# --------------------------------------------------------------------------------------------------

def _dist_world() -> Tuple[int, int]:
    if _DIST_AVAILABLE and dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _shard_items_even(items: Sequence[Any], rank: int, world: int) -> List[Any]:
    return [items[i] for i in range(rank, len(items), world)]


def _gather_object_list(local_obj: Any) -> List[Any]:
    rank, world = _dist_world()
    buf: List[Any] = [None for _ in range(world)]  # type: ignore
    dist.all_gather_object(buf, local_obj)  # type: ignore
    return buf


def _reduce_sum_int(val: int, dst: int = 0) -> int:
    t = torch.tensor([val], dtype=torch.int64)
    dist.reduce(t, dst=dst, op=dist.ReduceOp.SUM)
    return int(t.item())


def _global_dedup_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Stable global deduplication by 64-bit hash, keeping lexicographically smallest (doc_id, start, end).
    """
    seen: Dict[int, Tuple[str, int, int]] = {}
    out: List[Chunk] = []
    for c in chunks:
        h = c.hash64 if c.hash64 is not None else _fnv1a64_bytes(c.text.encode("utf-8"))
        probe = seen.get(h)
        key_span = (c.doc_id, c.start, c.end)
        if probe is None or key_span < probe:
            seen[h] = key_span
    for c in chunks:
        h = c.hash64 if c.hash64 is not None else _fnv1a64_bytes(c.text.encode("utf-8"))
        if seen.get(h) == (c.doc_id, c.start, c.end):
            out.append(c)
    return out


def chunk_documents_distributed(
    docs: Sequence[Document],
    config: ChunkConfig,
) -> List[Chunk]:
    """
    Distributed chunking across initialized torch.distributed process group.
    Rank 0 gathers and globally deduplicates; other ranks optionally return their local chunks.
    """
    if not (_DIST_AVAILABLE and dist.is_available() and dist.is_initialized()):
        return chunk_documents(docs, config)

    rank, world = _dist_world()
    local_docs = _shard_items_even(docs, rank, world)
    local_chunks = chunk_documents(local_docs, config)

    gathered_lists: List[List[Chunk]] = _gather_object_list(local_chunks)  # type: ignore
    flat_all = [c for part in gathered_lists for c in part]

    if rank == 0:
        uniq = _global_dedup_chunks(flat_all)
        total_bytes = sum(len(c.text.encode("utf-8")) for c in uniq)
        _ = _reduce_sum_int(total_bytes, dst=0)
        return uniq

    total_bytes_local = sum(len(c.text.encode("utf-8")) for c in local_chunks)
    _ = _reduce_sum_int(total_bytes_local, dst=0)
    return local_chunks if config.return_local_when_distributed else []


# --------------------------------------------------------------------------------------------------
# High-level pipeline
# --------------------------------------------------------------------------------------------------

def chunk_from_files(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    config: ChunkConfig,
) -> List[Chunk]:
    """
    End-to-end pipeline: load -> normalize -> chunk (local or distributed) -> return chunk objects.
    """
    docs = load_documents(paths, config)
    if config.distributed and _DIST_AVAILABLE and dist.is_available() and dist.is_initialized():
        return chunk_documents_distributed(docs, config)
    return chunk_documents(docs, config)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="TorchChuck: hierarchical GPU/distributed text chunker")
    ap.add_argument("--in", dest="inp", type=str, required=True, help="Input path or glob (files or directories)")
    ap.add_argument("--out", dest="out", type=str, required=True, help="Output path (.jsonl or .csv or .parquet)")
    ap.add_argument("--chunk-size", type=int, default=1024, help="Max chunk size in bytes (or tokens if token_boundaries provided)")
    ap.add_argument("--chunk-overlap", type=int, default=128, help="Chunk overlap size")
    ap.add_argument("--separators", type=str, nargs="*", default=["\n\n", "\n", " ", ""], help="Separator priority list (highest to lowest)")
    ap.add_argument("--keep-separator", type=str, choices=["end", "start", "none"], default="end", help="Keep separator policy")
    ap.add_argument("--device", type=str, default=None, help='"cuda" or "cpu"; default auto')
    ap.add_argument("--distributed", action="store_true", help="Use torch.distributed (must be initialized externally)")
    ap.add_argument("--return-offsets", dest="return_offsets", action="store_true", default=True, help="Include (start,end) byte offsets in outputs (post-normalization)")
    ap.add_argument("--no-return-offsets", dest="return_offsets", action="store_false", help="Disable offsets in outputs")
    ap.add_argument("--dedup", action="store_true", help="Enable local dedup (distributed path performs global dedup)")
    ap.add_argument("--compile", action="store_true", help="Best-effort torch.compile for hot kernels")

    # Whitespace/noise normalization flags
    ap.add_argument("--normalize-whitespace", dest="normalize_ws", action="store_true", default=True, help="Enable whitespace normalization to avoid long spaces and blank lines")
    ap.add_argument("--no-normalize-whitespace", dest="normalize_ws", action="store_false", help="Disable whitespace normalization")
    ap.add_argument("--max-space-run", type=int, default=1, help="Cap consecutive spaces to this many (>=1)")
    ap.add_argument("--max-newline-run", type=int, default=2, help="Cap consecutive newlines to this many (>=1)")
    ap.add_argument("--no-strip-line-edges", dest="strip_line_edges", action="store_false", help="Do not strip spaces around newlines")
    ap.add_argument("--normalize-punct-runs", dest="normalize_punct_runs", action="store_true", default=True, help="Limit long punctuation runs like '.....' or '------'")
    ap.add_argument("--no-normalize-punct-runs", dest="normalize_punct_runs", action="store_false", help="Do not limit punctuation runs")
    ap.add_argument("--max-punct-run", type=int, default=3, help="Cap punctuation runs to this many (>=1)")
    ap.add_argument("--strip-ascii-bars", dest="strip_ascii_bars", action="store_true", default=True, help="Remove punctuation-only bar lines like '-----'")
    ap.add_argument("--no-strip-ascii-bars", dest="strip_ascii_bars", action="store_false", help="Keep punctuation-only bar lines")
    ap.add_argument("--bar-min-len", type=int, default=6, help="Minimum bar length to strip when --strip-ascii-bars is enabled (>=1)")
    ap.add_argument("--punct-chars", type=str, default="._-~=*#•·–—─━═", help="Characters considered for punct-run limiting and bar-line stripping")

    # Parallelism + filtering
    ap.add_argument("--num-workers-io", type=int, default=max(1, min(8, (os.cpu_count() or 2))), help="Parallel IO workers")
    ap.add_argument("--num-workers-chunk", type=int, default=max(1, min(8, (os.cpu_count() or 2))), help="Parallel chunking workers (CPU only)")
    ap.add_argument("--drop-blank-chunks", action="store_true", default=True, help="Drop empty/whitespace-only chunks")

    args = ap.parse_args()

    cfg = ChunkConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=tuple(args.separators),
        keep_separator=args.keep_separator,  # type: ignore
        device=args.device,
        distributed=args.distributed,
        return_offsets=args.return_offsets,
        deduplicate=args.dedup,
        compile=args.compile,

        # Normalization
        normalize_whitespace=args.normalize_ws,
        max_space_run=max(1, args.max_space_run),
        max_newline_run=max(1, args.max_newline_run),
        strip_line_edges=args.strip_line_edges,
        normalize_punct_runs=args.normalize_punct_runs,
        max_punct_run=max(1, args.max_punct_run),
        strip_ascii_bars=args.strip_ascii_bars,
        bar_min_len=max(1, args.bar_min_len),
        punct_chars=args.punct_chars,

        # Parallel + output filtering
        num_workers_io=max(1, args.num_workers_io),
        num_workers_chunk=max(1, args.num_workers_chunk),
        drop_blank_chunks=args.drop_blank_chunks,
    )

    chunks = chunk_from_files(args.inp, cfg)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.suffix.lower() == ".jsonl":
        write_chunks_jsonl(chunks, outp)
    elif outp.suffix.lower() == ".csv":
        write_chunks_csv(chunks, outp)
    elif outp.suffix.lower() == ".parquet":
        write_chunks_parquet(chunks, outp)
    else:
        write_chunks_jsonl(chunks, outp)
