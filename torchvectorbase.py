# file: torchvectorbase.py
# --------------------------------------------------------------------------------------------------
# VectorBase — torch-native, distributed, high-performance vector engine.
# This module provides:
#   - Torch-native IVF + OPQ/PQ index with LUT scanning and exact re-rank
#   - Flat (exact) baseline
#   - Robust embedding ingestion and metadata bitset filtering
#   - Duplicate-aware insert
#   - Generalized, resilient search with adaptive nprobe, non-empty cell masking, and flat fallback
#   - CPU/GPU support with optional AMP and torch.compile
#
# Notes:
# - This revision fixes empty-results by:
#   (1) Masking empty IVF cells during probe (never probe empty lists)
#   (2) Adaptive probe sizing (cap by non-empty lists)
#   (3) Flat fallback when no candidates or insufficient candidates are found
#   (4) Clamping nlist to [1, N] to avoid pathological k-means configurations
# - CUDA/ROCm robustness and speed updates:
#   (A) Strict dtype alignment for all matmul paths (prevents "expected scalar type" errors)
#   (B) OPQ.apply now internally casts to rotation dtype and back (prevents dtype mismatch)
#   (C) Coarse probe casts centroids to query dtype (prevents dtype mismatch on GPU)
#   (D) Exact re-rank and flat paths align dtypes before matmul/ops
#   (E) All core kernels ensure contiguity and use promote-types to the widest float where needed
#   (F) Vectorized LUT scanning across subspaces (accelerates IVF scan significantly on GPU)
#   (G) Vectorized reorder-by-cell using argsort (removes Python loop and per-cell nonzero scans)
#   (H) Runtime fast-math knobs (TF32 on Ampere+; high matmul precision) enabled when available
# - All explanations are in comments. No extra text is printed.
# --------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import os
import sys
import time
import types
import random
import struct
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist


# -------------------------------
# System / Runtime Configuration
# -------------------------------

def _configure_torch_runtime() -> None:
    # Enable faster matmul where supported (keeps numerical behavior stable for this use-case).
    # On NVIDIA CUDA: enable TF32 for conv/matmul if available. On ROCm this is a no-op.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass


_configure_torch_runtime()


def _pick_device(preferred: Optional[str] = None) -> torch.device:
    # Returns an appropriate device for compute (cuda if available). Works on CUDA and ROCm backends.
    if preferred is not None:
        dev = torch.device(preferred)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but CUDA is not available")
        return dev
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _to_dtype_optim(d: int) -> torch.dtype:
    # Choose dtype for index storage (fp16 on GPU for larger dims; float32 otherwise).
    if torch.cuda.is_available():
        return torch.float16 if d >= 256 else torch.float32
    return torch.float32


def _align_float_mm_dtypes(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Align dtypes for matmul-friendly computation across CUDA/ROCm. Promotes to widest float among inputs.
    # This prevents runtime errors like "expected scalar type Float but found Half".
    a_dt, b_dt = A.dtype, B.dtype
    if not (a_dt.is_floating_point and b_dt.is_floating_point):
        tgt = torch.float32
    else:
        tgt = torch.promote_types(a_dt, b_dt)
        if tgt not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            tgt = torch.float32
    if A.dtype != tgt:
        A = A.to(dtype=tgt)
    if B.dtype != tgt:
        B = B.to(dtype=tgt)
    return A, B


# -------------------------------
# Metrics and Distance Utilities
# -------------------------------

def l2_distances(X: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    # X: [n, d], Q: [b, d] => returns [b, n] of squared L2 distances. Ensures dtype alignment for GPU.
    X = X.contiguous()
    Q = Q.contiguous()
    X, Q = _align_float_mm_dtypes(X, Q)
    x2 = (X * X).sum(-1)                    # [n]
    q2 = (Q * Q).sum(-1)                    # [b]
    dots = Q @ X.T                          # [b, n]
    d2 = (q2[:, None] + x2[None, :] - 2.0 * dots).clamp_min_(0.0)
    return d2


def normalize_rows(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Row-wise L2 normalization; avoids division by zero. Keeps dtype as-is.
    nrm = torch.linalg.norm(X, dim=-1, keepdim=True).clamp_min_(eps)
    return X / nrm


def cosine_distances(Xn: torch.Tensor, Qn: torch.Tensor) -> torch.Tensor:
    # Xn, Qn normalized; returns 1 - cosine similarity; shape [b, n]. Ensures dtype alignment for GPU.
    Xn = Xn.contiguous()
    Qn = Qn.contiguous()
    Xn, Qn = _align_float_mm_dtypes(Xn, Qn)
    sims = Qn @ Xn.T
    return (1.0 - sims).clamp(min=0.0, max=2.0)


def metric_distances(X: torch.Tensor, Q: torch.Tensor, metric: Literal["l2", "cosine", "ip"]) -> torch.Tensor:
    # Unified distance computation; returns distance matrix [b, n], smaller is better.
    if metric == "l2":
        return l2_distances(X, Q)
    elif metric == "cosine":
        Xn = normalize_rows(X)
        Qn = normalize_rows(Q)
        return cosine_distances(Xn, Qn)
    elif metric == "ip":
        Xc, Qc = _align_float_mm_dtypes(X.contiguous(), Q.contiguous())
        return -(Qc @ Xc.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


try:
    torch_compile = torch.compile  # type: ignore[attr-defined]
except Exception:
    torch_compile = None


def _maybe_compile(fn):
    # Wrap a function in torch.compile if available; otherwise return as-is.
    if torch_compile is None:
        return fn
    try:
        return torch_compile(fn, fullgraph=True)  # type: ignore[misc]
    except Exception:
        return fn


@_maybe_compile
def fused_l2_topk(X: torch.Tensor, Q: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns (vals, idx) of top-k smallest L2 distances.
    d = l2_distances(X, Q)
    return torch.topk(d, k, dim=-1, largest=False)


def batched_topk_smallest(dist: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Distances shape [b, n] => smallest k per row.
    k_eff = min(k, dist.size(-1))
    if k_eff <= 0:
        return torch.empty((dist.size(0), 0), device=dist.device, dtype=dist.dtype), torch.empty(
            (dist.size(0), 0), device=dist.device, dtype=torch.long
        )
    vals, idx = torch.topk(dist, k_eff, dim=-1, largest=False)
    return vals, idx


def dedup_min_by_id(scores: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Keep min score per id (lower is better).
    uniq, inv = torch.unique(ids, return_inverse=True)
    best = torch.full((uniq.numel(),), float("inf"), device=scores.device, dtype=scores.dtype)
    best = best.scatter_reduce(0, inv, scores, reduce="amin", include_self=True)
    return best, uniq


# -------------------------------
# Embedding Adapter
# -------------------------------

class EmbeddingAdapter:
    # Adapter expecting:
    #   - embed_documents(texts: List[str]) -> List[List[float]]
    #   - embed_query(text: str) -> List[float]
    # Provides batching, preprocessing, device/dtype casting, optional normalization.

    def __init__(self, impl: Any, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
                 normalize: bool = False, batch_size: int = 64) -> None:
        self.impl = impl
        self.device = _pick_device() if device is None else device
        self.dtype = dtype or torch.float32
        self.normalize = normalize
        self.batch_size = batch_size

    @staticmethod
    def _preprocess(texts: Sequence[str]) -> List[str]:
        return [str(t).replace("\n", " ").strip() for t in texts]

    @torch.no_grad()
    def _call_embed_documents(self, texts: List[str]) -> Union[List[List[float]], torch.Tensor]:
        if hasattr(self.impl, "embed_documents"):
            return self.impl.embed_documents(texts)
        if isinstance(self.impl, EmbeddingAdapter) or hasattr(self.impl, "embed_texts"):
            return self.impl.embed_texts(texts)  # type: ignore[attr-defined]
        raise AttributeError("EmbeddingAdapter.impl must implement 'embed_documents' or provide 'embed_texts'.")

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        texts = self._preprocess(texts)
        texts = [t for t in texts if len(t) > 0]
        if len(texts) == 0:
            return torch.empty((0, 0), device=self.device, dtype=self.dtype)
        chunks: List[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            embs = self._call_embed_documents(batch)
            if torch.is_tensor(embs):
                X = embs.to(device=self.device, dtype=self.dtype, non_blocking=True)
            else:
                X = torch.tensor(embs, device=self.device, dtype=self.dtype)
            if self.normalize:
                X = normalize_rows(X)
            chunks.append(X)
        return torch.cat(chunks, dim=0).contiguous()

    @torch.no_grad()
    def embed_query(self, text: str) -> torch.Tensor:
        text = self._preprocess([text])[0]
        if len(text) == 0:
            return torch.empty((1, 0), device=self.device, dtype=self.dtype)
        if hasattr(self.impl, "embed_query"):
            vec = self.impl.embed_query(text)
            X = vec.view(1, -1) if torch.is_tensor(vec) else torch.tensor([vec], device=self.device, dtype=self.dtype)
        elif hasattr(self.impl, "embed_texts"):
            X = self.impl.embed_texts([text])  # type: ignore[attr-defined]
            if not torch.is_tensor(X):
                X = torch.tensor(X, device=self.device, dtype=self.dtype)
        else:
            embs = self.impl.embed_documents([text])
            X = torch.tensor(embs, device=self.device, dtype=self.dtype)
        X = X.to(device=self.device, dtype=self.dtype, non_blocking=True)
        if self.normalize:
            X = normalize_rows(X)
        return X


# -------------------------------
# KMeans, OPQ, and PQ (Torch-native)
# -------------------------------

class KMeans:
    # k-means with empty-cluster reinit for robustness (handles k > n gracefully by duplicates).

    def __init__(self, k: int, iters: int = 25, tol: float = 1e-4, seed: int = 0) -> None:
        self.k = int(k)
        self.iters = int(iters)
        self.tol = float(tol)
        self.seed = int(seed)
        self.C: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.shape
        g = torch.Generator(device=X.device).manual_seed(self.seed)
        idx = torch.randint(0, n, (self.k,), generator=g, device=X.device)
        C = X.index_select(0, idx).clone().contiguous()
        ones = torch.ones(n, 1, device=X.device, dtype=X.dtype)
        prev_inertia = None
        for _ in range(self.iters):
            x2 = (X * X).sum(-1, keepdim=True)      # [n,1]
            c2 = (C * C).sum(-1).unsqueeze(0)       # [1,k]
            dist2 = x2 + c2 - 2.0 * (X @ C.T)       # [n,k]
            a = dist2.argmin(-1)                    # [n]
            inertia = dist2.gather(1, a.view(-1, 1)).sum().item()
            C.zero_()
            S = torch.zeros(self.k, 1, device=X.device, dtype=X.dtype)
            C.index_add_(0, a, X)
            S.index_add_(0, a, ones)
            empty = (S.squeeze(1) == 0).nonzero(as_tuple=False).flatten()
            if empty.numel() > 0:
                refill = torch.randint(0, n, (empty.numel(),), generator=g, device=X.device)
                C.index_copy_(0, empty, X.index_select(0, refill))
                S.index_fill_(0, empty, 1.0)
            C /= S.clamp_min_(1.0)
            if prev_inertia is not None and abs(prev_inertia - inertia) <= self.tol * max(1.0, prev_inertia):
                break
            prev_inertia = inertia
        self.C = C.contiguous()
        return self.C


class OPQ:
    # Orthogonal rotation R \in O(d) via Procrustes-like updates.

    def __init__(self, d: int) -> None:
        self.d = int(d)
        self.R: torch.Tensor = torch.eye(d, dtype=torch.float32)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, iters: int = 4) -> torch.Tensor:
        # Fit rotation in float32 on the device of X for numerical robustness.
        R = torch.eye(self.d, device=X.device, dtype=torch.float32)
        for _ in range(iters):
            XR = (X.to(torch.float32) @ R)  # compute in float32 to stabilize SVD
            cov = (XR.T @ XR) / max(1, XR.shape[0])
            U, _, Vh = torch.linalg.svd(cov, full_matrices=False)
            R = (U @ Vh).to(torch.float32)
        self.R = R.contiguous()
        return self.R

    @torch.no_grad()
    def apply(self, X: torch.Tensor) -> torch.Tensor:
        # Apply rotation with internal dtype alignment: compute in R.dtype, return in original input dtype.
        Xin_dtype = X.dtype
        R = self.R.to(device=X.device, dtype=torch.float32)
        XR = (X.to(R.dtype) @ R)
        return XR.to(dtype=Xin_dtype)


class PQ:
    # Product Quantizer with M subspaces and B codewords per subspace (typically B=256).

    def __init__(self, d: int, m: int = 16, b: int = 256, iters: int = 20, seed: int = 0) -> None:
        assert d % m == 0, "d must be divisible by m"
        self.d, self.m, self.b, self.iters, self.seed = int(d), int(m), int(b), int(iters), int(seed)
        self.codebooks: List[torch.Tensor] = []  # List of [b, d/m] tensors

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> None:
        n, d = X.shape
        ds = d // self.m
        self.codebooks = []
        for i in range(self.m):
            Xi = X[:, i * ds : (i + 1) * ds].contiguous()
            km = KMeans(self.b, iters=self.iters, seed=self.seed + i)
            Ci = km.fit(Xi).to(X.dtype).contiguous()
            self.codebooks.append(Ci)

    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.shape
        ds = d // self.m
        codes = torch.empty((n, self.m), dtype=torch.uint8, device=X.device)
        for i in range(self.m):
            Xi = X[:, i * ds : (i + 1) * ds].contiguous()
            Ci = self.codebooks[i]
            x2 = (Xi * Xi).sum(-1, keepdim=True)
            c2 = (Ci * Ci).sum(-1).unsqueeze(0)
            dist2 = x2 + c2 - 2.0 * (Xi @ Ci.T)
            codes[:, i] = dist2.argmin(-1).to(torch.uint8)
        return codes.contiguous()

    @torch.no_grad()
    def build_lut(self, Q: torch.Tensor) -> torch.Tensor:
        # LUT of distances per subspace per codeword; compute with dtype alignment between Q-slice and codebook.
        B, d = Q.shape
        ds = d // self.m
        LUT = torch.empty((B, self.m, self.b), dtype=Q.dtype, device=Q.device)
        for i in range(self.m):
            Ci = self.codebooks[i]
            Qi = Q[:, i * ds : (i + 1) * ds].contiguous()
            Qi, Ci_ = _align_float_mm_dtypes(Qi, Ci)
            q2 = (Qi * Qi).sum(-1, keepdim=True)
            c2 = (Ci_ * Ci_).sum(-1).unsqueeze(0)
            LUT[:, i, :] = q2 + c2 - 2.0 * (Qi @ Ci_.T)
        return LUT.contiguous()


# -------------------------------
# IVF + OPQ/PQ Index (Torch-native)
# -------------------------------

@dataclass
class IVFBuildParams:
    nlist: int = 4096
    pq_m: int = 32
    pq_b: int = 256
    pq_iters: int = 20
    coarse_iters: int = 25
    opq_iters: int = 4
    seed: int = 0
    train_samples: Optional[int] = None


@dataclass
class IVFSearchParams:
    nprobe: int = 16
    refine: int = 200
    topk: int = 10
    metric: Literal["l2", "cosine", "ip"] = "cosine"
    use_amp: bool = True
    per_query_probe: bool = False
    flat_fallback: bool = True  # flat fallback when candidate set is empty/insufficient


class IVF_OPQ_PQ_Index:
    # Torch-native IVF + OPQ/PQ with non-empty masking, adaptive probe, and flat fallback.

    def __init__(self, d: int, device: torch.device, dtype: torch.dtype) -> None:
        self.d = int(d)
        self.device = device
        self.dtype = dtype
        self.opq = OPQ(d)
        self.coarse: Optional[KMeans] = None
        self.pq: Optional[PQ] = None
        self.centroids: Optional[torch.Tensor] = None
        self.cell_offsets: Optional[torch.Tensor] = None
        self.codes: Optional[torch.Tensor] = None
        self.ids: Optional[torch.Tensor] = None
        self.nlist: int = 0
        self.m: int = 0
        self.non_empty_mask: Optional[torch.Tensor] = None  # [nlist] bool

    @torch.no_grad()
    def _assign_cells(self, XR: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Compute closest centroid ids; assumes dtype alignment.
        XR, C = _align_float_mm_dtypes(XR, C)
        x2 = (XR * XR).sum(-1, keepdim=True)
        c2 = (C * C).sum(-1).unsqueeze(0)
        dist2 = x2 + c2 - 2.0 * (XR @ C.T)
        return dist2.argmin(-1)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, params: IVFBuildParams) -> None:
        N, d = X.shape
        assert d == self.d, f"d mismatch: expected {self.d}, got {d}"

        # Clamp nlist to [1, N] for stability; k-means with k > N leads to many empties.
        nlist = max(1, min(params.nlist, N))
        self.nlist = nlist

        # Optional training subset for scalability
        Xtrain = X
        if params.train_samples is not None and params.train_samples < N:
            idx = torch.randperm(N, device=X.device)[: params.train_samples]
            Xtrain = X.index_select(0, idx).contiguous()

        # OPQ rotation (fit on subset if provided)
        self.opq.fit(Xtrain, iters=params.opq_iters)
        XR = self.opq.apply(X)

        # Coarse k-means
        self.coarse = KMeans(nlist, iters=params.coarse_iters, seed=params.seed)
        C = self.coarse.fit(XR).to(self.dtype).contiguous()
        self.centroids = C

        # Assign and residual encode
        cell = self._assign_cells(XR, C)                   # [N]
        Rres = XR - C.index_select(0, cell)

        self.pq = PQ(d, m=params.pq_m, b=params.pq_b, iters=params.pq_iters, seed=params.seed)
        self.pq.fit(Rres)
        codes_all = self.pq.encode(Rres)
        self.m = params.pq_m

        # Vectorized reorder by cell: use argsort on cell labels
        counts = torch.bincount(cell, minlength=nlist)
        offsets = torch.zeros(nlist + 1, dtype=torch.long, device=X.device)
        offsets[1:] = torch.cumsum(counts, dim=0)
        order = torch.argsort(cell)  # [N], groups by cell id
        self.codes = codes_all.index_select(0, order).contiguous()
        self.ids = order.to(torch.long).contiguous()
        self.cell_offsets = offsets.contiguous()
        self.non_empty_mask = (counts > 0).to(torch.bool).contiguous()

    @torch.no_grad()
    def _coarse_probe(self, QR: torch.Tensor, nprobe: int) -> torch.Tensor:
        # Returns [B, k] of coarse cell ids, masking empty cells to avoid empty scans. Align dtypes for GPU.
        assert self.centroids is not None
        C = self.centroids
        B = QR.size(0)
        QRc, Cc = _align_float_mm_dtypes(QR.contiguous(), C.contiguous())
        q2 = (QRc * QRc).sum(-1, keepdim=True)               # [B,1]
        c2 = (Cc * Cc).sum(-1).unsqueeze(0)                  # [1,nlist]
        dist2 = q2 + c2 - 2.0 * (QRc @ Cc.T)                 # [B,nlist]
        if self.non_empty_mask is not None:
            if (~self.non_empty_mask).any():
                dist2 = dist2.masked_fill((~self.non_empty_mask).unsqueeze(0), float("inf"))
            k_eff = int(min(nprobe, int(self.non_empty_mask.sum().item())))
        else:
            k_eff = int(min(nprobe, C.size(0)))
        if k_eff <= 0:
            return torch.empty((B, 0), device=QR.device, dtype=torch.long)
        _, probe = torch.topk(dist2, k=k_eff, dim=-1, largest=False)
        return probe.contiguous()

    @torch.no_grad()
    def _flat_exact(
        self, Q: torch.Tensor, X: torch.Tensor, k: int, metric: str, mask_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Exact top-k for fallback: returns (scores [k], ids [k]) for a single query Q[1,d].
        D = metric_distances(X, Q, metric)  # [1, N]
        d = D.squeeze(0)
        if mask_ids is not None and mask_ids.numel() > 0:
            d.index_fill_(0, mask_ids, float("inf"))
        k_eff = min(k, d.numel())
        vals, idx = torch.topk(d, k_eff, largest=False)
        return vals, idx

    @torch.no_grad()
    def search(
        self,
        Q: torch.Tensor,
        raw_vectors: torch.Tensor,
        sp: IVFSearchParams,
        filter_bitset: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Returns per-query (scores, ids); robust to empty candidate sets via flat fallback.
        assert self.centroids is not None and self.pq is not None
        assert self.codes is not None and self.ids is not None and self.cell_offsets is not None
        assert Q.shape[1] == self.d

        device = self.device
        dtype_compute = torch.float16 if (sp.use_amp and device.type == "cuda") else self.dtype

        # Rotate queries and compute coarse probe (masked) with dtype alignment inside apply/probe.
        QR = self.opq.apply(Q.to(dtype_compute))
        probe = self._coarse_probe(QR, sp.nprobe)

        # Build LUTs for PQ
        LUT = self.pq.build_lut(QR)  # [B, m, b]

        final_scores: List[torch.Tensor] = []
        final_ids: List[torch.Tensor] = []

        # Per-query scanning
        for b in range(Q.shape[0]):
            ids_q: List[torch.Tensor] = []
            scores_q: List[torch.Tensor] = []

            # If no cells to probe (all empty), skip to flat fallback
            if probe.size(1) == 0:
                ids_q = []
            else:
                for j in range(probe.size(1)):
                    c = int(probe[b, j].item())
                    s = int(self.cell_offsets[c].item())
                    e = int(self.cell_offsets[c + 1].item())
                    if e <= s:
                        continue
                    codes_c = self.codes[s:e]    # [Nc, m]
                    ids_c = self.ids[s:e]        # [Nc]
                    # Apply filter bitset if provided
                    if filter_bitset is not None:
                        mask = filter_bitset.index_select(0, ids_c)
                        if mask.any():
                            codes_c = codes_c[mask]
                            ids_c = ids_c[mask]
                        else:
                            continue
                    if codes_c.shape[0] == 0:
                        continue
                    # Vectorized ADC via LUT:
                    # LUT[b]: [m, b]; codes_c: [Nc, m] -> indices [m, Nc] then gather and sum across m
                    L = LUT[b]                                 # [m, Bcode]
                    idx_mnc = codes_c.t().long()               # [m, Nc]
                    dist_apx = L.gather(1, idx_mnc).sum(0)     # [Nc]
                    # Keep refine shortlist
                    keep = min(sp.refine, dist_apx.numel())
                    if keep > 0:
                        v, idx = torch.topk(-dist_apx, k=keep)
                        scores_q.append((-v).to(torch.float32))
                        ids_q.append(ids_c.index_select(0, idx))

            # If no approximate candidates found, flat fallback
            if not ids_q:
                if sp.flat_fallback:
                    if filter_bitset is not None and filter_bitset.any():
                        kept = filter_bitset.nonzero(as_tuple=False).flatten()
                        Xc = raw_vectors.index_select(0, kept)
                        vals, idx = self._flat_exact(Q[b : b + 1].to(torch.float32), Xc.to(torch.float32), sp.topk, sp.metric)
                        final_scores.append(vals.to(torch.float32).contiguous())
                        final_ids.append(kept.index_select(0, idx).to(torch.long).contiguous())
                    else:
                        vals, idx = self._flat_exact(Q[b : b + 1].to(torch.float32), raw_vectors.to(torch.float32), sp.topk, sp.metric)
                        final_scores.append(vals.to(torch.float32).contiguous())
                        final_ids.append(idx.to(torch.long).contiguous())
                else:
                    final_scores.append(torch.empty((0,), device=device, dtype=torch.float32))
                    final_ids.append(torch.empty((0,), device=device, dtype=torch.long))
                continue

            # Dedup and exact re-rank on top refine candidates
            ids_cat = torch.cat(ids_q)                # [C]
            s_cat = torch.cat(scores_q)               # [C]
            best_apx, uniq_ids = dedup_min_by_id(s_cat, ids_cat)
            refine_k = min(sp.refine, best_apx.numel())
            _, refine_idx = torch.topk(-best_apx, k=refine_k)
            cand_ids = uniq_ids.index_select(0, refine_idx)
            Xc = raw_vectors.index_select(0, cand_ids).to(Q.device)
            q = Q[b : b + 1].to(torch.float32)
            Xc = Xc.to(dtype=q.dtype)

            if sp.metric == "cosine":
                Xc = normalize_rows(Xc)
                qn = normalize_rows(q)
                exact = cosine_distances(Xc, qn).flatten()
            elif sp.metric == "l2":
                exact = l2_distances(Xc, q).flatten()
            elif sp.metric == "ip":
                Xc, q = _align_float_mm_dtypes(Xc, q)
                exact = -(q @ Xc.T).flatten()
            else:
                raise ValueError(f"Unsupported metric: {sp.metric}")

            # Final top-k from exact candidates
            topk_eff = min(sp.topk, exact.numel())
            vals, idx = torch.topk(exact, k=topk_eff, largest=False)
            out_ids = cand_ids.index_select(0, idx).to(torch.long)
            out_vals = vals.to(torch.float32)

            # If insufficient candidates, optionally fill via flat fallback excluding already selected ids
            if sp.flat_fallback and topk_eff < sp.topk:
                need = sp.topk - topk_eff
                mask_ids = out_ids
                if filter_bitset is not None and filter_bitset.any():
                    kept = filter_bitset.nonzero(as_tuple=False).flatten()
                    Xall = raw_vectors.index_select(0, kept).to(torch.float32)
                    id_map = {int(kept[i].item()): i for i in range(kept.numel())}
                    mask_local = torch.tensor([id_map[int(t.item())] for t in mask_ids], device=device, dtype=torch.long)
                    add_vals, add_idx = self._flat_exact(q, Xall, need, sp.metric, mask_ids=mask_local)
                    add_ids = kept.index_select(0, add_idx).to(torch.long)
                else:
                    add_vals, add_idx = self._flat_exact(q, raw_vectors.to(torch.float32), need, sp.metric, mask_ids=mask_ids)
                    add_ids = add_idx.to(torch.long)
                out_vals = torch.cat([out_vals, add_vals.to(torch.float32)], dim=0)
                out_ids = torch.cat([out_ids, add_ids], dim=0)

            final_scores.append(out_vals.contiguous())
            final_ids.append(out_ids.contiguous())

        return final_scores, final_ids


# -------------------------------
# Flat Index (Exact Baseline)
# -------------------------------

class FlatIndex:
    # Exact baseline index.

    def __init__(self, d: int, device: torch.device) -> None:
        self.d = d
        self.device = device

    @torch.no_grad()
    def search(self, Q: torch.Tensor, X: torch.Tensor, topk: int, metric: Literal["l2", "cosine", "ip"]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        B = Q.shape[0]
        dist = metric_distances(X, Q, metric)
        vals, idx = batched_topk_smallest(dist, min(topk, X.shape[0]))
        scores = [vals[b].to(torch.float32).contiguous() for b in range(B)]
        ids = [idx[b].to(torch.long).contiguous() for b in range(B)]
        return scores, ids


# -------------------------------
# Metadata Filters (Bitset)
# -------------------------------

class FilterIndex:
    # Dense bitset per field/value for clarity; production would use compressed bitsets.

    def __init__(self, size_hint: int = 0, device: Optional[torch.device] = None) -> None:
        self.device = _pick_device() if device is None else device
        self.fields: Dict[str, Dict[Any, torch.Tensor]] = {}
        self.N = int(size_hint)

    @torch.no_grad()
    def ensure_size(self, N: int) -> None:
        if N <= self.N:
            return
        for fld in self.fields:
            for val in self.fields[fld]:
                old = self.fields[fld][val]
                self.fields[fld][val] = torch.nn.functional.pad(old, (0, N - self.N), value=0)
        self.N = N

    @torch.no_grad()
    def update(self, ids: torch.Tensor, metas: List[Dict[str, Any]]) -> None:
        if len(metas) == 0:
            return
        max_id = int(ids.max().item()) if ids.numel() else -1
        self.ensure_size(max(max_id + 1, self.N))
        for j, did in enumerate(ids.tolist()):
            meta = metas[j]
            for fld, val in meta.items():
                fmap = self.fields.setdefault(fld, {})
                mask = fmap.get(val)
                if mask is None:
                    mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
                    fmap[val] = mask
                mask[did] = True

    @torch.no_grad()
    def materialize(self, N: int, filters: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        if not filters:
            return None
        self.ensure_size(N)
        mask = torch.ones(N, dtype=torch.bool, device=self.device)
        for fld, val in filters.items():
            vmap = self.fields.get(fld, {})
            vmask = vmap.get(val)
            if vmask is None:
                return torch.zeros(N, dtype=torch.bool, device=self.device)
            mask &= vmask[:N]
        return mask


# -------------------------------
# Document Store and Collection
# -------------------------------

@dataclass
class DocChunk:
    doc_id: str
    index: int
    start: int
    end: int
    text: str
    hash64: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class Collection:
    # Holds raw vectors, metadata, and an index.

    def __init__(self, name: str, dim: int, metric: Literal["l2", "cosine", "ip"], device: torch.device) -> None:
        self.name = name
        self.dim = int(dim)
        self.metric = metric
        self.device = device

        self.texts: List[str] = []
        self.records: List[DocChunk] = []
        self.metadata: List[Dict[str, Any]] = []
        self._hash_to_id: Dict[int, int] = {}

        self.vectors: torch.Tensor = torch.empty((0, dim), dtype=_to_dtype_optim(dim), device=device)
        self.index_kind: Literal["IVF_OPQ_PQ", "FLAT"] = "FLAT"
        self.index_ivf: Optional[IVF_OPQ_PQ_Index] = None
        self.index_flat: FlatIndex = FlatIndex(dim, device=device)
        self.filters = FilterIndex(size_hint=0, device=device)

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @torch.no_grad()
    def append(self, embeddings: torch.Tensor, chunks: List[DocChunk]) -> List[int]:
        # Two-phase dedup (batch + global) w/o mutating global state until commit.
        assert embeddings.dim() == 2 and embeddings.shape[1] == self.dim
        assert len(chunks) == embeddings.shape[0]

        if embeddings.shape[0] == 0:
            return []

        start_id = self.size
        seen_global = set(self._hash_to_id.keys())
        seen_batch: set[int] = set()
        keep_idx: List[int] = []
        for i, ch in enumerate(chunks):
            h = ch.hash64
            if h is None:
                keep_idx.append(i)
                continue
            if (h not in seen_global) and (h not in seen_batch):
                keep_idx.append(i)
                seen_batch.add(h)

        if len(keep_idx) == 0:
            return []

        idx_t = torch.tensor(keep_idx, device=embeddings.device, dtype=torch.long)
        kept_embeddings = embeddings.index_select(0, idx_t).contiguous()
        kept_chunks = [chunks[i] for i in keep_idx]
        K = kept_embeddings.shape[0]

        self.vectors = torch.cat([self.vectors, kept_embeddings.to(self.vectors.dtype)], dim=0).contiguous()
        for ch in kept_chunks:
            self.texts.append(ch.text)
            self.records.append(ch)
            self.metadata.append(ch.meta)

        ids = list(range(start_id, start_id + K))

        for off, ch in enumerate(kept_chunks):
            if ch.hash64 is not None:
                self._hash_to_id[ch.hash64] = start_id + off

        self.filters.update(torch.tensor(ids, device=self.device), self.metadata[-K:])
        return ids

    @torch.no_grad()
    def build_index(self, kind: Literal["IVF_OPQ_PQ", "FLAT"], params: Optional[IVFBuildParams] = None) -> None:
        self.index_kind = kind
        if kind == "FLAT":
            self.index_ivf = None
            return
        if self.size == 0:
            raise RuntimeError("Cannot build IVF_OPQ_PQ index on an empty collection")
        params = params or IVFBuildParams()
        self.index_ivf = IVF_OPQ_PQ_Index(self.dim, self.device, _to_dtype_optim(self.dim))
        self.index_ivf.fit(self.vectors.to(self.device), params)

    @torch.no_grad()
    def search(
        self,
        Q: torch.Tensor,
        topk: int,
        filters: Optional[Dict[str, Any]] = None,
        params: Optional[IVFSearchParams] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.size == 0:
            return [torch.empty((0,), device=self.device)] * Q.shape[0], [torch.empty((0,), dtype=torch.long, device=self.device)] * Q.shape[0]
        params = params or IVFSearchParams(topk=topk, metric=self.metric)
        bitset = self.filters.materialize(self.size, filters)

        if self.index_kind == "FLAT" or self.index_ivf is None:
            X = self.vectors
            if bitset is not None:
                kept = bitset.nonzero(as_tuple=False).flatten()
                if kept.numel() == 0:
                    return [torch.empty((0,), device=self.device)] * Q.shape[0], [torch.empty((0,), dtype=torch.long, device=self.device)] * Q.shape[0]
                X = X.index_select(0, kept)
                sc, idx = self.index_flat.search(Q.to(self.device), X, topk, metric=params.metric)
                out_ids = [kept.index_select(0, i).contiguous() for i in idx]
                return sc, out_ids
            else:
                return self.index_flat.search(Q.to(self.device), X, topk, metric=params.metric)

        sc, ids = self.index_ivf.search(Q.to(self.device), self.vectors, params, filter_bitset=bitset)
        return sc, ids


# -------------------------------
# VectorBase API
# -------------------------------

class VectorBase:
    # High-level API: create collection, insert, build index, search.

    def __init__(
        self,
        embedder: Any,
        *,
        dim: Optional[int] = None,
        metric: Literal["l2", "cosine", "ip"] = "cosine",
        device: Optional[Union[str, torch.device]] = None,
        normalize_embeddings: Optional[bool] = None,
        batch_size: int = 64,
    ) -> None:
        dev = _pick_device(str(device)) if isinstance(device, str) else (device or _pick_device())
        self.device = dev
        self.metric = metric
        self._dim = dim

        if isinstance(embedder, EmbeddingAdapter):
            self.embedder = embedder
            self.embedder.device = dev
            self.embedder.dtype = torch.float32
            if normalize_embeddings is not None:
                self.embedder.normalize = normalize_embeddings
        else:
            norm = normalize_embeddings if normalize_embeddings is not None else (metric == "cosine")
            self.embedder = EmbeddingAdapter(embedder, device=dev, dtype=torch.float32, normalize=norm, batch_size=batch_size)

        self.collection: Optional[Collection] = None

    @torch.no_grad()
    def create_collection(self, name: str, dim: Optional[int] = None, metric: Optional[str] = None) -> None:
        d = int(dim if dim is not None else (self._dim or 0))
        if d <= 0:
            raise ValueError("Dimension must be provided (argument 'dim' or during VectorBase init)")
        m = metric or self.metric
        if m not in ("l2", "cosine", "ip"):
            raise ValueError(f"Unsupported metric: {m}")
        self.collection = Collection(name, d, m, device=self.device)

    @staticmethod
    def _text_from_record(rec: Dict[str, Any]) -> str:
        # Robust extraction across common JSONL schemas.
        if "text" in rec and isinstance(rec["text"], str):
            return rec["text"]
        for key in ("page_content", "content", "body", "chunk"):
            val = rec.get(key, "")
            if isinstance(val, str) and len(val.strip()) > 0:
                return val
        return ""

    @staticmethod
    def _safe_hash64_from_text(text: str) -> int:
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return struct.unpack("<Q", h)[0]

    @torch.no_grad()
    def insert(self, records: Sequence[Dict[str, Any]], *, return_stats: bool = False) -> Union[List[int], Tuple[List[int], Dict[str, int]]]:
        if self.collection is None:
            raise RuntimeError("No collection; call create_collection first")
        if not records:
            return ([], {"total": 0, "empty_text": 0, "embedded": 0, "inserted": 0}) if return_stats else []

        chunks: List[DocChunk] = []
        texts: List[str] = []
        counters = {"total": len(records), "empty_text": 0, "embedded": 0, "inserted": 0}

        for rec in records:
            text = self._text_from_record(rec).replace("\n", " ").strip()
            if len(text) == 0:
                counters["empty_text"] += 1
                continue
            doc_id = str(rec.get("doc_id", rec.get("id", "")))
            idx = int(rec.get("index", 0))
            start = int(rec.get("start", 0))
            end = int(rec.get("end", 0))
            meta_raw = {k: v for k, v in rec.items() if k not in ("doc_id", "id", "index", "start", "end", "text", "page_content", "content", "body", "chunk", "hash64", "metadata")}
            meta = rec.get("metadata", {})
            if isinstance(meta, dict):
                meta_raw.update(meta)
            h = rec.get("hash64", None)
            if not isinstance(h, int):
                h = self._safe_hash64_from_text(text)
            chunks.append(DocChunk(doc_id=doc_id, index=idx, start=start, end=end, text=text, hash64=h, meta=meta_raw))
            texts.append(text)

        if len(texts) == 0:
            return ([], counters) if return_stats else []

        vecs = self.embedder.embed_texts(texts)
        counters["embedded"] = int(vecs.shape[0])
        if vecs.numel() == 0 or vecs.ndim != 2 or vecs.shape[0] == 0:
            return ([], counters) if return_stats else []

        if self._dim is None:
            self._dim = int(vecs.shape[1])
            if self.collection.dim != self._dim:
                self.collection = Collection(self.collection.name, self._dim, self.collection.metric, self.device)
        elif vecs.shape[1] != self.collection.dim:
            raise ValueError(f"Embedding dimension mismatch: collection {self.collection.dim}, got {vecs.shape[1]}")

        ids = self.collection.append(vecs.to(self.device), chunks)
        counters["inserted"] = len(ids)
        return (ids, counters) if return_stats else ids

    @torch.no_grad()
    def build_index(self, kind: Literal["IVF_OPQ_PQ", "FLAT"] = "IVF_OPQ_PQ", params: Optional[IVFBuildParams] = None) -> None:
        if self.collection is None:
            raise RuntimeError("No collection; call create_collection first")
        if self.collection.size == 0 and kind != "FLAT":
            raise RuntimeError("Cannot build IVF_OPQ_PQ index on an empty collection")
        self.collection.build_index(kind, params=params)

    @torch.no_grad()
    def search(
        self,
        queries: Union[str, Sequence[str], torch.Tensor],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        params: Optional[IVFSearchParams] = None,
    ) -> Tuple[List[List[Tuple[int, float]]], List[Dict[str, Any]]]:
        if self.collection is None:
            raise RuntimeError("No collection; call create_collection first")
        coll = self.collection
        params = params or IVFSearchParams(topk=k, metric=coll.metric, flat_fallback=True)

        if isinstance(queries, str):
            Q = self.embedder.embed_query(queries)
        elif isinstance(queries, (list, tuple)) and len(queries) > 0 and isinstance(queries[0], str):
            Q = self.embedder.embed_texts(queries)  # type: ignore[arg-type]
        elif torch.is_tensor(queries):
            Q = queries.to(self.device)
        else:
            raise ValueError("queries must be str, List[str], or Tensor")

        local_scores, local_ids = coll.search(Q, topk=k, filters=filters, params=params)

        results: List[List[Tuple[int, float]]] = []
        contexts: List[Dict[str, Any]] = []
        for b in range(len(local_scores)):
            s = local_scores[b].tolist()
            i = local_ids[b].tolist()
            paired = sorted(zip(i, s), key=lambda t: (t[1], t[0]))[:k]
            results.append(paired)
            ctx_items = []
            for (vid, distv) in paired:
                rec = coll.records[vid]
                ctx_items.append({
                    "id": int(vid),
                    "doc_id": rec.doc_id,
                    "index": int(rec.index),
                    "start": int(rec.start),
                    "end": int(rec.end),
                    "distance": float(distv),
                    "text": rec.text,
                })
            contexts.append({"topk": ctx_items})
        return results, contexts


# -------------------------------
# Example Usage (guarded)
# -------------------------------

if __name__ == "__main__":
    # Demo:
    # - Reads JSONL from VB_INPUT_JSONL or 'out.jsonl'
    # - Embeds, inserts, builds IVF index, and queries with robust fallback
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        import json

        in_path = os.environ.get("VB_INPUT_JSONL", "out.jsonl")
        if not os.path.exists(in_path):
            print(f"[WARN] Input file '{in_path}' not found; skipping demo run.")
            sys.exit(0)

        records = [json.loads(line) for line in open(in_path, "r", encoding="utf-8")]
        if len(records) == 0:
            print("[WARN] Input file contains zero records; skipping index build.")
            sys.exit(0)

        # Use a 768-dim model to match collection dim and avoid dimension mismatch errors.
        hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        adapter = EmbeddingAdapter(hf, device=("cuda" if torch.cuda.is_available() else "cpu"), normalize=True, batch_size=64)
        vb = VectorBase(adapter, dim=768, metric="cosine", device=adapter.device)
        vb.create_collection("docs", dim=768, metric="cosine")

        inserted_ids, stats = vb.insert(records, return_stats=True)
        print(f"[INFO] Ingest stats: total={stats['total']} empty_text={stats['empty_text']} embedded={stats['embedded']} inserted={stats['inserted']}")

        if len(inserted_ids) == 0:
            print("[WARN] No vectors inserted. Skipping index build.")
            sys.exit(0)

        # Use nlist <= N to avoid empty-cell dominance; IVFBuildParams clamps internally as well.
        nlist = min(max(1, stats['inserted'] // 2), 4096)
        vb.build_index(kind="IVF_OPQ_PQ", params=IVFBuildParams(nlist=nlist, pq_m=16, train_samples=None))

        q = os.environ.get("VB_QUERY", " Key Usage Purposes (as per X.509 v3 Key Usage Field)")
        res, ctx = vb.search(q, k=5, filters=None, params=IVFSearchParams(nprobe=16, refine=200, topk=5, metric="cosine", flat_fallback=True))
        for r, c in zip(res, ctx):
            print("Results:", r)
            print("Contexts:", c)

    except ImportError as e:
        print(f"[WARN] Missing dependencies for demo run: {e}. Install 'langchain-huggingface' and 'huggingface_hub' or import this module as a library.")