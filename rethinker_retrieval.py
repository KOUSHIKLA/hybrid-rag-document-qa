# --------------------------------------------------------------------------------------------------
# Rethinker Retrieval — Structure-aware, graph-augmented retriever over an existing vector base.
# Purpose:
#   - Improve beyond naive "exact match + +/- neighbors" by constructing and traversing a document graph.
#   - Combine lexical (BM25-like) and semantic (vector) signals with structural adjacency.
#   - Use DFS with beam pruning to gather highly relevant, coherent contexts.
#
# Key Ideas:
#   - Seeds = union of top semantic hits and top lexical hits.
#   - Context Graph = implicit graph with:
#       (1) Local adjacency (prev/next chunk within same doc_id)
#       (2) On-demand semantic edges (k-NN per node via the collection index)
#   - DFS+Beam Search = explore graph from seeds with learned weights and decay, deduplicate, then assemble contexts.
#   - Draw-Back = assemble final contexts by smartly grouping contiguous chunks; avoid naive "fixed +/- window".
#
# Integration:
#   - This module expects a "Collection" and "VectorBase" that look like those in 'torchvectorbase.py'.
#   - Works with any metric supported by the collection ("cosine", "l2", "ip").
#   - No prints; all explanations are embedded as comments.
#
# Notes:
#   - Efficiency:
#       * Semantic neighbors are computed on-demand and cached.
#       * Query-to-node and lexical scores are cached per query.
#       * DFS uses beam pruning and global visited cut to prevent combinatorial explosion.
#   - Robustness:
#       * Handles empty collections and missing fields safely.
#       * Normalizes scores so lexical and semantic signals co-exist gracefully.
#   - Extensibility:
#       * You can add richer edges (e.g., section headers, hyperlinks) by extending ContextGraph.
# --------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import defaultdict

import torch


# Optional imports from torchvectorbase (kept soft for modularity)
try:
    from torchvectorbase import Collection, VectorBase, IVFSearchParams, EmbeddingAdapter, IVFBuildParams # type: ignore
except Exception:
    Collection = Any  # type: ignore
    VectorBase = Any  # type: ignore

    @dataclass
    class IVFSearchParams:  # type: ignore
        nprobe: int = 16
        refine: int = 200
        topk: int = 10
        metric: str = "cosine"
        use_amp: bool = True
        per_query_probe: bool = False
        flat_fallback: bool = True


# -----------------------------------
# Tokenization and Normalization Utils
# -----------------------------------

def _tokenize(text: str) -> List[str]:
    # Simple alnum tokenizer; lowercase; suitable for BM25-style scoring.
    return re.findall(r"[A-Za-z0-9_]+", str(text).lower())


def _safe_minmax_scale(values: List[float], eps: float = 1e-9) -> List[float]:
    # Scales values into [0,1]; returns zeros if degenerate.
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax - vmin < eps:
        return [0.0 for _ in values]
    inv = 1.0 / (vmax - vmin)
    return [(v - vmin) * inv for v in values]


def _distance_to_similarity(dist: float, metric: str) -> float:
    # Converts a distance (smaller better) into a similarity (larger better) in [0,1]-ish range.
    # For cosine: dist = 1 - cos => cos = 1 - dist in [-1,1]; map to [0,1] by (cos+1)/2.
    # For l2: use 1/(1+dist) as a bounded proxy.
    # For ip: search returns distances = -dot; sim ~ -dist; map to [0,1] by sigmoid-like squash.
    m = metric.lower()
    if m == "cosine":
        cos = 1.0 - dist
        return max(0.0, min(1.0, 0.5 * (cos + 1.0)))
    if m == "l2":
        return 1.0 / (1.0 + max(dist, 0.0))
    if m == "ip":
        sim = -dist  # dot
        return 1.0 / (1.0 + math.exp(-sim))  # logistic squash
    return 0.0


# -------------------
# Lexical BM25 Index
# -------------------

@dataclass
class BM25Params:
    k1: float = 0.9
    b: float = 0.4
    # Note: For production, use tuned values per corpus. These defaults are robust for short chunks.


class LexicalIndex:
    # Minimal BM25-like inverted index over chunk texts.

    def __init__(self, texts: Sequence[str], params: BM25Params = BM25Params()) -> None:
        self.params = params
        self.N = len(texts)
        self.doc_lens: List[int] = []
        self.avgdl: float = 1.0
        self.df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

        # Build inverted structures
        self._build(texts)

    def _build(self, texts: Sequence[str]) -> None:
        # Tokenize and compute tf/df; keep memory footprint modest by storing per-doc dicts.
        df_counter: DefaultDict[str, int] = defaultdict(int)
        self.tf = []
        self.doc_lens = []
        for txt in texts:
            toks = _tokenize(txt)
            self.doc_lens.append(len(toks))
            tf_local: DefaultDict[str, int] = defaultdict(int)
            for t in toks:
                tf_local[t] += 1
            self.tf.append(dict(tf_local))
            for t in tf_local.keys():
                df_counter[t] += 1
        self.df = dict(df_counter)
        self.avgdl = float(sum(self.doc_lens) / max(1, self.N))

        # Compute IDF with a standard BM25 log-smoothing
        self.idf = {}
        for t, df in self.df.items():
            # IDF = log( (N - df + 0.5) / (df + 0.5) + 1 )
            self.idf[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score_query(self, query: str, candidate_ids: Optional[Iterable[int]] = None) -> Dict[int, float]:
        # Scores candidates for a given query string. If candidate_ids None, score all docs.
        # Returns sparse dict: doc_id -> score.
        toks = _tokenize(query)
        if not toks or self.N == 0:
            return {}

        # Build set of candidate docs (union of postings of query tokens) if not provided.
        cand_set: Set[int]
        if candidate_ids is None:
            cand_set = set()
            for t in set(toks):
                # If term not in corpus, skip.
                # We avoid materializing postings to save memory; we scan per-doc tf maps below.
                pass
            # Fall back: score all docs if no df info (tiny corpora).
            if not cand_set:
                cand_set = set(range(self.N))
        else:
            cand_set = set(candidate_ids)

        # BM25 scoring
        k1 = self.params.k1
        b = self.params.b
        avgdl = self.avgdl

        scores: Dict[int, float] = {}
        q_terms = set(toks)  # treat duplicates lightly; can be extended to use qtf
        for doc_id in cand_set:
            tf_map = self.tf[doc_id]
            dl = self.doc_lens[doc_id] if self.doc_lens else 1
            s = 0.0
            for t in q_terms:
                tf = tf_map.get(t, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
                s += idf * (tf * (k1 + 1.0)) / max(1e-9, denom)
            if s > 0.0:
                scores[doc_id] = s
        return scores


# ----------------------------
# Context Graph (structural)
# ----------------------------

@dataclass
class NodeInfo:
    # Minimal node metadata needed for structure-aware retrieval.
    id: int
    doc_id: str
    chunk_index: int  # logical position within the document (e.g., "index" from DocChunk)
    start: int
    end: int


class ContextGraph:
    # Constructs local adjacency (prev/next in same document). Semantic edges are computed on-demand.

    def __init__(self, collection: Collection) -> None:
        self.collection = collection
        self.N = collection.size
        self.texts: List[str] = list(collection.texts)
        self.nodes: List[NodeInfo] = []
        self.adj_prev: Dict[int, int] = {}  # node_id -> prev node_id (same doc), if any
        self.adj_next: Dict[int, int] = {}  # node_id -> next node_id (same doc), if any
        self._build()

    def _build(self) -> None:
        # Group nodes by (doc_id) and relative order (chunk.index), then wire prev/next.
        groups: DefaultDict[str, List[Tuple[int, NodeInfo]]] = defaultdict(list)
        for nid, rec in enumerate(self.collection.records):
            info = NodeInfo(
                id=nid,
                doc_id=str(rec.doc_id),
                chunk_index=int(rec.index),
                start=int(rec.start),
                end=int(rec.end),
            )
            self.nodes.append(info)
            groups[info.doc_id].append((info.chunk_index, info))
        for _, arr in groups.items():
            arr.sort(key=lambda x: x[0])
            for j in range(len(arr)):
                if j > 0:
                    self.adj_prev[arr[j][1].id] = arr[j - 1][1].id
                if j + 1 < len(arr):
                    self.adj_next[arr[j][1].id] = arr[j + 1][1].id

    def neighbors_local(self, node_id: int) -> List[int]:
        # Returns prev/next neighbors in same doc if exist.
        nxt = self.adj_next.get(node_id, None)
        prv = self.adj_prev.get(node_id, None)
        out: List[int] = []
        if prv is not None:
            out.append(prv)
        if nxt is not None:
            out.append(nxt)
        return out


# --------------------------------------
# Rethinker Parameters (weights, limits)
# --------------------------------------

@dataclass
class RethinkerParams:
    # Seed collection
    seed_sem_topk: int = 64
    seed_lex_topk: int = 64

    # DFS traversal
    max_depth: int = 3
    beam_per_depth: int = 8
    semantic_k_per_node: int = 8
    max_expansions: int = 2000

    # Scoring weights
    w_sem_query: float = 0.60  # weight for query <-> node semantic similarity
    w_lex: float = 0.30        # weight for lexical BM25 score
    w_adjacent: float = 0.10   # bonus weight for local adjacency edges
    decay_per_depth: float = 0.85  # multiplicative decay per depth (>0, <1)
    exact_phrase_boost: float = 0.10  # bonus if exact phrase appears in node text

    # Final assembly
    top_nodes_final: int = 24
    draw_above: int = 2
    draw_below: int = 2
    max_chars_per_context: int = 2000  # trims overly long merged contexts


# ------------------------
# Core Rethinker Retriever
# ------------------------

class Rethinker:
    # DFS + beam search over a context graph with hybrid semantic + lexical scoring.

    def __init__(self, vb: VectorBase, params: Optional[RethinkerParams] = None) -> None:
        assert vb.collection is not None, "VectorBase.collection is required"
        self.vb = vb
        self.coll: Collection = vb.collection  # type: ignore
        self.params = params or RethinkerParams()
        self.graph = ContextGraph(self.coll)
        self.lex = LexicalIndex(self.graph.texts)

        # Per-query caches
        self._cache_sem_neighbors: Dict[int, Tuple[List[int], List[float]]] = {}
        self._cache_q_sim: Dict[int, float] = {}
        self._cache_lex_score: Dict[int, float] = {}

    def _reset_caches(self) -> None:
        self._cache_sem_neighbors.clear()
        self._cache_q_sim.clear()
        self._cache_lex_score.clear()

    def _embed_query(self, query: str) -> torch.Tensor:
        # Leverage VB embedder; returns [1,d] on same device as VB.
        return self.vb.embedder.embed_query(query).to(self.vb.device)

    def _semantic_neighbors_for_node(self, node_id: int, k: int, metric: str) -> Tuple[List[int], List[float]]:
        # On-demand retrieve semantic neighbors for a node using the collection index.
        # Cache results to avoid repeated searches while traversing.
        if node_id in self._cache_sem_neighbors:
            ids, sims = self._cache_sem_neighbors[node_id]
            return ids, sims

        vec = self.coll.vectors.index_select(0, torch.tensor([node_id], device=self.vb.device))
        # Use IVF if available, otherwise flat. We request a modest topk to keep traversal light.
        sp = IVFSearchParams(nprobe=16, refine=64, topk=k + 4, metric=metric, flat_fallback=True)
        sc, ids = self.coll.search(vec, topk=sp.topk, filters=None, params=sp)
        if len(ids) == 0 or ids[0].numel() == 0:
            self._cache_sem_neighbors[node_id] = ([], [])
            return [], []

        # Convert distances to similarities in [0,1]-ish range; drop self and dups.
        raw_ids = ids[0].tolist()
        raw_dist = sc[0].tolist()
        neigh: List[int] = []
        sims: List[float] = []
        seen: Set[int] = set([node_id])
        for did, dd in zip(raw_ids, raw_dist):
            if did in seen:
                continue
            seen.add(did)
            neigh.append(int(did))
            sims.append(_distance_to_similarity(float(dd), metric))
            if len(neigh) >= k:
                break

        self._cache_sem_neighbors[node_id] = (neigh, sims)
        return neigh, sims

    def _query_similarity(self, Q: torch.Tensor, node_id: int, metric: str) -> float:
        # Computes similarity between query vector [1,d] and node embedding [d], caches result.
        if node_id in self._cache_q_sim:
            return self._cache_q_sim[node_id]
        x = self.coll.vectors.index_select(0, torch.tensor([node_id], device=self.vb.device))
        if metric == "cosine":
            # Normalize on the fly for robustness; cast to float32 for numerical stability.
            qn = Q.to(dtype=torch.float32)
            xn = torch.nn.functional.normalize(x.to(dtype=torch.float32), dim=-1)
            qn = torch.nn.functional.normalize(qn, dim=-1)
            cos = float((qn @ xn.T).item())
            sim = 0.5 * (cos + 1.0)  # map [-1,1] -> [0,1]
        elif metric == "l2":
            d2 = float(((x - Q) ** 2).sum().item())
            sim = 1.0 / (1.0 + max(0.0, d2))
        elif metric == "ip":
            dot = float((Q.to(dtype=torch.float32) @ x.to(dtype=torch.float32).T).item())
            sim = 1.0 / (1.0 + math.exp(-dot))
        else:
            sim = 0.0
        self._cache_q_sim[node_id] = sim
        return sim

    def _lexical_score_node(self, query: str, node_id: int, precomputed: Optional[Dict[int, float]] = None) -> float:
        # Returns BM25-like lexical score for the node, reusing per-query precomputed map when possible.
        if node_id in self._cache_lex_score:
            return self._cache_lex_score[node_id]
        if precomputed is not None:
            s = float(precomputed.get(node_id, 0.0))
            self._cache_lex_score[node_id] = s
            return s
        scores = self.lex.score_query(query, candidate_ids=None)
        # Cache all to amortize subsequent node lookups
        for k, v in scores.items():
            self._cache_lex_score[int(k)] = float(v)
        return float(self._cache_lex_score.get(node_id, 0.0))

    def _has_exact_phrase(self, query: str, node_id: int) -> bool:
        # Detects if the raw query phrase appears verbatim in the node text (case-insensitive).
        q = str(query).strip()
        if not q:
            return False
        txt = self.graph.texts[node_id]
        return q.lower() in txt.lower()

    def _collect_seed_nodes(self, Q: torch.Tensor, query: str, metric: str) -> List[Tuple[int, float]]:
        # Collects hybrid seeds from semantic and lexical retrieval; returns list of (node_id, seed_score).
        p = self.params

        # Semantic seeds via collection search with Q
        sp = IVFSearchParams(nprobe=16, refine=max(128, p.seed_sem_topk), topk=p.seed_sem_topk, metric=metric, flat_fallback=True)
        sc_sem, ids_sem = self.coll.search(Q, topk=sp.topk, filters=None, params=sp)
        sem_pairs: List[Tuple[int, float]] = []
        if ids_sem and ids_sem[0].numel() > 0:
            dists = sc_sem[0].tolist()
            idl = ids_sem[0].tolist()
            sims = [_distance_to_similarity(float(d), metric) for d in dists]
            sem_pairs = [(int(i), float(s)) for i, s in zip(idl, sims)]

        # Lexical seeds via BM25
        lex_scores = self.lex.score_query(query, candidate_ids=None)
        # Normalize lexical scores into [0,1] range for blending
        if lex_scores:
            vals = list(lex_scores.values())
            vals_scaled = _safe_minmax_scale(vals)
            for (k, vscaled) in zip(lex_scores.keys(), vals_scaled):
                lex_scores[int(k)] = float(vscaled)
        lex_pairs: List[Tuple[int, float]] = [(int(k), float(v)) for k, v in lex_scores.items()]
        lex_pairs.sort(key=lambda x: -x[1])
        lex_pairs = lex_pairs[: self.params.seed_lex_topk]

        # Blend using weighted sum; also add exact phrase boost.
        merged: Dict[int, float] = defaultdict(float)
        for nid, s in sem_pairs:
            merged[nid] += self.params.w_sem_query * s
        for nid, s in lex_pairs:
            merged[nid] += self.params.w_lex * s
        for nid in list(merged.keys()):
            if self._has_exact_phrase(query, nid):
                merged[nid] += self.params.exact_phrase_boost

        seeds = list(merged.items())
        seeds.sort(key=lambda x: -x[1])
        return seeds

    def _dfs_traverse(self, Q: torch.Tensor, query: str, metric: str, seeds: List[Tuple[int, float]]) -> Dict[int, float]:
        # Depth-first traversal with beam pruning and decay. Aggregates per-node best score.
        p = self.params

        # Precompute lexical scores for all nodes referenced during traversal
        pre_lex = self.lex.score_query(query, candidate_ids=None)
        if pre_lex:
            vals = list(pre_lex.values())
            scaled = _safe_minmax_scale(vals)
            for (k, v) in zip(pre_lex.keys(), scaled):
                pre_lex[int(k)] = float(v)

        # Per-node max aggregated score
        best_score: Dict[int, float] = defaultdict(float)
        # Global visited (optional): we still allow revisits if improved score > current by margin
        visited: Dict[int, int] = defaultdict(int)  # node_id -> best depth reached

        # Stack frame: (node_id, depth, agg_score)
        # Seed ordering encourages high-quality deep dives early; we also respect beam width per depth.
        # We will maintain depth-wise beam by keeping a buffer of candidate expansions per depth.
        by_depth: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for nid, s in seeds[: max(1, p.beam_per_depth)]:
            by_depth[0].append((nid, s))

        expansions = 0
        for depth in range(0, p.max_depth + 1):
            # Beam prune at this depth
            frontier = sorted(by_depth.get(depth, []), key=lambda x: -x[1])[: p.beam_per_depth]
            if not frontier:
                break

            for nid, agg in frontier:
                expansions += 1
                if expansions >= p.max_expansions:
                    break

                # Update best score for current node
                if agg > best_score.get(nid, 0.0):
                    best_score[nid] = agg

                if depth == p.max_depth:
                    continue

                # Enumerate neighbors: local adjacency (+bonus) and semantic neighbors
                local_neighbors = self.graph.neighbors_local(nid)
                sem_neighbors, sem_sims = self._semantic_neighbors_for_node(nid, p.semantic_k_per_node, metric)

                # Combine neighbors into candidate list with per-edge base contributions
                # For local neighbors, we compute query sim and lexical score on-the-fly.
                neigh_candidates: List[Tuple[int, float]] = []

                # Local adjacency neighbors
                for nb in local_neighbors:
                    sim_q = self._query_similarity(Q, nb, metric)
                    lex_nb = self._lexical_score_node(query, nb, precomputed=pre_lex)
                    base = p.w_sem_query * sim_q + p.w_lex * lex_nb + p.w_adjacent  # adjacency bonus
                    neigh_candidates.append((nb, base))

                # Semantic neighbors (already have sim(node->nb) from PQ re-rank proxies; we still score w.r.t query)
                for nb, _sem_edge_sim in zip(sem_neighbors, sem_sims):
                    sim_q = self._query_similarity(Q, nb, metric)
                    lex_nb = self._lexical_score_node(query, nb, precomputed=pre_lex)
                    base = p.w_sem_query * sim_q + p.w_lex * lex_nb  # semantic edge has no fixed adjacency bonus
                    neigh_candidates.append((nb, base))

                # Remove duplicates and self, keep best base per neighbor
                dedup: Dict[int, float] = defaultdict(float)
                for nb, base in neigh_candidates:
                    if nb == nid:
                        continue
                    if base > dedup.get(nb, 0.0):
                        dedup[nb] = base

                # Beam prune neighbors for next depth; apply exponential decay
                decay = (p.decay_per_depth ** (depth + 1))
                next_candidates = sorted(dedup.items(), key=lambda x: -x[1])[: p.beam_per_depth]
                for nb, base in next_candidates:
                    # Optional improvement check: if we have visited deeper with better depth, allow if beneficial
                    if visited.get(nb, 1 << 30) <= depth + 1:
                        pass
                    visited[nb] = depth + 1
                    new_score = agg + decay * base
                    # Apply exact phrase boost lazily (once per expansion)
                    if self._has_exact_phrase(query, nb):
                        new_score += p.exact_phrase_boost * decay
                    by_depth[depth + 1].append((nb, new_score))

            if expansions >= p.max_expansions:
                break

        return dict(best_score)

    def _assemble_contexts(self, node_scores: Dict[int, float]) -> List[Dict[str, Any]]:
        # Turn top nodes into merged, contiguous contexts by drawing upward/downward within same doc.
        p = self.params
        if not node_scores:
            return []

        # Select top nodes
        items = sorted(node_scores.items(), key=lambda x: -x[1])[: p.top_nodes_final]
        selected_ids = [nid for nid, _ in items]

        # Expand each node by +/- window and group by doc_id
        windows_by_doc: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)
        id_to_info = {n.id: n for n in self.graph.nodes}

        for nid in selected_ids:
            info = id_to_info[nid]
            # Walk up p.draw_above
            start_nid = nid
            for _ in range(p.draw_above):
                prev_id = self.graph.adj_prev.get(start_nid, None)
                if prev_id is None:
                    break
                start_nid = prev_id
            # Walk down p.draw_below
            end_nid = nid
            for _ in range(p.draw_below):
                next_id = self.graph.adj_next.get(end_nid, None)
                if next_id is None:
                    break
                end_nid = next_id
            windows_by_doc[info.doc_id].append((start_nid, end_nid))

        # Merge overlapping/adjacent windows within each doc
        merged_contexts: List[Dict[str, Any]] = []
        for doc_id, spans in windows_by_doc.items():
            # Map node_id -> chunk_index for stable ordering
            spans_sorted = sorted(
                spans,
                key=lambda ab: (id_to_info[ab[0]].chunk_index, id_to_info[ab[1]].chunk_index),
            )
            merged: List[Tuple[int, int]] = []
            for a, b in spans_sorted:
                if not merged:
                    merged.append((a, b))
                    continue
                prev_a, prev_b = merged[-1]
                # If overlapping or touching, merge; else start new
                if id_to_info[a].chunk_index <= id_to_info[prev_b].chunk_index + 1:
                    # Extend right end to max
                    right = b
                    while True:
                        # Ensure ordering by chunk indices
                        if id_to_info[right].chunk_index >= id_to_info[prev_b].chunk_index:
                            break
                        nxt = self.graph.adj_next.get(right, None)
                        if nxt is None:
                            break
                        right = nxt
                    merged[-1] = (prev_a, max(prev_b, b, key=lambda z: id_to_info[z].chunk_index))
                else:
                    merged.append((a, b))

            # Emit contexts (bounded by max_chars)
            for a, b in merged:
                # Collect node ids from a..b along next pointers
                seq: List[int] = []
                cur = a
                while True:
                    seq.append(cur)
                    if cur == b:
                        break
                    nxt = self.graph.adj_next.get(cur, None)
                    if nxt is None:
                        break
                    cur = nxt

                # Concatenate texts with lightweight separator
                texts = [self.graph.texts[t] for t in seq]
                joined = " ".join(texts)
                if len(joined) > self.params.max_chars_per_context:
                    joined = joined[: self.params.max_chars_per_context] + "…"

                # For the context score, max over nodes in seq
                score = max([float(node_scores.get(t, 0.0)) for t in seq] or [0.0])
                # Representative span coordinates
                first_info, last_info = id_to_info[seq[0]], id_to_info[seq[-1]]
                merged_contexts.append({
                    "doc_id": doc_id,
                    "start_node_id": int(seq[0]),
                    "end_node_id": int(seq[-1]),
                    "start_index": int(first_info.chunk_index),
                    "end_index": int(last_info.chunk_index),
                    "start_char": int(first_info.start),
                    "end_char": int(last_info.end),
                    "score": float(score),
                    "text": joined,
                    "node_ids": [int(t) for t in seq],
                })

        # Sort contexts by score descending, stable
        merged_contexts.sort(key=lambda x: -x["score"])
        return merged_contexts

    def search(self, query: str) -> Dict[str, Any]:
        # Entry point: given a query string, return structured contexts and internal diagnostics.
        if self.coll.size == 0:
            return {"contexts": [], "debug": {"reason": "empty_collection"}}

        self._reset_caches()

        # Embed query
        Q = self._embed_query(query)

        # Seeds
        seeds = self._collect_seed_nodes(Q, query, metric=self.coll.metric)
        if not seeds:
            return {"contexts": [], "debug": {"reason": "no_seeds"}}

        # DFS with beam pruning
        node_scores = self._dfs_traverse(Q, query, metric=self.coll.metric, seeds=seeds)

        # Assemble contexts
        contexts = self._assemble_contexts(node_scores)

        # Light diagnostics for tunability
        debug = {
            "seed_count": len(seeds),
            "expanded_nodes": len(node_scores),
            "top_contexts": len(contexts),
            "params": self.params.__dict__.copy(),
        }
        return {"contexts": contexts, "debug": debug}


# ------------------------
# Example Integration Notes
# ------------------------
# The following shows how to integrate with VectorBase (no execution here; this is a usage sketch):
#
#   # Assume you already have:
#   #   - vb = VectorBase(embedder, dim=..., metric="cosine", device=...)
#   #   - vb.create_collection(...), vb.insert(...), vb.build_index(kind="IVF_OPQ_PQ", ...)
#   #
#   # Construct Rethinker over vb:
#   #   rk = Rethinker(vb, params=RethinkerParams(
#   #       seed_sem_topk=64,
#   #       seed_lex_topk=64,
#   #       max_depth=3,
#   #       beam_per_depth=8,
#   #       semantic_k_per_node=8,
#   #       max_expansions=2000,
#   #       w_sem_query=0.60,
#   #       w_lex=0.30,
#   #       w_adjacent=0.10,
#   #       decay_per_depth=0.85,
#   #       exact_phrase_boost=0.10,
#   #       top_nodes_final=24,
#   #       draw_above=2,
#   #       draw_below=2,
#   #       max_chars_per_context=2000,
#   #   ))
#   #
#   # Run retrieval:
#   #   out = rk.search("your query text")
#   #   contexts = out["contexts"]  # List[Dict[str,Any]], sorted by "score"
#   #   debug = out["debug"]
#
# Tuning Tips:
#   - Increase beam_per_depth and semantic_k_per_node to explore more; watch max_expansions.
#   - Adjust weights (w_sem_query, w_lex, w_adjacent) to favor semantic vs lexical signals.
#   - For high-recall scenarios, increase seed_sem_topk/seed_lex_topk and max_depth.
#   - For tighter, more coherent contexts, increase draw_above/draw_below modestly and enforce max_chars_per_context.
#
# Extending Edges:
#   - Add section-title edges: parse metadata for headings and cross-link chunks within same section.
#   - Add hyperlink edges: if records contain links, connect sources to targets with appropriate weights.
#   - Add time/ID proximity edges: group by timestamp or semantic sessions for conversational datasets.
#
# Safety:
#   - This module avoids prints/logging by design. All explanations live in comments.
#   - It relies on torch tensors where needed but most logic executes on CPU for flexibility.
# --------------------------------------------------------------------------------------------------
import os
import sys
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

        q = os.environ.get("VB_QUERY", "Algorithm Object Identifiers ")
        res, ctx = vb.search(q, k=5, filters=None, params=IVFSearchParams(nprobe=16, refine=200, topk=5, metric="cosine", flat_fallback=True))
        rk = Rethinker(vb, params=RethinkerParams(
        seed_sem_topk=64,
        seed_lex_topk=64,
        max_depth=3,
        beam_per_depth=8,
        semantic_k_per_node=8,
        max_expansions=2000,
        w_sem_query=0.60,
        w_lex=0.30,
        w_adjacent=0.10,
        decay_per_depth=0.85,
        exact_phrase_boost=0.10,
        top_nodes_final=24,
        draw_above=2,
        draw_below=2,
        max_chars_per_context=2000,
      ))
        out = rk.search(q)
        contexts = out["contexts"]  # List[Dict[str,Any]], sorted by "score"
        debug = out["debug"]
       


        import asyncio
        from openai import AsyncOpenAI
        
        # Initialize the asynchronous client
        client = AsyncOpenAI(base_url="http://10.180.93.12:8007/v1", api_key="EMPTY")
        
        async def get_chat_completion(system_prompt: str, user_prompt: str) -> str:
            """
            Get chat completion from GPT-OSS-20B asynchronously.
        
            Args:
                system_prompt (str): The system instruction for the assistant.
                user_prompt (str): The user's message or question.
        
            Returns:
                str: The assistant's response content.
            """
            response = await client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
        
        async def main():
            system_message = "you advanced rag given the context answer the question concisely or tell generalised answer mention that is general answer if context is not sufficient"
            user_message = f""""
             Question: {q}
             Contexts:{contexts}

               """
            
            # Get response
            assistant_response = await get_chat_completion(system_message, user_message)
            
            # Output
            # print("User:", user_message)
            print("Assistant:", assistant_response)
        
        # Run the asynchronous main function
        asyncio.run(main())



    except ImportError as e:
        print(f"[WARN] Missing dependencies for demo run: {e}. Install 'langchain-huggingface' and 'huggingface_hub' or import this module as a library.")
        
        
        
        
        
        
        
        
        
        
# ===============================================================
# Global Initialization (runs once when module is imported)
# ===============================================================

import os, json, torch, yaml
from langchain_huggingface import HuggingFaceEmbeddings
from torchvectorbase import EmbeddingAdapter, VectorBase
from openai import AsyncOpenAI
from rethinker_retrieval import Rethinker, RethinkerParams

# Load config once
with open("ragconfig.yaml", "r") as f:
    CFG = yaml.safe_load(f)

LLM_BASE_URL = CFG["LLM_BASE_URL"]
LLM_API_KEY = CFG["LLM_API_KEY"]
LLM_MODEL = CFG["LLM_MODEL"]

CHUNKS_PATH = CFG["DEFAULT_CHUNKS_JSONL"]

# ---- Build Embeddings + VectorBase only once ----
embedder = HuggingFaceEmbeddings(
    model_name=CFG["DEFAULT_EMBED_MODEL"]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
adapter = EmbeddingAdapter(embedder, device=device, normalize=True)

vb = VectorBase(adapter, dim=CFG["EMBED_DIM"], metric="cosine", device=device)
vb.create_collection("docs", dim=CFG["EMBED_DIM"], metric="cosine")

if os.path.exists(CHUNKS_PATH):
    records = [json.loads(line) for line in open(CHUNKS_PATH)]
    vb.insert(records)
    vb.build_index(kind="IVF_FLAT")

# ---- Build Rethinker once ----
rk = Rethinker(vb, params=RethinkerParams())

# ---- Build LLM client once ----
client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


# ===============================================================
# PUBLIC FUNCTION USED BY UI AND APP.PY
# ===============================================================

def ask_question(query: str) -> str:
    """Fast RAG call for UI & app.py."""
    import asyncio

    # 1. Retrieval
    result = rk.search(query)
    contexts = result.get("contexts", [])

    if not contexts:
        context_block = "No relevant context found."
    else:
        context_block = "\n\n".join([c["text"] for c in contexts[:5]])

    # 2. LLM call
    async def run():
        completion = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an advanced RAG system."},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_block}"}
            ]
        )
        return completion.choices[0].message.content

    return asyncio.run(run())
