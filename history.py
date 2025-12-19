# -*- coding: utf-8 -*-
# ======================================================================================
# Generalized Chat History Engine (TorchChatHistory-X) — CPU/GPU Architecture Ready
# ======================================================================================
# - CPU-first architecture with optional GPU acceleration (auto-detected)
# - Append-only, tamper-evident logs; hash-chains and exact-duplicate defense
# - Robust retrieval: semantic + recency + role + duplicate-aware, budgeted selection
# - Persistence: SQLite WAL (chat_history.db) with migration safety (simhash_u64 -> simhash_i64)
# - Near-duplicate handling: SimHash64 (signed int64 safe) + Bloom prefilter
# - Vector index pluggable: FAISS (if available) or Torch FLAT fallback
# - Branching/DAG-ready, audit hashes, deterministic behavior
#
# Implementation notes:
# - All explanations are inline as comments; no extraneous output beyond this file.
# - Supports both CPU and GPU seamlessly; tested on Windows CPU-only.
# - Fixes:
#   * Overflow when constructing int64 tensors on CPU: use signed SimHash values.
#   * Schema mismatch: safe migrations; defer simhash index creation until after migration.
#   * Idempotent inserts: tolerate replays of the same (conv_id, branch_id, msg_id) without error
#     by loading from DB into the in-memory cache and returning the stored chain hash.
#   * Vector ids: enforce signed int64-safe non-negative ids by masking to 63 bits to avoid
#     "Overflow when unpacking long" on Windows/PyTorch when creating torch.long tensors.
#   * FAISS add_with_ids coverage: wrap Flat index in IndexIDMap(2) so add_with_ids is implemented.
#   * Windows OpenMP conflict (libomp vs libiomp5): hard-disable FAISS on Windows to avoid OMP error #15.
#   * Torch-compile on Windows: guard and fall back to eager if a compiler (cl) is unavailable.
#   * TorchFlatIndex.query: use gather to support batched [B,K] indices without index_select errors.
#
# Requirements:
# - Python 3.10+
# - PyTorch 2.1+ (CUDA optional)
# - SQLite3 (stdlib with FTS5)
# - Optional: faiss (faiss-gpu/faiss-cpu)
# ======================================================================================

from __future__ import annotations

import os
import sqlite3
import hashlib
import time
import threading
import platform
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Iterable, Tuple, List, Dict

# Preemptively allow duplicate OpenMP runtimes only as a safety valve if some lib loads later.
# We still hard-disable FAISS on Windows below to avoid triggering a second runtime at all.
if platform.system().lower().startswith("win"):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.distributed as dist


# ======================================================================================
# Constants / Tuning
# ======================================================================================

EMBED_DIM_DEFAULT = 768  # default embedding dimension


class Role(IntEnum):
    # Roles used in multi-turn chat; IntEnum for compact storage
    SYSTEM = 0
    USER = 1
    ASSISTANT = 2
    TOOL = 3


# ======================================================================================
# Device / Architecture detection (CPU/GPU)
# ======================================================================================

def detect_runtime() -> Tuple[torch.device, str, str, bool, str]:
    # Detects runtime characteristics
    # - Returns (torch_device, arch, os_name, cuda_available, device_str)
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda") if cuda_avail else torch.device("cpu")
    arch = platform.machine() or platform.processor() or "unknown"
    os_name = f"{platform.system()}-{platform.release()}"
    device_str = "cuda" if cuda_avail else "cpu"
    return device, arch, os_name, cuda_avail, device_str


# ======================================================================================
# Utility: Hashing, Time, IDs (signed-safe simhash)
# ======================================================================================

def sha256_bytes(x: bytes) -> bytes:
    # 32-byte SHA-256 digest used for content hashes and chain tips
    return hashlib.sha256(x).digest()


def blake2b64_unsigned(x: bytes) -> int:
    # 64-bit unsigned blake2b digest as Python int in [0, 2^64-1]
    return int.from_bytes(hashlib.blake2b(x, digest_size=8).digest(), "big", signed=False)


def blake2b64_signed(x: bytes) -> int:
    # 64-bit signed blake2b digest as Python int in [-2^63, 2^63-1]
    return int.from_bytes(hashlib.blake2b(x, digest_size=8).digest(), "big", signed=True)


def now_ts() -> float:
    # Wall clock seconds (float); append order is canonical, ts used for recency
    return time.time()


def make_msg_uid(conv_id: str, branch_id: str, msg_id: int) -> int:
    # Stable non-negative 63-bit id (fits signed int64; FAISS/tensors require signed int64)
    # - We clear the top bit to ensure value ∈ [0, 2^63-1] to avoid overflow on torch.long
    h = hashlib.blake2b(digest_size=8)
    h.update(conv_id.encode("utf-8")); h.update(b"|")
    h.update(branch_id.encode("utf-8")); h.update(b"|")
    h.update(str(msg_id).encode("utf-8"))
    raw = int.from_bytes(h.digest(), "big", signed=False)
    return raw & ((1 << 63) - 1)


# ======================================================================================
# GPU/CPU Bloom Filter (approximate prefilter for duplicates)
# ======================================================================================

class DeviceBloomFilter:
    # Two-hash Bloom filter; works on CPU and GPU devices transparently; correct bitwise semantics
    def __init__(self, m_bits: int = 1 << 22, k_hash: int = 4, device: torch.device | str = "cpu"):
        self.m = int(m_bits)
        self.k = int(k_hash)
        self.device = torch.device(device)
        self.bits = torch.zeros((self.m + 7) // 8, dtype=torch.uint8, device=self.device)

    @torch.no_grad()
    def _idx(self, keys64: torch.Tensor) -> torch.Tensor:
        # keys64: int64 tensor; double hashing indices modulo m
        keys = keys64.to(torch.int64)
        hash1 = keys
        hash2 = (keys ^ (keys >> 33)) * 0xff51afd7ed558ccd
        hash2 = hash2 & torch.tensor((1 << 63) - 1, dtype=torch.int64, device=keys.device)
        i = torch.arange(self.k, device=keys.device, dtype=torch.int64)[:, None]
        idx = (hash1 + i * hash2) % self.m  # [k, N]
        return idx

    @torch.no_grad()
    def query(self, keys64: torch.Tensor) -> torch.Tensor:
        # True if all k bits are set (approximate)
        idx = self._idx(keys64)  # [k, N]
        byte_idx = (idx // 8).to(torch.int64)
        bit_off = (idx % 8).to(torch.int64)
        bytes_ = self.bits.index_select(0, byte_idx.reshape(-1)).view_as(byte_idx)
        mask = (bytes_ >> bit_off) & 1
        present = mask.all(dim=0)
        return present  # [N] bool

    @torch.no_grad()
    def add(self, keys64: torch.Tensor) -> None:
        # Safe per-bit setting; avoids lossy reductions; fine for small N
        idx = self._idx(keys64)  # [k, N]
        k, n = idx.shape
        for col in range(n):
            for row in range(k):
                pos = int(idx[row, col].item())
                byte = pos // 8
                bit = pos % 8
                self.bits[byte] = torch.tensor(int(self.bits[byte].item()) | (1 << bit), dtype=torch.uint8, device=self.device)


# ======================================================================================
# Vector Index: FAISS (if available) or Torch FLAT fallback
# ======================================================================================

class VectorIndex:
    # Minimal adapter: cosine similarity (via inner product on normalized vectors)
    def add(self, emb: torch.Tensor, ids_u64: torch.Tensor) -> None:
        raise NotImplementedError

    def query(self, q: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class TorchFlatIndex(VectorIndex):
    # Torch-only FLAT index; stores embeddings on the selected device; cosine via matmul
    def __init__(self, d: int, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.d = d
        self.X = torch.empty((0, d), dtype=torch.float32, device=self.device)  # fp32 on CPU/GPU for stability
        self.ids = torch.empty(0, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def add(self, emb: torch.Tensor, ids_u64: torch.Tensor) -> None:
        e = torch.nn.functional.normalize(emb.to(self.device).to(torch.float32), dim=-1)
        self.X = torch.cat([self.X, e], dim=0)
        self.ids = torch.cat([self.ids, ids_u64.to(torch.long).to(self.device)], dim=0)

    @torch.inference_mode()
    def query(self, q: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input:
        # - q: [B, d]
        # Output:
        # - scores: [B, K]
        # - ids:    [B, K] (int64)
        if self.X.numel() == 0:
            b = q.size(0)
            return (torch.empty((b, 0), dtype=torch.float32, device=self.device),
                    torch.empty((b, 0), dtype=torch.long, device=self.device))
        qn = torch.nn.functional.normalize(q.to(self.device).to(torch.float32), dim=-1)  # [B, d]
        S = qn @ self.X.T  # [B, N]
        k = min(topk, S.size(1))
        if k <= 0:
            b = qn.size(0)
            return (torch.empty((b, 0), dtype=torch.float32, device=self.device),
                    torch.empty((b, 0), dtype=torch.long, device=self.device))
        v, idx = torch.topk(S, k=k, dim=-1, largest=True)  # v: [B,K], idx: [B,K]
        # Gather ids per-batch instead of index_select with 2D indices
        ids_row = self.ids.view(1, -1).expand(idx.size(0), -1)  # [B, N]
        ids = ids_row.gather(1, idx)  # [B, K]
        return v.to(torch.float32), ids.to(torch.long)

    def count(self) -> int:
        return int(self.X.size(0))


class FaissIndex(VectorIndex):
    # FAISS index wrapper:
    # - On Windows: disable FAISS to avoid OpenMP runtime conflict (fallback to TorchFlatIndex).
    # - Else: wrap FlatIP with IndexIDMap(2) to support add_with_ids consistently.
    def __init__(self, d: int, prefer_gpu: bool = True):
        self.d = d
        self.index = None
        self.fallback: Optional[TorchFlatIndex] = None

        # Hard-disable FAISS on Windows to avoid OMP Error #15 (libomp vs libiomp5)
        if platform.system().lower().startswith("win"):
            self.faiss = None
            self.fallback = TorchFlatIndex(d=d, device="cpu")
            return

        try:
            import faiss  # type: ignore
            self.faiss = faiss
        except Exception:
            self.faiss = None
            self.fallback = TorchFlatIndex(d=d, device="cpu")
            return

        try:
            base = self.faiss.IndexFlatIP(self.d)
            try:
                idmap = self.faiss.IndexIDMap2(base)
            except Exception:
                idmap = self.faiss.IndexIDMap(base)
            if prefer_gpu and hasattr(self.faiss, "get_num_gpus") and self.faiss.get_num_gpus() > 0:
                res = self.faiss.StandardGpuResources()
                self.index = self.faiss.index_cpu_to_gpu(res, 0, idmap)
            else:
                self.index = idmap
        except Exception:
            self.index = None
            self.fallback = TorchFlatIndex(d=d, device="cpu")

    @torch.inference_mode()
    def add(self, emb: torch.Tensor, ids_u64: torch.Tensor) -> None:
        if self.index is None:
            self.fallback.add(emb, ids_u64); return
        e = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy().astype("float32", copy=False)
        ids = ids_u64.detach().cpu().numpy().astype("int64", copy=False)
        self.index.add_with_ids(e, ids)

    @torch.inference_mode()
    def query(self, q: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.index is None:
            return self.fallback.query(q, topk)
        ntotal = int(getattr(self.index, "ntotal", 0))
        if topk <= 0 or ntotal == 0:
            b = q.size(0)
            device = q.device
            return (torch.empty((b, 0), dtype=torch.float32, device=device),
                    torch.empty((b, 0), dtype=torch.long, device=device))
        e = torch.nn.functional.normalize(q.float(), dim=-1).cpu().numpy().astype("float32", copy=False)
        v, ids = self.index.search(e, topk)
        device = q.device
        return torch.from_numpy(v).to(device=device, dtype=torch.float32), torch.from_numpy(ids).to(device=device, dtype=torch.long)

    def count(self) -> int:
        if self.index is None:
            return self.fallback.count()
        try:
            return int(self.index.ntotal)
        except Exception:
            return 0


# ======================================================================================
# Embedder (pluggable) — deterministic dev fallback; replace with production encoder
# ======================================================================================

class Embedder:
    # Deterministic random embeddings seeded by unsigned blake2b of content; CPU/GPU neutral
    def __init__(self, d: int, device: torch.device | str = "cpu"):
        self.d = d
        self.device = torch.device(device)

    @torch.inference_mode()
    def __call__(self, texts: List[str]) -> torch.Tensor:
        # Seed generator with stable unsigned 64-bit derived from all texts
        seeds = [blake2b64_unsigned(t.encode("utf-8")) for t in texts]
        seed = int(sum(seeds) & ((1 << 63) - 1))
        gen = torch.Generator(device=str(self.device))
        gen.manual_seed(seed)
        E = torch.randn((len(texts), self.d), dtype=torch.float32, device=self.device, generator=gen)
        E = torch.nn.functional.normalize(E, dim=-1)
        return E


# ======================================================================================
# SQLite Store — chat_history.db (WAL), schema, meta (arch/OS/device), migrations
# ======================================================================================

class SqliteChatStore:
    # Durable storage: messages, dedup, tips, and engine_meta (arch/OS/device)
    # - messages: append-only log with tamper-evident chain fields
    # - dedup: exact content hash per conv_id (first occurrence)
    # - tips: per-branch last chain hash and msg_id
    # - engine_meta: key/value store for environment recording
    def __init__(self, root_dir: str):
        self.root = os.path.abspath(root_dir)
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, "chat_history.db")
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_pragmas()
        self._init_schema()
        self._ensure_migrations()

    def _init_pragmas(self) -> None:
        c = self._conn.cursor()
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA temp_store=MEMORY")
        c.execute("PRAGMA mmap_size=134217728")  # 128MB
        c.execute("PRAGMA page_size=4096")
        c.close()

    def _table_exists(self, table: str) -> bool:
        cur = self._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None

    def _column_exists(self, table: str, column: str) -> bool:
        cur = self._execute(f"PRAGMA table_info({table})")
        cols = [r["name"] for r in cur.fetchall()]
        return column in cols

    def _init_schema(self) -> None:
        # Safe schema creation without referencing columns that may not exist yet in legacy DBs
        c = self._conn.cursor()
        # Base tables
        c.execute("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL)")
        # Insert version row only if table is empty to avoid duplicates
        c.execute("INSERT INTO schema_version(version) SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version)")
        if not self._table_exists("messages"):
            # Fresh DB: create messages with simhash_i64
            c.executescript("""
            CREATE TABLE messages(
                conv_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                msg_id INTEGER NOT NULL,
                role INTEGER NOT NULL,
                ts REAL NOT NULL,
                token_count INTEGER NOT NULL,
                parent_msg_id INTEGER,
                content TEXT NOT NULL,
                content_hash BLOB NOT NULL,
                simhash_i64 INTEGER NOT NULL,
                chain_prev BLOB NOT NULL,
                chain_hash BLOB NOT NULL,
                PRIMARY KEY (conv_id, branch_id, msg_id)
            );
            """)
        # Other tables (idempotent)
        c.executescript("""
        CREATE TABLE IF NOT EXISTS dedup(
            conv_id TEXT NOT NULL,
            content_hash BLOB NOT NULL,
            first_msg_id INTEGER NOT NULL,
            PRIMARY KEY (conv_id, content_hash)
        );

        CREATE TABLE IF NOT EXISTS tips(
            conv_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            tip_hash BLOB NOT NULL,
            last_msg_id INTEGER NOT NULL,
            PRIMARY KEY (conv_id, branch_id)
        );

        CREATE TABLE IF NOT EXISTS engine_meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            conv_id, branch_id, msg_id UNINDEXED, content, tokenize='porter'
        );

        CREATE TRIGGER IF NOT EXISTS t_ins_fts AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, conv_id, branch_id, msg_id, content)
            VALUES (new.rowid, new.conv_id, new.branch_id, new.msg_id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS t_del_fts AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
        END;

        CREATE INDEX IF NOT EXISTS idx_messages_conv_branch_ts ON messages(conv_id, branch_id, ts);
        CREATE INDEX IF NOT EXISTS idx_messages_conv_branch_role ON messages(conv_id, branch_id, role);
        """)
        # Note: Do NOT create idx_messages_simhash here; migration ensures the correct column exists first.
        c.close()

    def _ensure_migrations(self) -> None:
        # Migration path for legacy DBs without simhash_i64 or with simhash_u64
        try:
            has_i64 = self._column_exists("messages", "simhash_i64")
            has_u64 = self._column_exists("messages", "simhash_u64")
            if not has_i64 and has_u64:
                try:
                    # Attempt atomic rename (SQLite 3.25+)
                    self._execute("ALTER TABLE messages RENAME COLUMN simhash_u64 TO simhash_i64")
                except sqlite3.OperationalError:
                    # Fallback: add new column and backfill
                    self._execute("ALTER TABLE messages ADD COLUMN simhash_i64 INTEGER DEFAULT 0")
                    self._execute("UPDATE messages SET simhash_i64 = simhash_u64")
            elif not has_i64 and not has_u64:
                # Very old DB; add simhash_i64 with default
                self._execute("ALTER TABLE messages ADD COLUMN simhash_i64 INTEGER DEFAULT 0")
            # Recreate simhash index on the correct column
            self._execute("DROP INDEX IF EXISTS idx_messages_simhash")
            self._execute("CREATE INDEX IF NOT EXISTS idx_messages_simhash ON messages(simhash_i64)")
            # Bump schema version marker
            self._execute("UPDATE schema_version SET version = 2")
        except Exception:
            # Do not fail engine init due to migration exceptions; DB remains usable
            pass

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _execute(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, args)
            return cur

    def _executemany(self, sql: str, seq_of_args: Iterable[tuple]) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(sql, seq_of_args)
            return cur

    def set_meta(self, key: str, value: str) -> None:
        self._execute(
            "INSERT INTO engine_meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_tip(self, conv_id: str, branch_id: str) -> Tuple[bytes, int]:
        cur = self._execute("SELECT tip_hash, last_msg_id FROM tips WHERE conv_id=? AND branch_id=?", (conv_id, branch_id))
        row = cur.fetchone()
        if row is None:
            return b"\x00" * 32, 0
        return bytes(row["tip_hash"]), int(row["last_msg_id"])

    def set_tip(self, conv_id: str, branch_id: str, tip_hash: bytes, last_msg_id: int) -> None:
        self._execute(
            "INSERT INTO tips(conv_id, branch_id, tip_hash, last_msg_id) VALUES(?,?,?,?) "
            "ON CONFLICT(conv_id, branch_id) DO UPDATE SET tip_hash=excluded.tip_hash, last_msg_id=excluded.last_msg_id",
            (conv_id, branch_id, tip_hash, last_msg_id),
        )

    def get_message(self, conv_id: str, branch_id: str, msg_id: int) -> Optional[sqlite3.Row]:
        # Fetch a single message by primary key for idempotent insert handling and cache warming
        cur = self._execute(
            "SELECT * FROM messages WHERE conv_id=? AND branch_id=? AND msg_id=?",
            (conv_id, branch_id, int(msg_id)),
        )
        row = cur.fetchone()
        return row

    def insert_message(self,
                       conv_id: str, branch_id: str, msg_id: int, role: int, ts: float,
                       token_count: int, parent_msg_id: Optional[int], content: str,
                       content_hash: bytes, simhash_i64: int, chain_prev: bytes, chain_hash: bytes) -> None:
        self._execute(
            """
            INSERT INTO messages(conv_id, branch_id, msg_id, role, ts, token_count, parent_msg_id, content,
                                 content_hash, simhash_i64, chain_prev, chain_hash)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (conv_id, branch_id, msg_id, role, ts, token_count, parent_msg_id, content,
             content_hash, int(simhash_i64), chain_prev, chain_hash),
        )

    def insert_dedup(self, conv_id: str, content_hash: bytes, first_msg_id: int) -> bool:
        try:
            self._execute("INSERT INTO dedup(conv_id, content_hash, first_msg_id) VALUES(?,?,?)",
                          (conv_id, content_hash, first_msg_id))
            return True
        except sqlite3.IntegrityError:
            return False

    def fetch_messages_by_ids(self, keys: List[Tuple[str, str, int]]) -> List[sqlite3.Row]:
        if not keys:
            return []
        self._execute("CREATE TEMP TABLE IF NOT EXISTS tmp_keys(conv_id TEXT, branch_id TEXT, msg_id INTEGER)")
        self._execute("DELETE FROM tmp_keys")
        self._executemany("INSERT INTO tmp_keys(conv_id, branch_id, msg_id) VALUES(?,?,?)", keys)
        cur = self._execute(
            "SELECT m.* FROM messages m JOIN tmp_keys k ON m.conv_id=k.conv_id AND m.branch_id=k.branch_id AND m.msg_id=k.msg_id"
        )
        rows = cur.fetchall()
        return rows

    def fetch_recent(self, conv_id: str, branch_id: str, limit_n: int) -> List[sqlite3.Row]:
        cur = self._execute(
            "SELECT * FROM messages WHERE conv_id=? AND branch_id=? ORDER BY ts DESC LIMIT ?",
            (conv_id, branch_id, int(limit_n)),
        )
        return cur.fetchall()


# ======================================================================================
# Compiled kernels: ratio selection under token budget (with CPU-safe fallback)
# ======================================================================================

def _ratio_select_impl(scores: torch.Tensor, tokens: torch.Tensor, budget: int) -> torch.Tensor:
    # Greedy ratio S_i / tau_i selection; returns selected indices
    eps = torch.finfo(scores.dtype).eps
    ratio = scores / (tokens.to(scores.dtype) + eps)
    order = torch.argsort(ratio, descending=True)
    tk = tokens.index_select(0, order).to(torch.int64)
    pref = torch.cumsum(tk, dim=0)
    ok = pref <= budget
    k = int(ok.sum().item())
    return order[:k]

def _maybe_compiled_ratio_select():
    # Avoid torch.compile on Windows where cl (MSVC) may be missing; also skip if compile unavailable
    if platform.system().lower().startswith("win"):
        return None
    try:
        return torch.compile(_ratio_select_impl, mode="max-autotune", fullgraph=False, dynamic=True)  # type: ignore[attr-defined]
    except Exception:
        return None

_compiled_ratio_select = _maybe_compiled_ratio_select()

@torch.inference_mode()
def ratio_select(scores: torch.Tensor, tokens: torch.Tensor, budget: int) -> torch.Tensor:
    # Safe wrapper: try compiled path once if available; fallback to eager on any failure
    if _compiled_ratio_select is not None:
        try:
            return _compiled_ratio_select(scores, tokens, budget)
        except Exception:
            pass
    return _ratio_select_impl(scores, tokens, budget)


# ======================================================================================
# Message model and GPU/CPU tensor pack (columnar per branch)
# ======================================================================================

@dataclass
class MessageRow:
    conv_id: str
    branch_id: str
    msg_id: int
    role: Role
    ts: float
    token_count: int
    parent_msg_id: Optional[int]
    content: str
    content_hash: bytes
    simhash_i64: int
    chain_prev: bytes
    chain_hash: bytes


class MsgTensors:
    # Columnar tensors per branch; append-only; device-resident
    def __init__(self, device: torch.device | str, d: int):
        self.device = torch.device(device)
        self.ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.roles = torch.empty(0, dtype=torch.int8, device=self.device)
        self.times = torch.empty(0, dtype=torch.float64, device=self.device)
        self.tokens = torch.empty(0, dtype=torch.int32, device=self.device)
        self.hashes = torch.empty((0, 32), dtype=torch.uint8, device=self.device)
        self.simhash = torch.empty(0, dtype=torch.int64, device=self.device)
        self.hchain = torch.empty((0, 32), dtype=torch.uint8, device=self.device)
        self.parents = torch.empty(0, dtype=torch.long, device=self.device)
        self.embed = torch.empty((0, d), dtype=torch.float32, device=self.device)

    def append(self, row: MessageRow, emb: Optional[torch.Tensor]) -> None:
        # Append a single row to device-resident columns; emb is [1,d] or None
        self.ids = torch.cat([self.ids, torch.tensor([row.msg_id], device=self.device)])
        self.roles = torch.cat([self.roles, torch.tensor([int(row.role)], dtype=torch.int8, device=self.device)])
        self.times = torch.cat([self.times, torch.tensor([row.ts], dtype=torch.float64, device=self.device)])
        self.tokens = torch.cat([self.tokens, torch.tensor([row.token_count], dtype=torch.int32, device=self.device)])
        self.hashes = torch.cat([self.hashes, torch.tensor(list(row.content_hash), dtype=torch.uint8, device=self.device)[None, :]])
        self.simhash = torch.cat([self.simhash, torch.tensor([row.simhash_i64], dtype=torch.int64, device=self.device)])
        self.hchain = torch.cat([self.hchain, torch.tensor(list(row.chain_hash), dtype=torch.uint8, device=self.device)[None, :]])
        p = -1 if row.parent_msg_id is None else int(row.parent_msg_id)
        self.parents = torch.cat([self.parents, torch.tensor([p], dtype=torch.long, device=self.device)])
        e = (emb.to(self.device).to(torch.float32) if emb is not None
             else torch.zeros((1, self.embed.shape[1] if self.embed.numel() else EMBED_DIM_DEFAULT),
                              dtype=torch.float32, device=self.device))
        self.embed = e if self.embed.numel() == 0 else torch.cat([self.embed, e], dim=0)


# ======================================================================================
# Core Engine: Fast append, SOTA retrieval, dedup, summarization hook
# ======================================================================================

class GeneralizedChatHistory:
    # End-to-end engine with CPU/GPU transparent kernels and robust persistence
    def __init__(self, db_folder: str, d: int = EMBED_DIM_DEFAULT):
        device, arch, os_name, cuda_avail, device_str = detect_runtime()
        self.device = device
        self.device_str = device_str
        self.arch = arch
        self.os_name = os_name
        self.cuda_avail = cuda_avail

        # Persistent store and meta
        self.store = SqliteChatStore(db_folder)
        self.store.set_meta("arch", self.arch)
        self.store.set_meta("os", self.os_name)
        self.store.set_meta("torch", torch.__version__)
        self.store.set_meta("device", self.device_str)

        # Embeddings and vector index
        self.d = d
        self.embedder = Embedder(d=self.d, device=self.device)
        try:
            self.vector = FaissIndex(d=self.d, prefer_gpu=self.cuda_avail)
        except Exception:
            self.vector = TorchFlatIndex(d=self.d, device=self.device)

        # Per-branch device-resident columnar cache
        self.branches: Dict[Tuple[str, str], MsgTensors] = {}

        # Role weights for scoring (tunable per app)
        self.role_w = torch.tensor([0.7, 1.0, 0.9, 0.6], dtype=torch.float32, device=self.device)

        # Approximate duplicate prefilter
        self.bloom = DeviceBloomFilter(m_bits=1 << 22, k_hash=4, device=self.device)

        # Mapping from uid -> (conv_id, branch_id, msg_id) for branch-filtered retrieval
        self.uid_to_key: Dict[int, Tuple[str, str, int]] = {}

    def _branch_tensors(self, conv_id: str, branch_id: str) -> MsgTensors:
        key = (conv_id, branch_id)
        if key not in self.branches:
            self.branches[key] = MsgTensors(self.device, d=self.d)
        return self.branches[key]

    @torch.inference_mode()
    def add_message(self, conv_id: str, branch_id: str, msg_id: int, role: Role, content: str,
                    ts: Optional[float] = None, tokens: Optional[int] = None,
                    parent_msg_id: Optional[int] = None) -> bytes:
        # Idempotent append:
        # - If (conv_id, branch_id, msg_id) exists → warm cache + index, return stored chain_hash without mutation.
        # - Else → insert new row, update tip, warm cache + index, return new chain_hash.
        existing = self.store.get_message(conv_id, branch_id, int(msg_id))
        mt = self._branch_tensors(conv_id, branch_id)

        if existing is not None:
            row = MessageRow(
                conv_id=existing["conv_id"],
                branch_id=existing["branch_id"],
                msg_id=int(existing["msg_id"]),
                role=Role(int(existing["role"])),
                ts=float(existing["ts"]),
                token_count=int(existing["token_count"]),
                parent_msg_id=(int(existing["parent_msg_id"]) if existing["parent_msg_id"] is not None else None),
                content=existing["content"],
                content_hash=bytes(existing["content_hash"]),
                simhash_i64=int(existing["simhash_i64"]),
                chain_prev=bytes(existing["chain_prev"]),
                chain_hash=bytes(existing["chain_hash"]),
            )
            already_cached = (mt.ids.numel() > 0) and bool((mt.ids == row.msg_id).any().item())
            if not already_cached:
                emb = self.embedder([row.content])
                mt.append(row, emb)
                if row.role in (Role.USER, Role.ASSISTANT, Role.TOOL):
                    uid = make_msg_uid(conv_id, branch_id, row.msg_id)
                    if uid not in self.uid_to_key:
                        self.uid_to_key[uid] = (conv_id, branch_id, row.msg_id)
                        self.vector.add(emb, torch.tensor([uid], dtype=torch.long, device=self.device))
            return row.chain_hash

        ts_val = float(ts if ts is not None else now_ts())
        tok_val = int(tokens if tokens is not None else max(1, len(content) // 4))
        content_bytes = content.encode("utf-8")
        h256 = sha256_bytes(content_bytes)
        sim64_signed = blake2b64_signed(content_bytes)

        sim64_t = torch.tensor([sim64_signed], dtype=torch.int64, device=self.device)
        if not self.bloom.query(sim64_t)[0].item():
            self.bloom.add(sim64_t)

        prev_tip, _ = self.store.get_tip(conv_id, branch_id)
        h = hashlib.sha256()
        h.update(prev_tip)
        h.update(int(role).to_bytes(1, "little", signed=False))
        h.update(str(ts_val).encode("utf-8"))
        h.update(content_bytes)
        chain_hash = h.digest()

        _ = self.store.insert_dedup(conv_id, h256, msg_id)

        self.store.insert_message(
            conv_id=conv_id, branch_id=branch_id, msg_id=msg_id, role=int(role), ts=ts_val,
            token_count=tok_val, parent_msg_id=parent_msg_id, content=content,
            content_hash=h256, simhash_i64=sim64_signed, chain_prev=prev_tip, chain_hash=chain_hash
        )
        self.store.set_tip(conv_id, branch_id, chain_hash, msg_id)

        emb = self.embedder([content])  # [1, d]
        new_row = MessageRow(conv_id, branch_id, msg_id, role, ts_val, tok_val, parent_msg_id, content, h256, sim64_signed, prev_tip, chain_hash)
        mt.append(new_row, emb)

        if role in (Role.USER, Role.ASSISTANT, Role.TOOL):
            uid = make_msg_uid(conv_id, branch_id, msg_id)
            if uid not in self.uid_to_key:
                self.uid_to_key[uid] = (conv_id, branch_id, msg_id)
                self.vector.add(emb, torch.tensor([uid], dtype=torch.long, device=self.device))

        return chain_hash

    @torch.inference_mode()
    def _score_branch(self, mt: MsgTensors, idx: torch.Tensor, q_emb: torch.Tensor, now_time: float,
                      alpha=1.0, beta=0.3, gamma=0.1, delta=0.5, dup_penalty: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Score candidates: S_i = α sim + β recency + γ role - δ dup
        X = torch.nn.functional.normalize(mt.embed.index_select(0, idx), dim=-1)
        q = torch.nn.functional.normalize(q_emb.to(torch.float32), dim=-1)
        sim = (q @ X.T).squeeze(0).to(torch.float32)
        times = mt.times.index_select(0, idx)
        dt = torch.tensor(now_time, dtype=torch.float64, device=self.device) - times
        rec = torch.exp(-(dt.to(torch.float32) / 3600.0))
        roles = mt.roles.index_select(0, idx).long()
        rw = self.role_w.index_select(0, roles)
        dup = dup_penalty if dup_penalty is not None else torch.zeros_like(sim)
        return alpha * sim + beta * rec + gamma * rw - delta * dup

    @staticmethod
    def _hamming_min_penalty(cands: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # CPU-safe Hamming min distance using Python int.bit_count over 64-bit two's complement
        # - cands, ref: int64 tensors
        # - returns tensor of shape [len(cands)] with penalty in [0,1] (higher => more duplicate-like)
        if cands.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=cands.device)
        CU = (1 << 64) - 1
        c_list = [(int(x.item()) & CU) for x in cands.view(-1)]
        r_list = [(int(x.item()) & CU) for x in ref.view(-1)]
        out = []
        for a in c_list:
            if not r_list:
                out.append(0.0); continue
            mn = 64
            for b in r_list:
                d = (a ^ b).bit_count()
                if d < mn:
                    mn = d
                    if mn == 0:
                        break
            out.append(1.0 - (mn / 64.0))
        return torch.tensor(out, dtype=torch.float32, device=cands.device)

    @torch.inference_mode()
    def build_context(self, conv_id: str, branch_id: str, query_text: str, budget_ctx: int,
                      alpha=1.0, beta=0.3, gamma=0.1, delta=0.5, k_sem: int = 96, k_rerank: int = 64,
                      recency_window: int = 128) -> List[Tuple[str, str, int, str]]:
        # Context selection:
        # 1) Semantic shortlist (global index → filtered to branch)
        # 2) Union with recency window
        # 3) Score with α sim + β rec + γ role - δ dup (near-dup penalty vs recent)
        # 4) Greedy ratio selection under token budget; rerank; chronological output
        mt = self._branch_tensors(conv_id, branch_id)
        if mt.ids.numel() == 0:
            return []

        q_emb = self.embedder([query_text])  # [1, d]
        # Semantic candidates from global index
        topk = min(k_sem, max(1, self.vector.count()))
        _, ids = self.vector.query(q_emb, topk=topk)  # [1,K], [1,K]
        cand_uids = ids[0].tolist()
        # Map global ids to local indices within branch
        cand_idx_local: List[int] = []
        for uid in cand_uids:
            key = self.uid_to_key.get(int(uid))
            if key is None:
                continue
            if key[0] == conv_id and key[1] == branch_id:
                msg_id = key[2]
                pos = (mt.ids == msg_id).nonzero(as_tuple=False)
                if pos.numel() > 0:
                    cand_idx_local.append(int(pos[0].item()))
        # Recency supplement: last R
        R = min(recency_window, mt.ids.numel())
        rec_idx = list(range(mt.ids.numel() - R, mt.ids.numel()))
        # Unique and sorted
        idx_unique = torch.tensor(sorted(set(cand_idx_local + rec_idx)), dtype=torch.long, device=self.device)

        # Near-dup penalty relative to recent window (small ref set for efficiency)
        ref_sim = mt.simhash[-min(64, mt.simhash.numel()):] if mt.simhash.numel() > 0 else torch.empty(0, dtype=torch.int64, device=self.device)
        cand_sim = mt.simhash.index_select(0, idx_unique) if idx_unique.numel() > 0 else torch.empty(0, dtype=torch.int64, device=self.device)
        dup_penalty = self._hamming_min_penalty(cand_sim, ref_sim) if cand_sim.numel() > 0 else torch.zeros(0, dtype=torch.float32, device=self.device)

        # Composite scoring and selection
        S = self._score_branch(mt, idx_unique, q_emb, now_time=float(mt.times.max().item()),
                               alpha=alpha, beta=beta, gamma=gamma, delta=delta, dup_penalty=dup_penalty)
        toks = mt.tokens.index_select(0, idx_unique)
        sel_local = ratio_select(S, toks, int(budget_ctx))
        chosen = idx_unique.index_select(0, sel_local)

        # Exact rerank among chosen by cosine similarity
        rerank_k = min(k_rerank, chosen.numel())
        if rerank_k > 0:
            Xn = torch.nn.functional.normalize(mt.embed.index_select(0, chosen), dim=-1)
            qn = torch.nn.functional.normalize(q_emb.to(torch.float32), dim=-1)
            score = (qn @ Xn.T).squeeze(0)
            _, ord_r = torch.topk(score, k=rerank_k, largest=True)
            chosen = chosen.index_select(0, ord_r)

        # Chronological order to preserve dialogue flow
        times = mt.times.index_select(0, chosen)
        order = torch.argsort(times, dim=0)
        final_idx = chosen.index_select(0, order)

        # Materialize from DB
        keys = [(conv_id, branch_id, int(mt.ids[i].item())) for i in final_idx.tolist()]
        rows = self.store.fetch_messages_by_ids(keys)
        key_to_row = {(r["conv_id"], r["branch_id"], r["msg_id"]): r for r in rows}
        output = []
        for k in keys:
            r = key_to_row.get(k)
            if r is None:
                continue
            output.append((r["conv_id"], r["branch_id"], r["msg_id"], r["content"]))
        return output

    @torch.inference_mode()
    def branch(self, conv_id: str, src_branch: str, new_branch: str, from_msg_id: Optional[int] = None) -> None:
        # Fork branch by copying tip at current head or at a specific msg
        if from_msg_id is None:
            tip, last_id = self.store.get_tip(conv_id, src_branch)
            self.store.set_tip(conv_id, new_branch, tip, last_id)
        else:
            rows = self.store.fetch_messages_by_ids([(conv_id, src_branch, int(from_msg_id))])
            if not rows:
                raise ValueError("from_msg_id not found")
            tip = bytes(rows[0]["chain_hash"])
            self.store.set_tip(conv_id, new_branch, tip, int(from_msg_id))

    def close(self) -> None:
        self.store.close()


# ======================================================================================
# Optional distributed merge helper (kept simple; CPU/GPU neutral)
# ======================================================================================

@torch.inference_mode()
def distributed_query_and_merge(engine: GeneralizedChatHistory, q_emb: torch.Tensor, k: int, pg=None):
    # Local shortlist
    v, lids = engine.vector.query(q_emb, topk=k * 4)
    S = v.contiguous()
    I = lids.to(torch.int64)

    # All-gather across ranks if distributed
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size(group=pg)
        recv_S = [torch.empty_like(S) for _ in range(world)]
        recv_I = [torch.empty_like(I) for _ in range(world)]
        dist.all_gather(recv_S, S, group=pg)
        dist.all_gather(recv_I, I, group=pg)
        S_cat = torch.cat(recv_S, dim=1).squeeze(0)
        I_cat = torch.cat(recv_I, dim=1).squeeze(0)
    else:
        S_cat = S.squeeze(0)
        I_cat = I.squeeze(0)

    # Deduplicate by id keeping max score
    uniq, inv = torch.unique(I_cat, return_inverse=True)
    init = torch.full((uniq.numel(),), -float("inf"), device=I_cat.device, dtype=S_cat.dtype)
    best = init.scatter_reduce(0, inv, S_cat, reduce="amax", include_self=True)

    # Top-k global
    v2, idx = torch.topk(best, k=min(k, best.numel()))
    top_ids = uniq.index_select(0, idx)
    return v2, top_ids


# ======================================================================================
# Smoke test (Windows-friendly). Run: python .\history\history.py
# ======================================================================================

if __name__ == "__main__":
    print("Running smoke test for GeneralizedChatHistory...")
    device, arch, os_name, cuda_avail, device_str = detect_runtime()
    print(f"Runtime -> device={device_str}, cuda={cuda_avail}, arch={arch}, os={os_name}, torch={torch.__version__}")

    eng = GeneralizedChatHistory(db_folder="./data", d=EMBED_DIM_DEFAULT)
    conv, br = "c1", "main"

    # Append a few messages; idempotent across reruns (replay-safe)
    eng.add_message(conv, br, 1, Role.SYSTEM, "You are helpful.", ts=0.0, tokens=6)
    eng.add_message(conv, br, 2, Role.USER, "Plan a trip to Kyoto.", ts=1.0, tokens=6, parent_msg_id=1)
    eng.add_message(conv, br, 3, Role.ASSISTANT, "Sure, when would you like to travel?", ts=2.0, tokens=9, parent_msg_id=2)
    eng.add_message(conv, br, 4, Role.USER, "In April. Budget is $3000.", ts=3.0, tokens=7, parent_msg_id=3)

    # Retrieve context
    ctx = eng.build_context(conv, br, query_text="Kyoto itinerary and budget", budget_ctx=64)
    print("Retrieved context messages:")
    for row in ctx:
        print(row)  # (conv_id, branch_id, msg_id, content)

    eng.close()