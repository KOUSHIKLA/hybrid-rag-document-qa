
# filename: main_rag_with_history.py
# -*- coding: utf-8 -*-
"""
RAG pipeline (end-to-end) driven by ragconfig.yaml.
- Chunking: torchchuck (ChunkConfig, chunk_from_files, write_chunks_jsonl)
- Embeddings: langchain_huggingface (HuggingFaceEmbeddings) with torch
- Vector index: torchvectorbase (EmbeddingAdapter, VectorBase, IVFBuildParams, IVFSearchParams)
- Advanced retrieval: rethinker_retrieval (Rethinker, RethinkerParams)
- Generation: vllm_generation (PromptTemplate, LLMConfig, LLMClient, CompletionResult, ChatResult)
- Feedback/History: history (GeneralizedChatHistory, Role, EMBED_DIM_DEFAULT, detect_runtime)

Key change:
- All runtime parameters are sourced from ragconfig.yaml (with optional env overrides per key).
- No hidden defaults; explicit typing, validation, and casting are enforced.
"""

import os
import sys
import json
import time
import asyncio
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from torchchuck import ChunkConfig, chunk_from_files, write_chunks_jsonl
from torchvectorbase import EmbeddingAdapter, VectorBase, IVFBuildParams, IVFSearchParams
from rethinker_retrieval import Rethinker, RethinkerParams
from vllm_generation import PromptTemplate, LLMConfig, LLMClient, CompletionResult, ChatResult

from history import GeneralizedChatHistory, Role, EMBED_DIM_DEFAULT, detect_runtime


# ----------------------------
# YAML configuration loading (self-contained; no hard dependency on PyYAML)
# ----------------------------

def _parse_scalar(val: str) -> Any:
    """
    Parse a scalar YAML-like value into Python types without external dependencies.
    - Handles: null/None, booleans, integers, floats, and quoted/unquoted strings.
    - This is intentionally conservative to match the simple flat structure of ragconfig.yaml.
    """
    s = val.strip()
    if not s:
        return ""
    # Strip matching quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    low = s.lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"

    # Numeric detection
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        # Float detection; ignore if it contains spaces or non-numeric punctuation (e.g., URLs)
        if all(ch.isdigit() or ch in ".-+" for ch in s) and any(ch in ".," for ch in s.replace(",", ".")):
            return float(s.replace(",", "."))
    except Exception:
        pass

    return s


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a minimal YAML key: value mapping from disk.
    - Supports comments starting with '#', blank lines, and simple scalars (string/int/float/bool/null).
    - Raises if the file is missing or contains no recognized key-value pairs.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ragconfig.yaml not found at: {path}")

    config: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Remove inline comments outside of quotes by a simple split (sufficient for this flat config)
            if "#" in line:
                line = line.split("#", 1)[0].rstrip()
            if not line:
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            config[key] = _parse_scalar(val)
    if not config:
        raise RuntimeError(f"ragconfig.yaml at {path} yielded no configuration entries.")
    return config


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overlay environment variables on top of the YAML configuration for known keys.
    - Ensures proper type coercion per field schema.
    """
    # Map config keys to env names and coercers
    def _to_int(x: Optional[str], default: int) -> int:
        try:
            return int(x) if x is not None else default
        except Exception:
            return default

    def _to_float(x: Optional[str], default: float) -> float:
        try:
            return float(x) if x is not None else default
        except Exception:
            return default

    def _to_str(x: Optional[str], default: str) -> str:
        return x if x is not None else default

    def _to_opt_str(x: Optional[str], default: Optional[str]) -> Optional[str]:
        if x is None:
            return default
        xl = x.lower()
        if xl in ("none", "null", "nil"):
            return None
        return x

    overrides = {
        "DEFAULT_INPUT_GLOB": ("VB_INPUT_GLOB", _to_str),
        "DEFAULT_CHUNKS_JSONL": ("VB_CHUNKS_JSONL", _to_str),
        "DEFAULT_INPUT_JSONL": ("VB_INPUT_JSONL", _to_str),
        "DEFAULT_COLLECTION": ("VB_COLLECTION", _to_str),
        "DEFAULT_QUERY": ("VB_QUERY", _to_str),
        "DEFAULT_EMBED_MODEL": ("VB_EMBED_MODEL", _to_str),
        "FORCED_DEVICE": ("VB_DEVICE", _to_opt_str),
        "LLM_BASE_URL": ("LLM_BASE_URL", _to_str),
        "LLM_API_KEY": ("LLM_API_KEY", _to_str),
        "LLM_MODEL": ("LLM_MODEL", _to_str),
        "LLM_TEMPERATURE": ("LLM_TEMPERATURE", _to_float),
        "LLM_MAX_TOKENS": ("LLM_MAX_TOKENS", _to_int),
        "CONV_ID": ("VB_CONV_ID", _to_str),
        "BRANCH_ID": ("VB_BRANCH_ID", _to_str),
    }

    new_cfg = dict(cfg)
    for key, (env_name, caster) in overrides.items():
        env_val = os.environ.get(env_name, None)
        if env_val is not None:
            default = new_cfg.get(key, None)
            # Type-aware casting: provide default typed baseline where helpful
            if key in ("LLM_MAX_TOKENS",):
                default = int(default) if isinstance(default, (int, float, str)) else 512
                new_cfg[key] = caster(env_val, default)  # type: ignore[arg-type]
            elif key in ("LLM_TEMPERATURE",):
                default = float(default) if isinstance(default, (int, float, str)) else 0.2
                new_cfg[key] = caster(env_val, default)  # type: ignore[arg-type]
            else:
                default = str(default) if default is not None else ""
                if caster is _to_opt_str:
                    new_cfg[key] = caster(env_val, None)  # type: ignore[arg-type]
                else:
                    new_cfg[key] = caster(env_val, default)  # type: ignore[arg-type]
    return new_cfg


def _validate_and_cast(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate required keys and coerce to expected types with strict semantics.
    - Raises descriptive errors if required parameters are missing or invalid.
    """
    required = [
        "EMBED_DIM",
        "DEFAULT_INPUT_GLOB",
        "DEFAULT_CHUNKS_JSONL",
        "DEFAULT_INPUT_JSONL",
        "DEFAULT_COLLECTION",
        "DEFAULT_QUERY",
        "DEFAULT_EMBED_MODEL",
        "FORCED_DEVICE",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "CONV_ID",
        "BRANCH_ID",
    ]
    for k in required:
        if k not in cfg:
            raise KeyError(f"Missing required config key: {k}")

    out: Dict[str, Any] = {}
    out["EMBED_DIM"] = int(cfg["EMBED_DIM"])
    out["DEFAULT_INPUT_GLOB"] = str(cfg["DEFAULT_INPUT_GLOB"])
    out["DEFAULT_CHUNKS_JSONL"] = str(cfg["DEFAULT_CHUNKS_JSONL"])
    out["DEFAULT_INPUT_JSONL"] = str(cfg["DEFAULT_INPUT_JSONL"])
    out["DEFAULT_COLLECTION"] = str(cfg["DEFAULT_COLLECTION"])
    out["DEFAULT_QUERY"] = str(cfg["DEFAULT_QUERY"])
    out["DEFAULT_EMBED_MODEL"] = str(cfg["DEFAULT_EMBED_MODEL"])
    out["FORCED_DEVICE"] = None if cfg["FORCED_DEVICE"] in (None, "None", "null") else str(cfg["FORCED_DEVICE"])
    out["LLM_BASE_URL"] = str(cfg["LLM_BASE_URL"])
    out["LLM_API_KEY"] = str(cfg["LLM_API_KEY"])
    out["LLM_MODEL"] = str(cfg["LLM_MODEL"])
    out["LLM_TEMPERATURE"] = float(cfg["LLM_TEMPERATURE"])
    out["LLM_MAX_TOKENS"] = int(cfg["LLM_MAX_TOKENS"])
    out["CONV_ID"] = str(cfg["CONV_ID"])
    out["BRANCH_ID"] = str(cfg["BRANCH_ID"])

    # Guard device string if provided
    if out["FORCED_DEVICE"] not in (None, "cpu", "cuda"):
        raise ValueError('FORCED_DEVICE must be one of: null, "cpu", or "cuda"')

    # Embedding dim sanity guard
    if out["EMBED_DIM"] <= 0:
        raise ValueError("EMBED_DIM must be a positive integer")

    return out


# ----------------------------
# Core pipeline helpers (unchanged behavior; now parameterized by config)
# ----------------------------

def _select_device(forced_device: Optional[str]) -> str:
    """
    Decide the device for both embedding and vector base computations.
    - Honors explicit FORCED_DEVICE if provided.
    - Otherwise uses cuda if available, else cpu.
    """
    if forced_device in ("cuda", "cpu"):
        return forced_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _estimate_tokens(text: str) -> int:
    """
    Lightweight, deterministic token estimate (no external tokenizer assumptions).
    - Conservative overestimation to keep budgets safe.
    """
    if not text:
        return 1
    return max(1, int(len(text.split()) * 4 // 3))


def _ensure_chunks(input_glob: str, out_jsonl: str) -> str:
    """
    Chunk source files and write to JSONL on disk.
    - Idempotent: re-chunks and overwrites by design to track source changes precisely.
    - Raises for empty results to fail fast.
    """
    cfg = ChunkConfig(
        chunk_size=1024,
        chunk_overlap=128,
        separators=("\n\n", "\n", " ", "", "."),
        keep_separator="end",
        device=None,
        distributed=False,
        return_offsets=True,
        deduplicate=False,
        compile=False,
        normalize_whitespace=True,
        max_space_run=1,
        max_newline_run=2,
        strip_line_edges=True,
        num_workers_io=8,
        num_workers_chunk=8,
        drop_blank_chunks=True,
    )
    chunks = chunk_from_files(input_glob, cfg)
    if not chunks or len(chunks) == 0:
        raise RuntimeError(f"No chunks produced from input pattern: {input_glob}")
    write_chunks_jsonl(chunks, out_jsonl)
    return out_jsonl


def _load_records(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL records from disk.
    - Expects one JSON-serialized dict per line.
    - Raises if file missing or empty.
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise RuntimeError(f"JSONL appears empty: {jsonl_path}")
    records = [json.loads(line) for line in lines]
    if not records:
        raise RuntimeError(f"Decoded zero records from: {jsonl_path}")
    return records


def _init_vectorbase(
    records: List[Dict[str, Any]],
    collection: str,
    embed_model: str,
    embed_dim: int,
    forced_device: Optional[str],
) -> Tuple[VectorBase, Dict[str, Any]]:
    """
    Initialize VectorBase with IVF index suitable for mid-sized collections.
    - Embedding: HF mpnet-base-v2 by default (dim=768), respecting normalize=True for cosine metric.
    - Index kind: IVF_OPQ_PQ with pq_m=16; nlist calibrated to <= N/2, clamped to [1, 4096].
    """
    device = _select_device(forced_device)
    hf = HuggingFaceEmbeddings(model_name=embed_model)
    adapter = EmbeddingAdapter(hf, device=device, normalize=True, batch_size=64)
    vb = VectorBase(adapter, dim=embed_dim, metric="cosine", device=adapter.device)

    vb.create_collection(collection, dim=embed_dim, metric="cosine")

    inserted_ids, stats = vb.insert(records, return_stats=True)
    if not stats or stats.get("inserted", 0) <= 0:
        raise RuntimeError("No records inserted into VectorBase; aborting index build.")

    n = int(stats["inserted"])
    nlist = min(max(1, n // 2), 4096)
    vb.build_index(kind="IVF_OPQ_PQ", params=IVFBuildParams(nlist=nlist, pq_m=16, train_samples=None))

    return vb, stats


def _run_retrieval(vb: VectorBase, query: str) -> Dict[str, Any]:
    """
    Execute advanced retrieval via Rethinker (semantic + lexical + graph expansion).
    - Fall back to IVF search if rethinker yields no contexts to preserve robustness.
    """
    rk = Rethinker(
        vb,
        params=RethinkerParams(
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
            max_chars_per_context=32000,
        ),
    )
    out = rk.search(query)
    if "contexts" not in out or not out["contexts"]:
        _, ctx = vb.search(
            query,
            k=5,
            filters=None,
            params=IVFSearchParams(
                nprobe=16,
                refine=200,
                topk=5,
                metric="cosine",
                flat_fallback=True,
            ),
        )
        out = {"contexts": ctx, "debug": {"fallback": True}}
    return out


def _format_contexts_for_prompt(contexts: List[Dict[str, Any]]) -> str:
    """
    Prepare retrieved contexts for prompt injection.
    - Stable, index-labelled formatting; emphasizes readability and grounding.
    """
    buf: List[str] = []
    for i, c in enumerate(contexts):
        text = c.get("text") if isinstance(c, dict) else None
        if not text and isinstance(c, dict):
            text = c.get("content", "")
        if not text:
            text = str(c)
        text = " ".join(text.strip().split())
        buf.append(f"[doc:{i+1}] {text}")
    return "\n".join(buf)


def _format_history_for_prompt(
    eng: GeneralizedChatHistory,
    conv_id: str,
    branch_id: str,
    query_text: str,
    budget_ctx: int = 256,
) -> str:
    """
    Build relevant chat history context under a fixed budget using the engine's selector.
    """
    rows = eng.build_context(conv_id, branch_id, query_text=query_text, budget_ctx=budget_ctx)
    if not rows:
        return ""
    lines: List[str] = []
    for _, _, _, content in rows:
        content = " ".join(str(content).strip().split())
        if content:
            lines.append(f"- {content}")
    return "\n".join(lines)


def _init_llm_client(
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMClient:
    """
    Initialize the LLMClient with explicit configuration.
    """
    cfg = LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout_s=60.0,
        max_retries=2,
        backoff_initial_s=0.25,
        backoff_max_s=2.0,
        jitter_s=0.1,
        max_concurrency=4,
    )
    return LLMClient(cfg)


async def _generate_with_context(
    client: LLMClient,
    query: str,
    contexts_text: str,
    history_text: str,
) -> str:
    """
    Compose the final prompt and request a chat completion.
    - System template constrains behavior: grounded, concise, technically correct.
    - User template injects query, retrieved contexts, and relevant chat history.
    """
    sys_tmpl = PromptTemplate(
        "You are a precise assistant. Ground answers strictly in the provided CONTEXT and HISTORY when relevant. "
        "Prefer clear structure and correctness. If unsure or missing context, state limitations."
    )
    usr_tmpl = PromptTemplate(
        "QUESTION:\n{question}\n\n"
        "HISTORY (relevant excerpts):\n{history}\n\n"
        "CONTEXT (retrieved documents):\n{context}\n\n"
        "RESPONSE REQUIREMENTS:\n"
        "- Provide a comprehensive, technically accurate explanation.\n"
        "- Reference context snippets implicitly by content; do not fabricate citations.\n"
        "- If context is insufficient, explain what is missing and proceed with general knowledge cautiously."
    )
    chat_res = await client.chat(
        system_template=sys_tmpl,
        user_template=usr_tmpl,
        template_vars={
            "question": query,
            "history": history_text if history_text else "(none)",
            "context": contexts_text if contexts_text else "(no retrieved context)",
        },
    )
    assert isinstance(chat_res, ChatResult)
    return chat_res.content


async def main() -> None:
    """
    Orchestrate the full RAG pipeline with feedback (history) using ragconfig.yaml.
    Steps:
    1) Load YAML config -> apply env overrides -> validate & cast.
    2) Runtime diagnostics.
    3) Chunking -> JSONL.
    4) Load JSONL records.
    5) Initialize VectorBase and build IVF index.
    6) Rethinker retrieval for the query.
    7) Build chat history + context and generate answer.
    8) Append user+assistant to history for feedback continuity.
    """
    # Load configuration strictly from ragconfig.yaml (with optional env overrides).
    cfg_path = os.environ.get("RAG_CONFIG", "ragconfig.yaml")
    cfg_yaml = _load_yaml_config(cfg_path)
    cfg_yaml = _apply_env_overrides(cfg_yaml)
    cfg = _validate_and_cast(cfg_yaml)

    # Diagnostics
    device, arch, os_name, cuda_avail, device_str = detect_runtime()
    print(f"[runtime] device={device_str} cuda={cuda_avail} arch={arch} os={os_name} torch={torch.__version__}")

    # History engine
    eng = GeneralizedChatHistory(db_folder="./data", d=EMBED_DIM_DEFAULT)

    # Chunking (explicit, idempotent)
    try:
        jsonl_path = _ensure_chunks(cfg["DEFAULT_INPUT_GLOB"], cfg["DEFAULT_CHUNKS_JSONL"])
    except Exception as e:
        # If chunking fails, proceed with pre-existing JSONL if specified and present.
        alt = cfg.get("DEFAULT_INPUT_JSONL", "")
        if alt and os.path.isfile(alt):
            print(f"[warn] chunking failed ({e}); proceeding with DEFAULT_INPUT_JSONL={alt}", file=sys.stderr)
            jsonl_path = alt
        else:
            raise

    # Ingest records
    records = _load_records(jsonl_path)

    # VectorBase + index
    vb, stats = _init_vectorbase(
        records=records,
        collection=cfg["DEFAULT_COLLECTION"],
        embed_model=cfg["DEFAULT_EMBED_MODEL"],
        embed_dim=cfg["EMBED_DIM"],
        forced_device=cfg["FORCED_DEVICE"],
    )
    print(f"[index] inserted={stats.get('inserted')}  dim={cfg['EMBED_DIM']}")

    # Retrieval
    query = cfg["DEFAULT_QUERY"]
    out = _run_retrieval(vb, query)
    contexts = out.get("contexts", [])
    debug = out.get("debug", {})

    # Format contexts and history
    contexts_text = _format_contexts_for_prompt(contexts)
    history_text = _format_history_for_prompt(eng, cfg["CONV_ID"], cfg["BRANCH_ID"], query_text=query, budget_ctx=256)

    # Initialize LLM
    client = _init_llm_client(
        base_url=cfg["LLM_BASE_URL"],
        api_key=cfg["LLM_API_KEY"],
        model=cfg["LLM_MODEL"],
        temperature=cfg["LLM_TEMPERATURE"],
        max_tokens=cfg["LLM_MAX_TOKENS"],
    )

    # Record user message into history
    
    t0 = time.time()
    user_tokens = _estimate_tokens(query)
    eng.add_message(cfg["CONV_ID"], cfg["BRANCH_ID"], int(t0 * 1000) + 1, Role.USER, query, ts=t0, tokens=user_tokens)

    # Generate answer
    try:
        answer = await _generate_with_context(client, query, contexts_text, history_text)
    except Exception as e:
        answer = f"I encountered an error during generation. Details: {type(e).__name__}: {str(e)}"

    # Append assistant message
    t1 = time.time()
    asst_tokens = _estimate_tokens(answer)
    eng.add_message(cfg["CONV_ID"], cfg["BRANCH_ID"], int(t1 * 1000) + 2, Role.ASSISTANT, answer, ts=t1, tokens=asst_tokens)

    # Output final results and retrieval diagnostics
    print("\n[answer]\n")
    print(answer.strip())
    # print("\n[retrieval_debug]\n")
    # print(json.dumps(debug, indent=2, ensure_ascii=False))

    # Close history engine
    eng.close()
    
    
def run_query(user_query: str):
    from rethinker_retrieval import ask_question
    return ask_question(user_query)



if __name__ == "__main__":
    asyncio.run(main())
