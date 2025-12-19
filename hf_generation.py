
# -*- coding: utf-8 -*-
# Ultra-optimized, production-grade text generation pipeline leveraging Hugging Face Transformers.
# Goals:
#   - Top-tier throughput/latency
#   - Robust device autodetection (NVIDIA CUDA / AMD ROCm / Apple MPS / CPU)
#   - Single- and multi-GPU sharding (accelerate device_map="auto")
#   - Optional bitsandbytes quantization
#   - Correct attention kernel selection with automatic fallback
#   - Clean support for new Transformers dtype API (torch_dtype -> dtype)
#   - Completion + chat APIs, optional streaming, stop strings, reasoning-tag stripping
# All explanations are embedded as comments to adhere to "zero extraneous text" requirement.

from __future__ import annotations

import os
import inspect
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    pipeline as hf_pipeline,
)
from transformers.generation.utils import GenerateOutput  # type: ignore[import-untyped]

# Optional bitsandbytes (8-bit/4-bit) availability detection
try:
    from transformers import BitsAndBytesConfig  # type: ignore
    _BNB_AVAILABLE = True
except Exception:
    _BNB_AVAILABLE = False


# ------------------------------------------------------------------------------------------
# Public value objects
# ------------------------------------------------------------------------------------------

@dataclass
class CompletionPrediction:
    # Single completion output container
    text: str
    logprobs: Optional[List[float]] = None


@dataclass
class ChatPrediction:
    # Chat response output container
    response: str
    logprobs: Optional[List[float]] = None


class Role(Enum):
    # Roles for chat
    HUMAN = "Human"
    ASSISTANT = "Assistant"
    SYSTEM = "System"


class Message:
    # Simple message element
    def __init__(self, role: Role, content: str):
        self.role = role
        self.content = content


class Dialog(List[Message]):
    # A list of messages representing a conversation
    pass


# ------------------------------------------------------------------------------------------
# Device/dtype environment detection and helpers
# ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _DeviceEnv:
    # Execution environment summary
    backend: str            # "cuda", "rocm", "mps", or "cpu"
    n_devices: int          # Number of accelerators visible
    primary_device: str     # "cuda:0", "mps", or "cpu"
    preferred_dtype: torch.dtype  # bf16/fp16 on GPU, fp16 on MPS, fp32 on CPU
    bf16_supported: bool
    fp16_supported: bool


def _detect_device_env() -> _DeviceEnv:
    # Detect accelerator backend and pick optimal dtype for inference
    backend = "cpu"
    n_devices = 0
    primary_device = "cpu"
    bf16_supported = False
    fp16_supported = False

    if torch.cuda.is_available():
        # ROCm yields torch.version.hip != None
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        n_devices = torch.cuda.device_count()
        primary_device = "cuda:0"
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        fp16_supported = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backend = "mps"
        n_devices = 1
        primary_device = "mps"
        bf16_supported = False
        fp16_supported = True

    preferred_dtype = torch.float32
    if backend in ("cuda", "rocm"):
        preferred_dtype = torch.bfloat16 if bf16_supported else torch.float16
    elif backend == "mps":
        preferred_dtype = torch.float16

    return _DeviceEnv(
        backend=backend,
        n_devices=n_devices,
        primary_device=primary_device,
        preferred_dtype=preferred_dtype,
        bf16_supported=bf16_supported,
        fp16_supported=fp16_supported,
    )


def _gpu_total_memory_bytes(device_index: int) -> Optional[int]:
    # Returns total memory in bytes for GPU device_index. Returns None on failure.
    try:
        props = torch.cuda.get_device_properties(device_index)
        return getattr(props, "total_memory", None)
    except Exception:
        return None


def _estimate_should_quantize_4bit(env: _DeviceEnv) -> bool:
    # Heuristic: prefer 4-bit quantization if any visible GPU has <= 12.5 GiB, unless USE_4BIT=0 is set.
    if env.backend not in ("cuda", "rocm") or env.n_devices == 0:
        return False
    if os.getenv("USE_4BIT", "").strip() == "0":
        return False
    try:
        for i in range(env.n_devices):
            total = _gpu_total_memory_bytes(i) or 0
            if total / (1024**3) <= 12.5:
                return True
    except Exception:
        return False
    return False


def _choose_attn_impl(env: _DeviceEnv) -> Optional[str]:
    # Choose the fastest attention impl supported by the environment. Fallbacks are handled downstream.
    # Note: Some architectures (e.g., gpt-oss in current HF) do NOT support SDPA yet; we will force "eager" for them.
    if env.backend == "cuda":
        try:
            import flash_attn  # type: ignore
            return "flash_attention_2"
        except Exception:
            return "sdpa"
    if env.backend in ("rocm", "mps"):
        return "sdpa"
    return None


def _kwarg_name_for_dtype(func: Callable[..., Any]) -> str:
    # Uses signature introspection to decide whether to pass "dtype" (new) or "torch_dtype" (legacy).
    try:
        params = inspect.signature(func).parameters
        if "dtype" in params:
            return "dtype"
    except Exception:
        pass
    return "torch_dtype"


# ------------------------------------------------------------------------------------------
# Stopping and reasoning helpers
# ------------------------------------------------------------------------------------------

class _StopOnSubstrings(StoppingCriteria):
    # Stops once any of the provided strings appears in the decoded tail window.
    def __init__(self, tokenizer: AutoTokenizer, stop_strings: Iterable[str], max_window: int = 96):
        super().__init__()
        self._tok = tokenizer
        self._stops = [s for s in stop_strings if s]
        self._max_window = max(8, int(max_window))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        ids = input_ids[0].tolist() if input_ids.ndim == 2 else input_ids.tolist()
        tail = ids[-self._max_window :]
        text_tail = self._tok.decode(tail, skip_special_tokens=False)
        return any(s in text_tail for s in self._stops)


def _strip_think_segments(text: str, think_tokens: Tuple[str, str]) -> str:
    # Removes content enclosed by (<think>, </think>) or similar tags to avoid revealing chain-of-thought.
    start_tag, end_tag = think_tokens
    if not start_tag or not end_tag:
        return text
    out: List[str] = []
    i = 0
    while i < len(text):
        j = text.find(start_tag, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = text.find(end_tag, j + len(start_tag))
        if k < 0:
            break
        i = k + len(end_tag)
    return "".join(out)


def _find_reasoning_tags(tokenizer: AutoTokenizer) -> Optional[Tuple[str, str]]:
    # Known common thinking tags among reasoning models; we default to the first variant.
    candidates = [
        ("<|think|>", "<|endofthink|>"),
        ("<think>", "</think>"),
        ("<think_start>", "<think_end>"),
    ]
    return candidates[0] if candidates else None


# ------------------------------------------------------------------------------------------
# Main SOTA generation engine
# ------------------------------------------------------------------------------------------

class TextGenerator:
    # High-performance text generation with HF pipeline and direct generate() for logprobs/streaming.

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device_env: _DeviceEnv,
        use_pipeline: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device_env = device_env
        self._pipe = None

        # Ensure use_cache is enabled and pad_token_id is valid
        if getattr(self.model.config, "use_cache", True) is False:
            self.model.config.use_cache = True
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Build a text-generation pipeline for fast batched inference
        if use_pipeline:
            pipe_kwargs: Dict[str, Any] = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
            }
            # Respect new dtype API if available
            kw_dtype = _kwarg_name_for_dtype(hf_pipeline)
            pipe_kwargs[kw_dtype] = self.device_env.preferred_dtype
            self._pipe = hf_pipeline(**pipe_kwargs)

    # ---------------------------------------------
    # Loading
    # ---------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        trust_remote_code: bool = True,
        use_4bit: Optional[bool] = None,
        use_8bit: Optional[bool] = None,
        compile_model: bool = False,
        use_bettertransformer: bool = True,
        dtype: Optional[torch.dtype] = None,
        max_memory_utilization: float = 0.92,
        revision: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "TextGenerator":
        # Auto-detect environment and decide dtype/attn/quantization/sharding
        env = _detect_device_env()
        chosen_dtype = dtype or env.preferred_dtype

        # Pre-load config to detect architectures that cannot use SDPA (e.g., gpt-oss)
        cfg = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        model_type = str(getattr(cfg, "model_type", ""))  # e.g., "gpt_oss"
        force_eager = model_type in {"gpt_oss", "gpt-oss", "gptoss"}

        # Select attention impl
        attn_impl = "eager" if force_eager else (attn_implementation or _choose_attn_impl(env))

        # Decide quantization (prefer 4-bit on small VRAM). bitsandbytes support on ROCm is not universal; keep CUDA-only to be safe.
        if use_4bit is None:
            use_4bit = _estimate_should_quantize_4bit(env)
        use_8bit = False if use_4bit else bool(use_8bit)

        bnb_config = None
        if _BNB_AVAILABLE and env.backend == "cuda" and (use_4bit or use_8bit):
            compute_dtype = torch.bfloat16 if env.bf16_supported else torch.float16
            bnb_config = BitsAndBytesConfig(  # type: ignore
                load_in_4bit=bool(use_4bit),
                load_in_8bit=bool(use_8bit),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        # Multi-GPU memory cap for accelerate sharding
        max_memory: Optional[Dict[int, str]] = None
        if env.backend in ("cuda", "rocm") and env.n_devices > 1:
            max_memory = {}
            for i in range(env.n_devices):
                total = _gpu_total_memory_bytes(i) or 0
                alloc = int(total * max_memory_utilization)
                max_memory[i] = f"{alloc // (1024**3)}GiB"

        # Assemble model kwargs using new dtype API when available
        model_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
            "use_safetensors": True,
            "attn_implementation": attn_impl,
        }
        if env.backend in ("cuda", "rocm"):
            model_kwargs["device_map"] = "auto"
            if max_memory:
                model_kwargs["max_memory"] = max_memory
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        if revision is not None:
            model_kwargs["revision"] = revision
        if rope_scaling is not None:
            model_kwargs["rope_scaling"] = rope_scaling

        # Respect dtype API evolution
        kw_dtype = _kwarg_name_for_dtype(AutoModelForCausalLM.from_pretrained)
        model_kwargs[kw_dtype] = chosen_dtype

        # Merge user-provided kwargs last
        model_kwargs.update(kwargs)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
            revision=revision,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model with safe fallback for unsupported SDPA/FlashAttention (force eager on failure)
        def _load_model(_kwargs: Dict[str, Any]) -> AutoModelForCausalLM:
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **_kwargs)

        try:
            model = _load_model(model_kwargs)
        except ValueError as e:
            msg = str(e).lower()
            # Fallback to eager attention if backend-selected impl is not supported by the architecture
            if ("scaled_dot_product_attention" in msg or "attn_implementation" in msg or "sdpa" in msg) and model_kwargs.get("attn_implementation") != "eager":
                model_kwargs["attn_implementation"] = "eager"
                model = _load_model(model_kwargs)
            else:
                raise

        # Optional BetterTransformer for speed (skip for 4-bit in some cases)
        if use_bettertransformer and hasattr(model, "to_bettertransformer"):
            try:
                model = model.to_bettertransformer()
            except Exception:
                pass

        # Optional torch.compile to optimize single-device, non-quantized models
        if compile_model and env.n_devices <= 1 and not use_4bit:
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)  # type: ignore[attr-defined]
            except Exception:
                pass

        # CPU math tuning
        if env.backend == "cpu":
            try:
                torch.set_num_threads(max(1, os.cpu_count() or 1))
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        return cls(model=model, tokenizer=tokenizer, device_env=env, use_pipeline=True)

    # ---------------------------------------------
    # Internal utilities
    # ---------------------------------------------
    @staticmethod
    def _default_sampling_params() -> Dict[str, Any]:
        # Balanced creative defaults
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 0,
        }

    def _build_stopping_criteria(
        self,
        stop_strings: Optional[List[str]] = None,
    ) -> Optional[StoppingCriteriaList]:
        # Compose stopping criteria for custom stop strings
        criteria: List[StoppingCriteria] = []
        if stop_strings:
            criteria.append(_StopOnSubstrings(self.tokenizer, stop_strings))
        return StoppingCriteriaList(criteria) if criteria else None

    def _safe_max_new_tokens(self, tokenized_prompts: List[List[int]], requested_max_new_tokens: int) -> int:
        # Prevent prompt+gen from exceeding model's max context
        max_pos = getattr(self.model.config, "max_position_embeddings", None)
        if not max_pos:
            return requested_max_new_tokens
        longest_prompt = max((len(x) for x in tokenized_prompts), default=0)
        allowed = max(1, max_pos - longest_prompt)
        return max(1, min(requested_max_new_tokens, allowed))

    def _ensure_pipe(self) -> None:
        # Lazily initialize pipeline if missing
        if self._pipe is None:
            pipe_kwargs: Dict[str, Any] = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
            }
            kw_dtype = _kwarg_name_for_dtype(hf_pipeline)
            pipe_kwargs[kw_dtype] = self.device_env.preferred_dtype
            self._pipe = hf_pipeline(**pipe_kwargs)

    # ---------------------------------------------
    # Public API: Text completion (batched)
    # ---------------------------------------------
    def complete(
        self,
        prompts: List[str],
        max_gen_len: Optional[int] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        echo: bool = False,
        batch_size: int = 8,
        stop: Optional[List[str]] = None,
        strip_reasoning: bool = True,
    ) -> List[CompletionPrediction]:
        # Fast path via pipeline if logprobs=False; direct generate() if logprobs=True.
        if not prompts:
            return []

        self._ensure_pipe()

        # Build sampling config
        sampling = dict(self._default_sampling_params())
        if sampling_params:
            sampling.update(sampling_params)

        # Tokenize to compute safe max_new_tokens
        tokenized = self.tokenizer(prompts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        req_max_new_tokens = max_gen_len if max_gen_len is not None else 128
        max_new_tokens = self._safe_max_new_tokens(tokenized, req_max_new_tokens)

        stopping_criteria = self._build_stopping_criteria(stop)
        think_tags = _find_reasoning_tags(self.tokenizer) if strip_reasoning else None

        # Pipeline fast path (no token logprobs)
        if not logprobs:
            outputs = self._pipe(
                prompts,
                batch_size=batch_size,
                return_full_text=echo,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
                **sampling,
            )
            # Normalize to List[List[Dict]]
            if isinstance(outputs, dict):
                outputs = [outputs]  # type: ignore[assignment]
            results: List[CompletionPrediction] = []
            for item in outputs:
                if isinstance(item, list) and item and isinstance(item[0], dict):
                    text = item[0].get("generated_text", "")
                elif isinstance(item, dict):
                    text = item.get("generated_text", "")
                else:
                    text = str(item)
                if strip_reasoning and think_tags is not None:
                    text = _strip_think_segments(text, think_tags)
                results.append(CompletionPrediction(text=text, logprobs=None))
            return results

        # Direct generate() path (compute token-level logprobs)
        input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in tokenized]
        max_len = max(len(x) for x in input_ids_list)
        pad_id = self.tokenizer.pad_token_id
        batch_ids = torch.stack(
            [torch.nn.functional.pad(x, (0, max_len - len(x)), value=pad_id) for x in input_ids_list],
            dim=0,
        )
        device = (
            torch.device(self.device_env.primary_device)
            if self.device_env.backend in ("cuda", "rocm", "mps")
            else torch.device("cpu")
        )
        batch_ids = batch_ids.to(device)
        attn = (batch_ids != pad_id).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
        gen_kwargs.update(sampling)
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        with torch.no_grad():
            outputs: GenerateOutput = self.model.generate(
                input_ids=batch_ids,
                attention_mask=attn,
                **gen_kwargs,
            )

        sequences = outputs.sequences
        scores = outputs.scores or []
        results: List[CompletionPrediction] = []

        for i, prompt_ids in enumerate(tokenized):
            seq_ids = sequences[i].tolist()
            gen_ids = seq_ids if echo else seq_ids[len(prompt_ids):]

            token_logprobs: Optional[List[float]] = None
            if scores:
                # Align number of generated steps
                t_gen = min(len(scores), len(gen_ids))
                token_logprobs = []
                for t in range(t_gen):
                    step_scores = scores[t][i]  # [V]
                    logp = torch.log_softmax(step_scores, dim=-1)
                    token_logprobs.append(float(logp[gen_ids[t]].item()))

            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if strip_reasoning and think_tags is not None:
                text = _strip_think_segments(text, think_tags)
            results.append(CompletionPrediction(text=text, logprobs=token_logprobs))
        return results

    # ---------------------------------------------
    # Public API: Chat
    # ---------------------------------------------
    @staticmethod
    def _to_chatml(dialog: Dialog) -> List[Dict[str, str]]:
        # Convert Dialog to HF chat template format
        role_map = {Role.SYSTEM: "system", Role.HUMAN: "user", Role.ASSISTANT: "assistant"}
        return [{"role": role_map.get(m.role, "user"), "content": m.content} for m in dialog]

    def _format_dialog(self, dialog: Dialog) -> str:
        # Prefer tokenizer chat template; fallback to simple role-prefixed format
        chat = self._to_chatml(dialog)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        formatted = ""
        for m in dialog:
            formatted += f"{m.role.value}: {m.content}\n"
        formatted += f"{Role.ASSISTANT.value}: "
        return formatted

    def chat(
        self,
        dialogs: List[Dialog],
        max_gen_len: Optional[int] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        format_func: Optional[Callable[[Dialog], str]] = None,
        batch_size: int = 4,
        stop: Optional[List[str]] = None,
        strip_reasoning: bool = True,
    ) -> List[ChatPrediction]:
        # Chat built on top of completion
        if not dialogs:
            return []
        fmt = format_func or self._format_dialog
        prompts = [fmt(d) for d in dialogs]
        comps = self.complete(
            prompts=prompts,
            max_gen_len=max_gen_len,
            sampling_params=sampling_params,
            logprobs=logprobs,
            echo=False,
            batch_size=batch_size,
            stop=stop,
            strip_reasoning=strip_reasoning,
        )
        return [ChatPrediction(response=c.text, logprobs=c.logprobs) for c in comps]

    # ---------------------------------------------
    # Public API: Batch chat wrapper
    # ---------------------------------------------
    def batch_chat(
        self,
        dialogs: List[Dialog],
        max_gen_len: Optional[int] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        format_func: Optional[Callable[[Dialog], str]] = None,
        batch_size: int = 4,
        stop: Optional[List[str]] = None,
        strip_reasoning: bool = True,
    ) -> List[ChatPrediction]:
        # Delegates to chat(); present for API symmetry
        return self.chat(
            dialogs=dialogs,
            max_gen_len=max_gen_len,
            sampling_params=sampling_params,
            logprobs=logprobs,
            format_func=format_func,
            batch_size=batch_size,
            stop=stop,
            strip_reasoning=strip_reasoning,
        )

    # ---------------------------------------------
    # Public API: Streaming (token-by-token)
    # ---------------------------------------------
    def stream_complete(
        self,
        prompt: str,
        max_gen_len: int = 128,
        sampling_params: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        strip_reasoning: bool = True,
    ) -> Generator[str, None, None]:
        # Stream tokens using TextIteratorStreamer with background thread.
        sampling = dict(self._default_sampling_params())
        if sampling_params:
            sampling.update(sampling_params)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = (
            torch.device(self.device_env.primary_device)
            if self.device_env.backend in ("cuda", "rocm", "mps")
            else torch.device("cpu")
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_pos = getattr(self.model.config, "max_position_embeddings", None)
        req_max_new_tokens = max_gen_len
        if max_pos is not None:
            allowed = max(1, int(max_pos) - int(inputs["input_ids"].shape[-1]))
            req_max_new_tokens = max(1, min(req_max_new_tokens, allowed))

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        stopping_criteria = self._build_stopping_criteria(stop)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=req_max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
            use_cache=True,
        )
        gen_kwargs.update(sampling)
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        def _bg_generate():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        t = threading.Thread(target=_bg_generate, daemon=True)
        t.start()
        for piece in streamer:
            yield piece
        t.join()

    # ---------------------------------------------
    # Convenience: move model to a specific device (single-device scenarios)
    # ---------------------------------------------
    def to(self, device: Union[str, torch.device]) -> "TextGenerator":
        # No-op for sharded models; best-effort for single-device moves
        try:
            self.model.to(device)
        except Exception:
            pass
        return self


# Usage example (kept commented to avoid extraneous output):
# generator = TextGenerator.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# args = {"temperature": 0.7, "top_p": 0.9, "do_sample": True}
# completions = generator.complete(["Once upon a time"], max_gen_len=1024, sampling_params=args)
# dialogs = [Dialog([Message(Role.HUMAN, "Hi!"), Message(Role.HUMAN, "Tell me a joke.")])]
# responses = generator.chat(dialogs, max_gen_len=1024, sampling_params=args)

# print("Completion:", completions[0].text)
# print("Chat Response:", responses[0].response)