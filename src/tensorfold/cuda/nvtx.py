"""Opt-in NVTX ranges and per-iteration records for profiling the CUDA engine.

Off by default: with ``TENSORFOLD_NVTX`` unset, ``span`` returns one shared no-op
context and ``record`` returns at once, so the serving path builds no range
objects and formats no strings. Ranges are host-side, behavior-neutral and add
no synchronization; names are fixed (``tensorfold.<area>.<phase>``) and never
carry a request, round or token id.

``TENSORFOLD_ITER_RECORDS=<path>`` (with ``TENSORFOLD_NVTX=1``) also appends one
JSON line per prefill chunk and decode round: monotonic start/end nanoseconds
and the shapes a kernel model needs (rows, committed length, tree parents).
"""

from __future__ import annotations

import contextlib
import json
import os
import time

ENABLED = os.environ.get("TENSORFOLD_NVTX") == "1"
_OFF = contextlib.nullcontext()
_RECORDS = os.environ.get("TENSORFOLD_ITER_RECORDS") if ENABLED else None

if ENABLED:
    import torch

    @contextlib.contextmanager
    def _range(name: str):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def span(name: str):
        return _range(name)
else:
    def span(name: str):
        return _OFF


def now() -> int:
    return time.monotonic_ns() if _RECORDS else 0


_stream = None


def record(kind: str, start_ns: int, **shape) -> None:
    global _stream
    if not _RECORDS:
        return
    end_ns = time.monotonic_ns()
    if _stream is None:
        _stream = open(_RECORDS, "a", buffering=1)      # line-buffered: survives a killed server
    _stream.write(json.dumps({"kind": kind, "start_ns": start_ns, "end_ns": end_ns, **shape}) + "\n")
