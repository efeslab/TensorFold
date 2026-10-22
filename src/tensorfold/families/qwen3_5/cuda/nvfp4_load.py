"""ModelOpt and compressed-tensors checkpoints of Qwen3.8-27B into ``Weights``: each projection by its tensors' scheme."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorfold.cuda import prompt_precision

from .weights import Plain

SUFFIXES = ("weight", "weight_packed", "weight_scale", "weight_scale_2", "weight_global_scale", "input_scale",
            "input_global_scale")


def quantized(model_dir: Path) -> bool:
    """Whether config.json names ModelOpt or compressed-tensors weights (NVFP4, FP8 or bf16 per projection)."""

    from tensorfold.cuda.nvfp4.format import is_quantized

    return is_quantized(model_dir)


def skipped(name: str) -> bool:
    """Tensors the text model never reads: the vision tower and the MTP head."""

    return (name.startswith(("model.visual.", "visual.", "vision_tower", "mtp.")) or ".visual." in name
            or ".mtp." in name)


@dataclass
class Plain8(Plain):
    """A bf16 projection (the GDN gates) with an e4m3 copy made at load for --prefill-fp8's prompts."""

    rows8: object = None      # tensorfold.cuda.nvfp4.linear.Fp8Linear

    def prefill8(self, xq):
        return self.rows8.prefill8(xq)

    def nbytes(self) -> int:
        return super().nbytes() + self.rows8.nbytes()


def weight_bytes(name: str, info: dict) -> tuple[int, int]:
    """A tensor as loaded: stored bytes with outputs padded to 128, gates' e4m3 copies, A_log and dt_bias in fp32."""

    if skipped(name):
        return 0, 0
    shape, dtype = list(info["shape"]), info["dtype"]
    if name.endswith((".A_log", ".dt_bias")):
        return math.prod(shape) * 4, 0
    from tensorfold.cuda.capacity import SIZES

    if len(shape) == 2 and dtype in ("U8", "F8_E4M3") and not name.endswith("_scale"):
        shape[0] = -(-shape[0] // 128) * 128
    amount = math.prod(shape) * SIZES[dtype]
    if name.endswith(("in_proj_a.weight", "in_proj_b.weight")) and len(shape) == 2 and prompt_precision.fp8():
        npad = -(-shape[0] // 128) * 128
        amount += npad * shape[1] + shape[1] // 64 * npad * 2
    return amount, 0


def admission(geometry):
    """The MLX path's geometry (FP8 prompts: plus the widest NVFP4 projection's e4m3 staging) and tensor bytes."""

    from tensorfold.cuda.geometry import with_fixed

    def with_staging(text):
        d, i = int(text["hidden_size"]), int(text["intermediate_size"])
        return with_fixed(geometry(text), d * i + d * i // 32 + (4 << 20)) if prompt_precision.fp8() else geometry(text)

    return with_staging, weight_bytes


FULL_LINE = "full (bf16 activations, the stored weights exactly)"


def maths() -> tuple[dict[str, bool], str]:
    """Each format's math on this GPU under --precision, and the startup line naming it; no supported GPU is refused."""

    from tensorfold.cuda import precision

    if precision.mode() != precision.CHECKPOINT or not torch.cuda.is_available():
        return {"nvfp4": False, "fp8": False}, FULL_LINE
    major, minor = torch.cuda.get_device_capability()
    own = precision.own_math((major, minor))
    if all(own.values()):
        return own, ("checkpoint (the checkpoint's own math: its NVFP4 layers FP4 x FP4, per-16 scales under its "
                     "static input scales; its FP8 layers FP8 x FP8)")
    gpu = f"{torch.cuda.get_device_name()}, SM {major}.{minor}"
    if own["fp8"]:
        return own, (f"checkpoint where this GPU has it ({gpu}): its FP8 layers FP8 x FP8 under their static input "
                     "scales; its NVFP4 layers W4A16 (bf16 activations: the block-scaled FP4 mma is SM 12.x's)")
    return own, f"full ({gpu} has neither the block-scaled FP4 nor the FP8 mma): bf16 activations, the stored weights"


def load_nvfp4(model_dir: str | Path, device: str = "cuda"):
    """NVFP4 and FP8 projections at full precision or in the checkpoint's own math, bf16 ones as stored."""

    from tensorfold.cuda.capacity import headers
    from tensorfold.cuda.nvfp4 import format as fmt
    from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear, Staging

    from .weights import GDN, Attention, Config, Layer, Weights, _Tensors

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    block = fmt.config_block(json.loads((model_dir / "config.json").read_text())) or {}
    reciprocal = str(block.get("quant_method", "")).lower() == "compressed-tensors"
    info = {n: (i["dtype"], i["shape"]) for n, i in headers(model_dir).items() if not skipped(n)}
    root = "model.language_model." if any(n.startswith("model.language_model.") for n in info) else "model."
    t = _Tensors(model_dir, device, skip=skipped)
    staging = Staging()                                   # one e4m3 copy at a time, shared by every NVFP4 projection
    own, line = maths()

    def act_scale(name: str, got: dict, kind: str) -> float | None:
        """The checkpoint's static input scale (x = codes * scale), where ``kind`` runs its own math; else None."""

        if not own[kind]:
            return None
        if "input_scale" in got:
            return float(got["input_scale"].float().reshape(-1)[0])
        if "input_global_scale" in got:
            return 1.0 / float(got["input_global_scale"].float().reshape(-1)[0])
        raise ValueError(f"{name}: no static input scale, so its own math is unknown; serve it with --precision full")

    def linear(name: str, prompt: bool = True):
        parts = {s: info[f"{name}.{s}"] for s in SUFFIXES if f"{name}.{s}" in info}
        if not parts:
            raise ValueError(f"the checkpoint has no projection {name}")
        kind = fmt.scheme(parts)
        got = {s: t.pop(f"{name}.{s}") for s in parts}
        weight = got["weight"] if "weight" in got else got.get("weight_packed")
        if kind == "nvfp4":
            g = got.get("weight_global_scale" if reciprocal else "weight_scale_2")
            if g is None:
                raise ValueError(f"{name}: an NVFP4 weight without its global scale")
            g = float(g.float().reshape(-1)[0])
            lin = Fp4Linear.from_checkpoint(weight, got["weight_scale"], 1.0 / g if reciprocal else g,
                                            act=act_scale(name, got, "nvfp4"))
            lin.staging = staging
            return lin
        if kind == "fp8":
            s = got["weight_scale"].float().reshape(-1)
            if s.numel() != 1:
                raise ValueError(f"{name}: FP8 with {s.numel()} scales; the CUDA engine reads one scale a tensor")
            act = act_scale(name, got, "fp8")
            if REQUANT_FP8 and act is not None and prompt:
                lin = requant_nvfp4(weight, float(s[0]), act, Fp4Linear)
                lin.staging = staging
                return lin
            return Fp8Linear.from_checkpoint(weight, float(s[0]), act=act)
        if kind == "bf16":
            w = weight.to(torch.bfloat16).contiguous()
            return Plain8(w, rows8=Fp8Linear.from_bf16(w)) if prompt and prompt_precision.fp8() else Plain(w)
        raise ValueError(f"{name}: {kind} projections are not read on Qwen3.8-27B yet")

    def get(name: str) -> torch.Tensor:
        return t.pop(root + name).contiguous()

    def norm(name: str) -> torch.Tensor:
        """A centred RMSNorm weight (stored as gamma - 1) with its 1 back, as the MLX converter wrote it."""

        w = get(name)
        return (w.float() + 1.0).to(w.dtype)

    layers = []
    for i in range(cfg.layers):
        p = f"layers.{i}."
        q = root + p
        gdn = attn = None
        if cfg.is_linear(i):
            gdn = GDN(qkv=linear(q + "linear_attn.in_proj_qkv"), z=linear(q + "linear_attn.in_proj_z"),
                      b=linear(q + "linear_attn.in_proj_b"), a=linear(q + "linear_attn.in_proj_a"),
                      out=linear(q + "linear_attn.out_proj"),
                      conv=get(p + "linear_attn.conv1d.weight").reshape(-1, cfg.conv_kernel).contiguous(),
                      A_log=get(p + "linear_attn.A_log").float().contiguous(),
                      dt_bias=get(p + "linear_attn.dt_bias").float().contiguous(),
                      norm=get(p + "linear_attn.norm.weight"))
        else:
            attn = Attention(q=linear(q + "self_attn.q_proj"), k=linear(q + "self_attn.k_proj"),
                             v=linear(q + "self_attn.v_proj"), o=linear(q + "self_attn.o_proj"),
                             q_norm=norm(p + "self_attn.q_norm.weight"), k_norm=norm(p + "self_attn.k_norm.weight"))
        layers.append(Layer(linear=cfg.is_linear(i), input_norm=norm(p + "input_layernorm.weight"),
                            post_norm=norm(p + "post_attention_layernorm.weight"), gdn=gdn, attn=attn,
                            gate=linear(q + "mlp.gate_proj"), up=linear(q + "mlp.up_proj"),
                            down=linear(q + "mlp.down_proj")))
    if not any(n.startswith("lm_head.") for n in info):
        raise ValueError("this checkpoint ties its head to the embedding; the CUDA engine reads a separate lm_head")
    fuse_gate_up(layers)
    w = Weights(config=cfg, embed=Plain(get("embed_tokens.weight").to(torch.bfloat16)), layers=layers,
                norm=norm("norm.weight"), head=linear("lm_head", prompt=False), quant="nvfp4")
    w.precision = "checkpoint" if any(own.values()) else "full"
    w.own = own
    print(f"[tensorfold] precision: {line}", flush=True)
    half = cfg.rope_dims // 2
    inv = cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)
    w.inv_freq = inv.to(torch.float32).to(device)
    left = list(t)
    t.close()
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]} ...")
    torch.cuda.empty_cache()
    return w


# perf-study, OPT-IN (changes the model's numbers): TENSORFOLD_REQUANT_FP8=1 requantizes the checkpoint's per-tensor
# FP8 projections (attention q/k/v/o, DeltaNet in_proj_qkv/z and out_proj: ~7 GB a decode round) to NVFP4 at load, so
# they read half the bytes on the same FP4 lane matmul the checkpoint's own NVFP4 layers use.
REQUANT_FP8 = __import__("os").environ.get("TENSORFOLD_REQUANT_FP8", "0") == "1"
_E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)          # midpoints of 0, .5, 1, 1.5, 2, 3, 4, 6


def requant_nvfp4(weight: torch.Tensor, scale: float, act: float, cls):
    """A ModelOpt per-tensor FP8 projection as NVFP4: e2m1 codes, e4m3 scales per 16 inputs, one global scale.

    The weight is dequantized (codes * scale), its global scale set from its amax (amax / (6 * 448)), each 16-input
    block scaled by its own amax / 6 rounded to e4m3, and the codes rounded to the nearest e2m1 value.  The static
    input scale follows the same amax: FP8's amax / 448 becomes NVFP4's amax / (6 * 448), i.e. ``act / 6``.
    """

    w = weight.view(torch.float8_e4m3fn).float() * scale
    n, k = w.shape
    g = float(w.abs().max()) / (6.0 * 448.0) or 1.0
    blocks = w.view(n, k // 16, 16)
    bscale = (blocks.abs().amax(-1) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn)
    div = bscale.float() * g
    q = blocks / torch.where(div > 0, div, torch.ones_like(div))[..., None]
    mag = torch.bucketize(q.abs().clamp(max=6.0), torch.tensor(_E2M1_MID, device=w.device))
    codes = (mag | ((q < 0) & (mag > 0)).to(mag.dtype) << 3).to(torch.uint8).view(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return cls.from_checkpoint(packed, bscale.view(n, k // 16), g, act=act / 6.0)


def fuse_gate_up(layers) -> int:
    """perf-study: one decode launch for a layer's gate and up projections.

    Where both are checkpoint-math NVFP4 projections under the same input scale and the same weight scale (ModelOpt
    ties them: true for every layer of nvidia/Qwen3.8-27B-NVFP4), their packed words and block scales are stacked into
    one ``[gate | up]`` projection and ``gate`` / ``up`` become views into it (no extra memory; prompts keep using them
    apart).  A lane matmul computes each output column alone, so the stacked launch gives gate's and up's columns the
    bits their own launches give, and the 48-SM GB10 streams one 34,816-column launch faster than two 17,408-column
    ones (decode MLP 29.8 -> 28.6 ms a forward).  TENSORFOLD_FUSE_GATE_UP=0 keeps the two launches.
    """

    import copy
    import os

    if os.environ.get("TENSORFOLD_FUSE_GATE_UP", "1") == "0":
        return 0
    fused = 0
    for layer in layers:
        g, u = layer.gate, layer.up
        if not all(hasattr(x, "words") and hasattr(x, "bs") and getattr(x, "act", None) is not None for x in (g, u)):
            continue
        if (float(g.act), float(g.scale)) != (float(u.act), float(u.scale)) or g.k != u.k \
                or g.n % 64 or g.words.shape[0] * 64 != g.n:
            continue
        f = copy.copy(g)
        f.words = torch.cat([g.words, u.words])
        f.bs = torch.cat([g.bs, u.bs])
        f.n = g.n + u.n
        tiles = g.words.shape[0]
        g.words, u.words = f.words[:tiles], f.words[tiles:]
        g.bs, u.bs = f.bs[:tiles], f.bs[tiles:]
        layer.gate_up = f
        fused += 1
    if fused:
        print(f"[tensorfold] perf-study: gate|up stacked into one decode projection in {fused} layers", flush=True)
    return fused

