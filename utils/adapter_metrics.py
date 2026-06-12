"""Computes adapter-strength metrics for a LoKr epoch checkpoint.

A LoKr checkpoint stores the adapter delta itself, not merged weights, so every
metric is derived from the checkpoint alone — no base model needed.

Per LoKr module: delta_W = scale * kron(W1, W2). The Frobenius norm of a
Kronecker product factorizes exactly:

    ||delta_W||_F = scale * ||W1||_F * ||W2||_F

so the full delta is never materialized. Effective scale mirrors the
reconstruction in ai-toolkit/toolkit/lycoris_utils.py: the alpha/dim factor only
applies when a low-rank factor (w1_b or w2_b) exists; full-matrix LoKr keeps
scale = 1.0 regardless of the stored alpha sentinel.
"""

import torch
from safetensors import safe_open

LOKR_SUFFIXES = (
    "lokr_w1", "lokr_w1_a", "lokr_w1_b",
    "lokr_w2", "lokr_w2_a", "lokr_w2_b",
    "lokr_t1", "lokr_t2", "alpha",
)


def compute_adapter_metrics(checkpoint_path):
    modules = _group_modules(checkpoint_path)

    sum_sq = 0.0
    max_weight = 0.0
    param_count = 0
    scales = []

    for parts in modules.values():
        if "lokr_w1" not in parts and "lokr_w1_a" not in parts:
            continue

        w1 = _rebuild_w1(parts)
        w2 = _rebuild_w2(parts)
        scale = _effective_scale(parts)

        module_norm = scale * torch.linalg.norm(w1) * torch.linalg.norm(w2)
        sum_sq += float(module_norm) ** 2
        max_weight = max(max_weight, float(w2.abs().max()))
        scales.append(scale)

        for suffix, tensor in parts.items():
            if suffix != "alpha":
                param_count += tensor.numel()

    return {
        "delta_norm": sum_sq ** 0.5,
        "max_weight": max_weight,
        "effective_scale": max(scales) if scales else 1.0,
        "param_count": param_count,
    }


def _group_modules(checkpoint_path):
    modules = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            for suffix in LOKR_SUFFIXES:
                if key.endswith("." + suffix):
                    name = key[: -(len(suffix) + 1)]
                    modules.setdefault(name, {})[suffix] = handle.get_tensor(key).float()
                    break
    return modules


def _rebuild_w1(parts):
    if "lokr_w1" in parts:
        return parts["lokr_w1"]
    return parts["lokr_w1_a"] @ parts["lokr_w1_b"]


def _rebuild_w2(parts):
    if "lokr_w2" in parts:
        return parts["lokr_w2"]
    if "lokr_t2" in parts:
        return torch.einsum("i j k l, i p, j r -> p r k l", parts["lokr_t2"], parts["lokr_w2_a"], parts["lokr_w2_b"])
    return parts["lokr_w2_a"] @ parts["lokr_w2_b"]


def _effective_scale(parts):
    alpha = parts.get("alpha")
    w1b = parts.get("lokr_w1_b")
    w2b = parts.get("lokr_w2_b")
    if alpha is None or (w1b is None and w2b is None):
        return 1.0
    denom = w1b.shape[0] if w1b is not None else w2b.shape[0]
    return float(alpha.flatten()[0]) / denom
