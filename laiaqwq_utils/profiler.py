"""DIFFUSION_MODEL call profiler. One store per process; reset via /laiaqwq/profile/reset."""

import time

import torch

STATE = {"calls": []}


def reset():
    STATE["calls"].clear()


def diffusion_model_wrapper(executor, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = executor(*args, **kwargs)
    torch.cuda.synchronize()
    STATE["calls"].append({"dt": time.perf_counter() - t0,
                           "peak_alloc": torch.cuda.max_memory_allocated()})
    return out


def report():
    calls = STATE["calls"]
    dts = [c["dt"] for c in calls]
    return {
        "n_calls": len(dts),
        "total_s": sum(dts),
        "mean_s": sum(dts) / len(dts) if dts else 0.0,
        "min_s": min(dts) if dts else 0.0,
        "max_s": max(dts) if dts else 0.0,
        "peak_alloc_bytes": max((c["peak_alloc"] for c in calls), default=0),
        "calls": dts,
    }
