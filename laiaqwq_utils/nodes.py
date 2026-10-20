import logging
import time

import torch

import comfy.model_management
import comfy.patcher_extension
from comfy_api.latest import ComfyExtension, io

from . import profiler


class LaiaqwqSamplerProfiler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LaiaqwqSamplerProfiler",
            display_name="Laiaqwq Sampler Profiler",
            category="laiaqwq/utils",
            description="Times every diffusion model call (cuda-synchronized) into the /laiaqwq/profile store.",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")  # bench cache-buster: everything downstream re-executes

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               "laiaqwq_profiler", profiler.diffusion_model_wrapper)
        return io.NodeOutput(m)


class LaiaqwqNcuRange(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LaiaqwqNcuRange",
            display_name="Laiaqwq NCU Range",
            category="laiaqwq/utils",
            description="Brackets one diffusion model call with cudaProfilerStart/Stop for `ncu --profile-from-start off`.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("call_number", default=2, min=0, max=10000,
                             tooltip="1-based DiT call to profile; 0 brackets every call"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")  # must re-execute every run

    @classmethod
    def execute(cls, model, call_number=2) -> io.NodeOutput:
        cudart = torch.cuda.cudart()
        counter = {"n": 0}

        def wrapper(executor, *args, **kwargs):
            counter["n"] += 1
            active = call_number == 0 or counter["n"] == call_number
            if active:
                cudart.cudaProfilerStart()
            try:
                return executor(*args, **kwargs)
            finally:
                if active:
                    torch.cuda.synchronize()
                    cudart.cudaProfilerStop()

        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               "laiaqwq_ncu_range", wrapper)
        return io.NodeOutput(m)


class LaiaqwqTorchProfile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LaiaqwqTorchProfile",
            display_name="Laiaqwq Torch Profile",
            category="laiaqwq/utils",
            description="torch.profiler around one diffusion model call; writes a chrome trace and a key_averages table.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("call_number", default=2, min=0, max=10000,
                             tooltip="1-based DiT call to profile; 0 profiles every call"),
                io.String.Input("out_prefix", default="/tmp/laiaqwq_trace",
                                tooltip="output prefix; writes <prefix>_call<n>.json.gz and .table.txt"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")  # must re-execute every run

    @classmethod
    def execute(cls, model, call_number=2, out_prefix="/tmp/laiaqwq_trace") -> io.NodeOutput:
        counter = {"n": 0}

        def wrapper(executor, *args, **kwargs):
            counter["n"] += 1
            n = counter["n"]
            if not (call_number == 0 or n == call_number):
                return executor(*args, **kwargs)
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA])
            with prof:
                out = executor(*args, **kwargs)
                torch.cuda.synchronize()
            trace_path = "{}_call{}.json.gz".format(out_prefix, n)
            table_path = "{}_call{}.table.txt".format(out_prefix, n)
            prof.export_chrome_trace(trace_path)
            with open(table_path, "w") as f:
                f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=200))
            logging.info("[laiaqwq.torch_profile] call %d -> %s , %s", n, trace_path, table_path)
            return out

        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                               "laiaqwq_torch_profile", wrapper)
        return io.NodeOutput(m)


class LaiaqwqVRAMSnapshot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LaiaqwqVRAMSnapshot",
            display_name="Laiaqwq VRAM Snapshot",
            category="laiaqwq/utils",
            inputs=[
                io.AnyType.Input("value"),
                io.String.Input("label", default=""),
            ],
            outputs=[io.AnyType.Output(), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, value, label="") -> io.NodeOutput:
        device = comfy.model_management.get_torch_device()
        gib = 1024.0 ** 3
        report = ("[{}] vram allocated {:.2f} GiB (peak {:.2f}), reserved {:.2f} GiB (peak {:.2f})".format(
            label,
            torch.cuda.memory_allocated(device) / gib, torch.cuda.max_memory_allocated(device) / gib,
            torch.cuda.memory_reserved(device) / gib, torch.cuda.max_memory_reserved(device) / gib))
        logging.info(report)
        return io.NodeOutput(value, report)


LaiaqwqTimer = io.Custom("LAIAQWQ_TIMER")
# shared anchor for LaiaqwqTimerMark when no timer link is wired
_implicit_timer = {"label": "", "t0": None}


class LaiaqwqTimerStart(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LaiaqwqTimerStart",
            display_name="Laiaqwq Timer Start",
            category="laiaqwq/utils",
            description="Starts a timer and forwards `value` unchanged. Wire `dep` only to force execution order.",
            inputs=[
                io.AnyType.Input("value"),
                io.String.Input("label", default=""),
                io.AnyType.Input("dep", optional=True),
                io.Boolean.Input("log", default=True),
            ],
            outputs=[
                io.AnyType.Output(),
                LaiaqwqTimer.Output(display_name="timer"),
                io.Float.Output(display_name="start_ts"),
                io.AnyType.Output(display_name="dep"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")  # timing nodes must re-execute every run

    @classmethod
    def execute(cls, value, label="", log=True, dep=None) -> io.NodeOutput:
        timer = {"label": label, "t0": time.perf_counter()}
        _implicit_timer.update(timer)
        if log:
            logging.info("[laiaqwq.timer] %s start", label or "timer")
        return io.NodeOutput(value, timer, time.time(), dep if dep is not None else value)


class LaiaqwqTimerMark(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LaiaqwqTimerMark",
            display_name="Laiaqwq Timer Mark",
            category="laiaqwq/utils",
            description="Reports ms elapsed since the timer's anchor (start or last reset). Enable `reset` to re-anchor for lap times.",
            inputs=[
                io.AnyType.Input("value"),
                LaiaqwqTimer.Input("timer", optional=True),
                io.String.Input("label", default=""),
                io.AnyType.Input("dep", optional=True),
                io.Boolean.Input("log", default=True),
                io.Boolean.Input("reset", default=False),
            ],
            outputs=[
                io.AnyType.Output(),
                LaiaqwqTimer.Output(display_name="timer"),
                io.Float.Output(display_name="elapsed_ms"),
                io.AnyType.Output(display_name="dep"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, value, timer=None, label="", log=True, reset=False, dep=None) -> io.NodeOutput:
        now = time.perf_counter()
        t = timer if timer is not None else _implicit_timer
        if t["t0"] is None:
            t["t0"] = now
        elapsed_ms = (now - t["t0"]) * 1000.0
        if reset:
            t["t0"] = now
        if log:
            logging.info("[laiaqwq.timer] %s +%.3f ms", label or t["label"] or "timer", elapsed_ms)
        return io.NodeOutput(value, t, elapsed_ms, dep if dep is not None else value)


class LaiaqwqDepMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LaiaqwqDepMerge",
            display_name="Laiaqwq Dep Merge",
            category="laiaqwq/utils",
            description="Fan-in for ordering deps: connect any outputs just to make downstream wait on all of them. Forwards the first connected dep.",
            inputs=[
                io.Autogrow.Input("deps", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.AnyType.Input("dep"),
                        prefix="dep_", min=0, max=20)),
            ],
            outputs=[io.AnyType.Output(display_name="dep")],
        )

    @classmethod
    def execute(cls, deps=None) -> io.NodeOutput:
        value = None
        if deps:
            for name in sorted(deps):
                if deps[name] is not None:
                    value = deps[name]
                    break
        return io.NodeOutput(value)


class LaiaqwqUtilsExtension(ComfyExtension):
    async def get_node_list(self):
        return [LaiaqwqSamplerProfiler, LaiaqwqNcuRange, LaiaqwqTorchProfile, LaiaqwqVRAMSnapshot, LaiaqwqTimerStart, LaiaqwqTimerMark, LaiaqwqDepMerge]


async def comfy_entrypoint() -> LaiaqwqUtilsExtension:
    return LaiaqwqUtilsExtension()
