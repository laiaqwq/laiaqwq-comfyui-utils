"""aiohttp routes on the running PromptServer: VRAM snapshot + profiler readout."""

import torch
from aiohttp import web

import comfy.model_management
import server

from . import profiler

routes = server.PromptServer.instance.routes


def vram_payload():
    device = comfy.model_management.get_torch_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "max_allocated": torch.cuda.max_memory_allocated(device),
        "max_reserved": torch.cuda.max_memory_reserved(device),
        "mem_get_info_free": free,
        "mem_get_info_total": total,
    }


@routes.get("/laiaqwq/vram")
async def laiaqwq_vram(request):
    return web.json_response(vram_payload())


@routes.post("/laiaqwq/vram/reset")
async def laiaqwq_vram_reset(request):
    device = comfy.model_management.get_torch_device()
    torch.cuda.reset_peak_memory_stats(device)
    return web.json_response(vram_payload())


@routes.get("/laiaqwq/profile")
async def laiaqwq_profile(request):
    return web.json_response(profiler.report())


@routes.post("/laiaqwq/profile/reset")
async def laiaqwq_profile_reset(request):
    profiler.reset()
    return web.json_response(profiler.report())
