from .laiaqwq_utils.nodes import comfy_entrypoint
from .laiaqwq_utils import routes  # noqa: F401  registers /laiaqwq/* endpoints on PromptServer.instance

__all__ = ["comfy_entrypoint"]
