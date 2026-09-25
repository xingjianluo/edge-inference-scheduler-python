"""本地设备能力探测：仅依赖本机信息，不访问外部服务。

探测规则：
- 始终存在一个 CPU 设备，能力来自 os 模块（核数、可用内存）。
- 可选的模拟加速设备通过环境变量声明，便于本地复现异构场景：
    EDGE_SIM_DEVICES='[{"device_id":"npu0","kind":"npu",
                        "supported_quantizations":["int8","fp16"],
                        "supported_operators":["conv","matmul","softmax"],
                        "total_memory_mb":512,"max_concurrency":2}]'
  未设置时只有 CPU 设备，保证任何机器上结果一致。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# CPU 基线算子集：本仓库模拟运行时假定 CPU 支持这些通用算子
_CPU_OPERATORS = frozenset({"conv", "matmul", "softmax", "relu", "add", "layernorm"})
_CPU_QUANTIZATIONS = frozenset({"fp32", "fp16"})


@dataclass(frozen=True)
class DeviceCapability:
    """设备硬性能力描述。"""

    device_id: str
    kind: str  # "cpu" / "gpu" / "npu"（本地模拟）
    supported_quantizations: frozenset[str] = field(default_factory=frozenset)
    supported_operators: frozenset[str] = field(default_factory=frozenset)
    total_memory_mb: int = 0
    max_concurrency: int = 1


def _available_memory_mb() -> int:
    """读取本机可用内存（MB），失败时回退到保守值。"""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size // (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return 1024


def probe_local_devices() -> list[DeviceCapability]:
    """探测本地设备：CPU + 环境变量声明的模拟加速设备。"""
    cpu = DeviceCapability(
        device_id="cpu0",
        kind="cpu",
        supported_quantizations=_CPU_QUANTIZATIONS,
        supported_operators=_CPU_OPERATORS,
        total_memory_mb=_available_memory_mb(),
        max_concurrency=max(1, (os.cpu_count() or 1)),
    )
    devices = [cpu]

    raw = os.environ.get("EDGE_SIM_DEVICES", "").strip()
    if raw:
        for spec in json.loads(raw):
            devices.append(
                DeviceCapability(
                    device_id=spec["device_id"],
                    kind=spec.get("kind", "npu"),
                    supported_quantizations=frozenset(spec.get("supported_quantizations", [])),
                    supported_operators=frozenset(spec.get("supported_operators", [])),
                    total_memory_mb=int(spec.get("total_memory_mb", 0)),
                    max_concurrency=int(spec.get("max_concurrency", 1)),
                )
            )
    return devices
