"""归一化错误类型与可区分的拒绝原因。"""
from __future__ import annotations

from enum import Enum


class ErrorKind(str, Enum):
    """调度/推理链路中所有错误的归一化分类。"""

    CAPABILITY_MISSING = "capability_missing"          # 设备能力不满足硬性约束
    VARIANT_INCOMPATIBLE = "variant_incompatible"      # 没有任何设备能承载该变体
    INSUFFICIENT_MEMORY = "insufficient_memory"        # 内存/显存不足
    QUEUE_FULL = "queue_full"                          # 排队队列已满
    UNSUPPORTED_OP = "unsupported_op"                  # 运行时不支持算子
    INFERENCE_TIMEOUT = "inference_timeout"           # 单次调用超时
    DEVICE_UNAVAILABLE = "device_unavailable"         # 设备中途不可用
    FALLBACK_EXHAUSTED = "fallback_exhausted"         # 所有后备路径均失败
    DUPLICATE_REQUEST = "duplicate_request"           # 重复请求（副作用防护）


class SchedulerError(Exception):
    """携带归一化错误类型、判定依据与上下文的异常。"""

    def __init__(self, kind: ErrorKind, reason: str, *, details: dict | None = None):
        self.kind = kind
        self.reason = reason
        self.details = details or {}
        super().__init__(f"[{kind.value}] {reason}")
