"""归一化错误类型：所有调度与推理失败都被归类为可区分的错误码。"""

from __future__ import annotations

import enum


class ErrorCode(enum.Enum):
    """可区分的失败原因。"""

    CAPABILITY_MISSING = "capability_missing"        # 没有任何设备满足硬性能力约束
    VARIANT_INCOMPATIBLE = "variant_incompatible"    # 变体与目标设备硬性约束不匹配
    RESOURCE_EXHAUSTED = "resource_exhausted"        # 队列有位置但等待超时仍无槽位/内存，确定性拒绝
    QUEUE_FULL = "queue_full"                        # 等待队列已满，立即拒绝（背压）
    REQUEST_CANCELLED = "request_cancelled"          # 等待中的请求被取消，不再占用资源也不产出结果
    OPERATOR_UNSUPPORTED = "operator_unsupported"    # 运行时不支持所需算子
    INFERENCE_TIMEOUT = "inference_timeout"          # 单次调用超时
    DEVICE_UNAVAILABLE = "device_unavailable"        # 设备中途不可用
    RUNTIME_FAILURE = "runtime_failure"              # 其他运行时失败


class SchedulerError(Exception):
    """携带归一化错误码与可解释详情的异常。"""

    def __init__(self, code: ErrorCode, message: str, *, detail: dict | None = None):
        super().__init__(message)
        self.code = code
        self.detail = detail or {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"SchedulerError({self.code.value}: {self})"
