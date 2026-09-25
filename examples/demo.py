"""命令行演示：四类场景的输入、判定依据与归一化结果。

运行：uv run python examples/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge_scheduler.devices import DevicePool, LocalProbe
from edge_scheduler.errors import ErrorKind, SchedulerError
from edge_scheduler.logging_utils import build_logger
from edge_scheduler.runtime import FailureSpec, InferenceInput, SimulatedRuntime
from edge_scheduler.scheduler import EdgeScheduler, SchedulerConfig
from edge_scheduler.variants import FallbackChain, ModelVariant, VariantRegistry


def build():
    reg = VariantRegistry()
    reg.register(ModelVariant("llm", "int8", frozenset({"int8"}),
                              frozenset({"matmul", "rope"}), 400))
    reg.register(ModelVariant("llm", "fp16", frozenset({"fp16"}),
                              frozenset({"matmul", "flash_attention"}), 1500))
    reg.register(ModelVariant("llm", "fp32", frozenset({"fp32"}),
                              frozenset({"matmul"}), 200))
    reg.register_chain(FallbackChain("llm", ("int8", "fp16", "fp32")))
    # 任何设备都不满足的变体：精度 int4 + 专有算子 ghost_op
    reg.register(ModelVariant("llm", "int4-x", frozenset({"int4"}),
                              frozenset({"ghost_op"}), 10))
    logger = build_logger()
    sch = EdgeScheduler(
        reg,
        DevicePool(LocalProbe().detect()),
        SimulatedRuntime(),
        SchedulerConfig(queue_capacity=4, admission_timeout=2.0),
        logger=logger.info,
    )
    return sch, logger


def main() -> None:
    sch, logger = build()

    # 1) 能力缺失：严格模式指定不存在兼容设备的变体
    try:
        sch.infer("llm", "int4-x", InferenceInput("demo-cap", "p"))
    except SchedulerError as e:
        logger.info("[场景1 能力缺失] => %s, 尝试数=%d",
                    e.kind.value, len(e.details.get("attempts", [])))

    # 2) 变体不兼容 + 严格模式：int8 在非 npu 设备上不匹配，但 npu 可承载
    r = sch.infer("llm", "int8", InferenceInput("demo-match", "p"))
    logger.info("[场景2 严格匹配] => %s@%s", r.variant_id, r.device_id)

    # 3) 推理中途失败：int8 首次超时，显式后备链降级到 fp16
    r = sch.infer(
        "llm", "int8", InferenceInput("demo-fail", "p"),
        failure=FailureSpec(ErrorKind.INFERENCE_TIMEOUT, frozenset({1})),
        use_chain=True,
    )
    logger.info("[场景3 失败降级] => %s@%s attempt=%d 副作用=%d",
                r.variant_id, r.device_id, r.attempt, r.side_effects)


if __name__ == "__main__":
    main()
