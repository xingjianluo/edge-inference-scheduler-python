"""本地演示：探测设备 -> 注册变体 -> 提交请求 -> 打印调度决策。

运行：uv run python main.py
可用 EDGE_SIM_DEVICES 声明模拟加速设备（见 README）。
"""

import logging

from edge_scheduler import (
    Device,
    ModelVariant,
    Scheduler,
    SchedulerError,
    SimulatedRuntime,
    VariantRegistry,
    probe_local_devices,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("demo")


def main() -> None:
    caps = probe_local_devices()
    for c in caps:
        log.info("probed device: %s", c)

    registry = VariantRegistry()
    registry.register(ModelVariant(
        "demo-int8", "demo", "int8", frozenset({"conv"}), memory_mb=64,
        fallback_variant_ids=("demo-fp32",),
    ))
    registry.register(ModelVariant(
        "demo-fp32", "demo", "fp32", frozenset({"conv"}), memory_mb=128,
    ))

    scheduler = Scheduler(
        [Device(c) for c in caps], registry, SimulatedRuntime(),
    )

    try:
        result = scheduler.submit("demo-req-1", "demo-int8", "hello-edge")
        log.info("result: %s", result)
    except SchedulerError as e:
        log.info("rejected: %s (%s)", e, e.code.value)

    for decision in scheduler.decision_log:
        for a in decision.attempts:
            log.info("attempt: %s", a)


if __name__ == "__main__":
    main()
