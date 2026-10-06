"""本地演示：探测设备 -> 注册变体 -> 提交请求 -> 打印调度决策。

运行：uv run python main.py
可用 EDGE_SIM_DEVICES 声明模拟加速设备（见 README）。
"""

import logging
import threading
import time

from edge_scheduler import (
    Device,
    DeviceCapability,
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

    demo_queue_governance()


def demo_queue_governance() -> None:
    """排队治理演示：背压拒绝、取消、优先级准入（小规模可复现）。"""
    cap = DeviceCapability(
        device_id="sim0", kind="npu",
        supported_quantizations=frozenset({"fp32"}),
        supported_operators=frozenset({"conv"}),
        total_memory_mb=64, max_concurrency=1,
    )
    device = Device(cap, max_queue_size=1, aging_after_s=0.5)
    registry = VariantRegistry()
    registry.register(ModelVariant(
        "q-fp32", "q", "fp32", frozenset({"conv"}), memory_mb=64,
    ))
    sch = Scheduler([device], registry, SimulatedRuntime(latency_s=0.3),
                    queue_timeout_s=2.0)

    def submit(rid, priority=0):
        try:
            log.info("%s -> %s", rid,
                     sch.submit(rid, "q-fp32", "p", priority=priority).output)
        except SchedulerError as e:
            log.info("%s rejected: %s", rid, e.code.value)

    # r1 占用设备，r2 进入等待队列（容量 1）
    t1 = threading.Thread(target=submit, args=("r1",))
    t2 = threading.Thread(target=submit, args=("r2",))
    t1.start()
    time.sleep(0.05)
    t2.start()
    time.sleep(0.05)

    # 队列已满：r3 立即被背压拒绝
    submit("r3")

    # 取消等待中的 r2：它不会再占用资源，r4 随后补位成功
    log.info("cancel r2 -> %s", sch.cancel("r2"))
    t4 = threading.Thread(target=submit, args=("r4",), kwargs={"priority": 5})
    t4.start()
    for t in (t1, t2, t4):
        t.join()

    for decision in sch.decision_log:
        for a in decision.attempts:
            log.info("attempt: %s", a)


if __name__ == "__main__":
    main()
