import logging

import pytest

from edge_scheduler import (
    Device,
    DeviceCapability,
    Scheduler,
    SimulatedRuntime,
    VariantRegistry,
)

log = logging.getLogger("scenario")


def make_scheduler(caps, variants, *, fault_plan=None, queue_timeout_s=0.2):
    registry = VariantRegistry()
    for v in variants:
        registry.register(v)
    devices = [Device(c) for c in caps]
    runtime = SimulatedRuntime(fault_plan=fault_plan)
    return Scheduler(devices, registry, runtime, queue_timeout_s=queue_timeout_s)


def log_decision(scheduler, request_id, normalized):
    """打印输入、判定依据（每次尝试）与归一化结果。"""
    decision = scheduler.decision_log[-1]
    log.info("input request_id=%s", request_id)
    for a in decision.attempts:
        log.info("  attempt variant=%s device=%s -> %s %s",
                 a.variant_id, a.device_id, a.outcome, a.reason)
    log.info("normalized result: %s", normalized)


@pytest.fixture
def cpu_only_cap():
    return DeviceCapability(
        device_id="cpu0", kind="cpu",
        supported_quantizations=frozenset({"fp32", "fp16"}),
        supported_operators=frozenset({"conv", "matmul", "relu"}),
        total_memory_mb=512, max_concurrency=2,
    )


@pytest.fixture
def npu_cap():
    return DeviceCapability(
        device_id="npu0", kind="npu",
        supported_quantizations=frozenset({"int8"}),
        supported_operators=frozenset({"conv", "matmul"}),
        total_memory_mb=256, max_concurrency=1,
    )
