"""场景三：推理失败链路的归一化与确定性降级重试。"""

import logging

import pytest

from edge_scheduler import (
    Device,
    DeviceCapability,
    ErrorCode,
    ModelVariant,
    Scheduler,
    SchedulerError,
    SimulatedRuntime,
    VariantRegistry,
)
from conftest import log_decision, make_scheduler

log = logging.getLogger("scenario.failure")


@pytest.fixture
def two_devices():
    cpu = DeviceCapability(
        device_id="cpu0", kind="cpu",
        supported_quantizations=frozenset({"fp32"}),
        supported_operators=frozenset({"conv", "matmul"}),
        total_memory_mb=512, max_concurrency=2,
    )
    npu = DeviceCapability(
        device_id="npu0", kind="npu",
        supported_quantizations=frozenset({"int8"}),
        supported_operators=frozenset({"conv"}),
        total_memory_mb=256, max_concurrency=1,
    )
    return [cpu, npu]


def test_timeout_falls_back_in_declared_order(two_devices):
    """int8 主变体超时后，按声明顺序降级到 fp32 变体并成功。"""
    variants = [
        ModelVariant("m-int8", "m", "int8", frozenset({"conv"}), 64,
                     fallback_variant_ids=("m-fp32",)),
        ModelVariant("m-fp32", "m", "fp32", frozenset({"conv", "matmul"}), 128),
    ]
    sch = make_scheduler(two_devices, variants,
                         fault_plan={("npu0", "m-int8"): ["timeout"]})
    result = sch.submit("req-fb", "m-int8", "payload")
    assert result.variant_id == "m-fp32"
    assert result.device_id == "cpu0"
    outcomes = [a.outcome for a in sch.decision_log[-1].attempts]
    assert outcomes == [ErrorCode.INFERENCE_TIMEOUT.value, "success"]
    log_decision(sch, "req-fb", f"degraded to {result.variant_id}")


def test_device_unavailable_mid_inference(two_devices):
    """设备中途不可用被归一化为 DEVICE_UNAVAILABLE，且重试不产生重复副作用。"""
    variants = [
        ModelVariant("m-int8", "m", "int8", frozenset({"conv"}), 64,
                     fallback_variant_ids=("m-fp32",)),
        ModelVariant("m-fp32", "m", "fp32", frozenset({"conv", "matmul"}), 128),
    ]
    sch = make_scheduler(two_devices, variants,
                         fault_plan={("npu0", "m-int8"): ["device_unavailable"]})
    result = sch.submit("req-down", "m-int8", "payload")
    assert result.variant_id == "m-fp32"
    first = sch.decision_log[-1].attempts[0]
    assert first.outcome == ErrorCode.DEVICE_UNAVAILABLE.value
    # 每个请求只产生一次成功结果，资源全部释放
    for device in sch._devices:
        assert device.active_count == 0
        assert device.used_memory_mb == 0
    log_decision(sch, "req-down", f"degraded to {result.variant_id}")


def test_error_types_are_distinguishable(two_devices):
    """算子不支持 / 超时 / 设备不可用归一化为三种可区分错误。"""
    variants = [
        # 需要 npu 没有的算子 -> 运行时报 OPERATOR_UNSUPPORTED（能力探测后仍注入）
        ModelVariant("m-op", "m1", "int8", frozenset({"conv"}), 64),
        ModelVariant("m-to", "m2", "int8", frozenset({"conv"}), 64),
        ModelVariant("m-down", "m3", "int8", frozenset({"conv"}), 64),
    ]
    sch = make_scheduler(
        two_devices, variants,
        fault_plan={
            ("npu0", "m-op"): ["runtime_weird"],          # 未知注入 -> RUNTIME_FAILURE
            ("npu0", "m-to"): ["timeout"],                # -> INFERENCE_TIMEOUT
            ("npu0", "m-down"): ["device_unavailable"],   # -> DEVICE_UNAVAILABLE
        },
    )
    codes = {}
    for rid, vid in [("r-op", "m-op"), ("r-to", "m-to"), ("r-down", "m-down")]:
        with pytest.raises(SchedulerError) as exc:
            sch.submit(rid, vid, "p")
        codes[vid] = exc.value.code
        log_decision(sch, rid, exc.value.code.value)
    assert codes["m-op"] == ErrorCode.RUNTIME_FAILURE
    assert codes["m-to"] == ErrorCode.INFERENCE_TIMEOUT
    assert codes["m-down"] == ErrorCode.DEVICE_UNAVAILABLE
    assert len({c.value for c in codes.values()}) == 3


def test_released_resources_not_reused_after_failure(two_devices):
    """失败后资源被释放；后续请求可正常调度且不会用到已释放的占用。"""
    variants = [
        ModelVariant("m-int8", "m", "int8", frozenset({"conv"}), 64,
                     fallback_variant_ids=("m-fp32",)),
        ModelVariant("m-fp32", "m", "fp32", frozenset({"conv", "matmul"}), 128),
    ]
    sch = make_scheduler(two_devices, variants,
                         fault_plan={("npu0", "m-int8"): ["timeout"]})
    r1 = sch.submit("req-a", "m-int8", "p1")
    r2 = sch.submit("req-b", "m-int8", "p2")  # 第二次无故障，直接成功
    assert r1.variant_id == "m-fp32"
    assert r2.variant_id == "m-int8"
    assert r1.request_id == "req-a" and r2.request_id == "req-b"
    # 运行时调用日志：req-a 只在 npu0 尝试过一次（无重复副作用）
    runtime_calls = [c for c in sch._runtime.call_log if c[0] == "req-a"]
    assert runtime_calls == [("req-a", "npu0", "m-int8"),
                             ("req-a", "cpu0", "m-fp32")]
    for device in sch._devices:
        assert device.active_count == 0 and device.used_memory_mb == 0
    log_decision(sch, "req-b", "success after released resources reused safely")
