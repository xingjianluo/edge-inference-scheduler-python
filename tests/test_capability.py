"""场景一：能力缺失与变体不兼容，拒绝原因必须可区分。"""

import logging

import pytest

from edge_scheduler import ErrorCode, ModelVariant, SchedulerError
from conftest import log_decision, make_scheduler

log = logging.getLogger("scenario.capability")


def test_capability_missing_rejected(cpu_only_cap):
    """int8 变体在仅支持 fp32/fp16 的 CPU 上必须被拒绝为 CAPABILITY_MISSING。"""
    variants = [
        ModelVariant("m-int8", "m", "int8", frozenset({"conv"}), memory_mb=64),
    ]
    sch = make_scheduler([cpu_only_cap], variants)
    with pytest.raises(SchedulerError) as exc:
        sch.submit("req-cap", "m-int8", "payload")
    assert exc.value.code == ErrorCode.CAPABILITY_MISSING
    log_decision(sch, "req-cap", exc.value.code.value)


def test_missing_operator_rejected(cpu_only_cap):
    """需要 layernorm 的变体在算子缺失设备上被拒绝，原因中列出缺失算子。"""
    variants = [
        ModelVariant("m-ln", "m", "fp32", frozenset({"conv", "layernorm"}), 64),
    ]
    sch = make_scheduler([cpu_only_cap], variants)
    with pytest.raises(SchedulerError) as exc:
        sch.submit("req-op", "m-ln", "payload")
    assert exc.value.code == ErrorCode.CAPABILITY_MISSING
    attempt = sch.decision_log[-1].attempts[0]
    assert "layernorm" in attempt.reason
    log_decision(sch, "req-op", exc.value.code.value)


def test_unknown_variant_is_incompatible(cpu_only_cap):
    """未注册变体 -> VARIANT_INCOMPATIBLE，与 CAPABILITY_MISSING 可区分。"""
    sch = make_scheduler([cpu_only_cap], [])
    with pytest.raises(SchedulerError) as exc:
        sch.submit("req-unknown", "no-such-variant", "payload")
    assert exc.value.code == ErrorCode.VARIANT_INCOMPATIBLE
    log.info("normalized result: %s", exc.value.code.value)


def test_cross_model_fallback_rejected(cpu_only_cap):
    """降级链指向其他模型的变体时必须拒绝，不得静默换用。"""
    variants = [
        ModelVariant("a-fp32", "model-a", "fp32", frozenset({"conv"}), 64,
                     fallback_variant_ids=("b-fp32",)),
        ModelVariant("b-fp32", "model-b", "fp32", frozenset({"conv"}), 64),
    ]
    sch = make_scheduler([cpu_only_cap], variants)
    with pytest.raises(SchedulerError) as exc:
        sch.submit("req-xmodel", "a-fp32", "payload")
    assert exc.value.code == ErrorCode.VARIANT_INCOMPATIBLE
    log.info("normalized result: %s", exc.value.code.value)


def test_compatible_variant_scheduled(cpu_only_cap, npu_cap):
    """int8 变体只能落到 npu0，证明匹配不是随机选择。"""
    variants = [
        ModelVariant("m-int8", "m", "int8", frozenset({"conv"}), 64),
    ]
    sch = make_scheduler([cpu_only_cap, npu_cap], variants)
    result = sch.submit("req-ok", "m-int8", "payload")
    assert result.device_id == "npu0"
    assert result.request_id == "req-ok"
    log_decision(sch, "req-ok", f"success on {result.device_id}")
