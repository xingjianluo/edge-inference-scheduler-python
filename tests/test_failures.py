"""场景四：推理链路失败与降级。

- 运行时不支持算子 / 单次调用超时 / 设备中途不可用，分别归一化为
  unsupported_op / inference_timeout / device_unavailable，彼此可区分；
- 显式后备链下按确定顺序重试；
- 失败尝试不产生重复副作用；已释放资源不得被再次使用；
- 设备掉线后后续候选跳过该设备；全部失败聚合为 fallback_exhausted，
  设备状态最终一致。
"""
from __future__ import annotations

import threading

import pytest

from edge_scheduler.errors import ErrorKind, SchedulerError
from edge_scheduler.logging_utils import log_case
from edge_scheduler.runtime import FailureSpec, InferenceInput
from edge_scheduler.scheduler import SchedulerConfig


def test_unsupported_op_is_normalized(logger, scheduler_factory, pool):
    # 首选 int8 在 npu 上运行时算子不支持（注入），显式后备链继续到 fp16
    sch = scheduler_factory(log=True)
    res = sch.infer(
        "llm",
        "int8",
        InferenceInput("f-op", "p"),
        failure=FailureSpec(ErrorKind.UNSUPPORTED_OP, frozenset({1})),
        use_chain=True,
    )
    assert res.variant_id == "fp16" and res.device_id == "sim-gpu"
    assert sch.runtime.side_effect_count("f-op") == 1  # 仅成功的一次产生副作用
    log_case(
        logger,
        "fail_unsupported_op",
        inp="f-op preferred=int8 injected=unsupported_op@attempt1 use_chain=True",
        basis="normalized unsupported_op -> release npu -> next chain variant fp16",
        result=f"ok on {res.variant_id}@{res.device_id}, side_effects=1",
    )
    assert pool.get("sim-npu").used_memory_mb == 0
    assert pool.get("sim-gpu").used_memory_mb == 0


def test_timeout_is_normalized_and_retried_once(logger, scheduler_factory, pool):
    sch = scheduler_factory(log=True)
    # attempt1 超时；后备到 fp16 成功，且不重复执行 int8 的副作用
    res = sch.infer(
        "llm",
        "int8",
        InferenceInput("f-timeout", "p"),
        failure=FailureSpec(ErrorKind.INFERENCE_TIMEOUT, frozenset({1})),
        use_chain=True,
    )
    assert res.variant_id == "fp16"
    assert res.attempt == 2
    assert sch.runtime.side_effect_count("f-timeout") == 1
    log_case(
        logger,
        "fail_timeout",
        inp="f-timeout injected=timeout@attempt1 use_chain=True",
        basis="normalized inference_timeout -> next candidate; failed attempt no side effect",
        result=f"ok attempt=2 on {res.device_id}, side_effects=1",
    )
    assert pool.get("sim-npu").used_memory_mb == 0


def test_device_unavailable_marks_offline_and_skips(
    logger, registry, pool, scheduler_factory
):
    # int8 第一次执行时设备中途不可用 => npu 标记离线；
    # 链上后续 fp32 也可在 npu 跑，但 npu 已离线必须跳过，最终落到 fp16(sim-gpu)
    sch = scheduler_factory(log=True)
    res = sch.infer(
        "llm",
        "int8",
        InferenceInput("f-dev", "p"),
        failure=FailureSpec(ErrorKind.DEVICE_UNAVAILABLE, frozenset({1})),
        use_chain=True,
    )
    assert res.device_id == "sim-gpu"
    assert pool.get("sim-npu").state.value == "offline"
    # 离线设备上的失败占用已释放
    assert pool.get("sim-npu").used_memory_mb == 0
    assert pool.get("sim-npu").active_requests == set()
    log_case(
        logger,
        "fail_device_unavailable",
        inp="f-dev injected=device_unavailable@attempt1 use_chain=True",
        basis="normalized device_unavailable -> mark npu offline -> skip npu later",
        result=f"ok on {res.device_id}; npu offline, npu memory released",
    )


def test_all_paths_fail_aggregates_distinct_reasons(
    logger, scheduler_factory, pool
):
    # 让每一次尝试都失败（1..3 次尝试分别注入不同错误类型），
    # 最终聚合为 fallback_exhausted，且保留每步可区分原因
    sch = scheduler_factory(
        SchedulerConfig(queue_capacity=4, admission_timeout=2.0), log=True
    )

    # 用一个按 attempt 切换错误类型的 FailureSpec：通过三次不同 kind 注入
    # FailureSpec 只携带单一 kind，因此用三个变体链上分别注入：
    # 采用对 runtime.run 的计数包装实现按 attempt 变化
    kinds = [
        ErrorKind.UNSUPPORTED_OP,
        ErrorKind.INFERENCE_TIMEOUT,
        ErrorKind.DEVICE_UNAVAILABLE,
    ]
    orig_run = sch.runtime.run
    counter = {"n": 0}
    cvl = threading.Lock()

    def run_by_attempt(variant, device, inp, **kwargs):
        kwargs.pop("failure", None)  # 按 attempt 注入不同错误，覆盖入参
        with cvl:
            idx = counter["n"] % len(kinds)
            counter["n"] += 1
        failure = FailureSpec(kinds[idx], frozenset({kwargs["attempt"]}))
        return orig_run(variant, device, inp, failure=failure, **kwargs)

    sch.runtime.run = run_by_attempt  # type: ignore[method-assign]

    with pytest.raises(SchedulerError) as ei:
        sch.infer(
            "llm", "int8", InferenceInput("f-all", "p"), use_chain=True
        )
    err = ei.value
    assert err.kind is ErrorKind.FALLBACK_EXHAUSTED
    observed = [a["kind"] for a in err.details["attempts"]]
    # 至少三种归一化错误都可区分地出现
    for k in ("unsupported_op", "inference_timeout", "device_unavailable"):
        assert k in observed, observed
    log_case(
        logger,
        "fail_all_paths",
        inp="f-all use_chain=True; attempts fail with distinct kinds",
        basis=err.details["attempts"],
        result=f"{err.kind.value}: {len(observed)} attempts, reasons distinct",
    )
    # 任何成功副作用都没有；占用全部释放（含掉线设备）
    assert sch.runtime.side_effect_count("f-all") == 0
    for st in pool.list_devices():
        assert st.used_memory_mb == 0
        assert st.active_requests == set()


def test_retry_does_not_reuse_released_resource(
    logger, scheduler_factory, pool
):
    # 第一次尝试失败并释放 npu 占用后，重试到 fp16 成功；
    # 再直接对同一请求重复释放 npu 必须被拒绝（已释放资源不可再用）
    sch = scheduler_factory(log=True)
    res = sch.infer(
        "llm",
        "int8",
        InferenceInput("f-rel", "p"),
        failure=FailureSpec(ErrorKind.INFERENCE_TIMEOUT, frozenset({1})),
        use_chain=True,
    )
    assert res.variant_id == "fp16"
    # npu 上属于 f-rel 的占用已被调度器释放：重复释放必须报错
    with pytest.raises(SchedulerError) as ei:
        pool.release("sim-npu", "f-rel", 400)
    assert ei.value.kind is ErrorKind.DUPLICATE_REQUEST
    log_case(
        logger,
        "fail_no_reuse_after_release",
        inp="f-rel fail@npu then succeed@gpu; double-release npu afterwards",
        basis="ownership token absent after scheduler finally-release",
        result="double release rejected duplicate_request",
    )
    assert pool.get("sim-npu").used_memory_mb == 0
