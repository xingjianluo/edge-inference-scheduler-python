"""共享测试夹具：构造确定性的注册表/设备池/调度器，并统一日志。"""
from __future__ import annotations

import logging

import pytest

from edge_scheduler.devices import DevicePool, LocalProbe
from edge_scheduler.logging_utils import build_logger
from edge_scheduler.runtime import SimulatedRuntime
from edge_scheduler.scheduler import EdgeScheduler, SchedulerConfig
from edge_scheduler.variants import FallbackChain, ModelVariant, VariantRegistry


@pytest.fixture
def logger() -> logging.Logger:
    return build_logger()


def make_variant(
    variant_id: str,
    dtypes: set[str] | frozenset[str],
    ops: set[str] | frozenset[str],
    memory_mb: int,
    model: str = "llm",
) -> ModelVariant:
    return ModelVariant(
        model=model,
        variant_id=variant_id,
        required_dtypes=frozenset(dtypes),
        required_ops=frozenset(ops),
        memory_mb=memory_mb,
    )


@pytest.fixture
def registry() -> VariantRegistry:
    """标准注册表：
    - int8 变体只能在 sim-npu（400MB，需要 rope）
    - fp16 变体只能在 sim-gpu（1500MB，需要 flash_attention）
    - fp32 变体可在任意设备（200MB，基础算子）
    """
    reg = VariantRegistry()
    base = {"matmul", "softmax"}
    reg.register(make_variant("int8", {"int8"}, base | {"rope"}, 400))
    reg.register(make_variant("fp16", {"fp16"}, base | {"flash_attention"}, 1500))
    reg.register(make_variant("fp32", {"fp32"}, base, 200))
    reg.register_chain(FallbackChain("llm", ("int8", "fp16", "fp32")))
    return reg


@pytest.fixture
def pool() -> DevicePool:
    # local-cpu 内存很大；模拟设备分别为 2048MB / 512MB
    return DevicePool(LocalProbe(include_simulated=True).detect())


@pytest.fixture
def runtime() -> SimulatedRuntime:
    return SimulatedRuntime()


@pytest.fixture
def scheduler_factory(registry, pool, runtime):
    def _factory(config: SchedulerConfig | None = None, log: bool = False):
        logger = build_logger()
        return EdgeScheduler(
            registry,
            pool,
            runtime,
            config=config,
            logger=(logger.info if log else None),
        )

    return _factory
