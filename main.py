"""只读 HTTP 概览：展示本地设备探测、变体匹配与错误分类。

不执行真实推理、不访问外部服务。启动：uv run python main.py
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from edge_scheduler.devices import DevicePool, LocalProbe
from edge_scheduler.errors import ErrorKind
from edge_scheduler.runtime import SimulatedRuntime
from edge_scheduler.scheduler import EdgeScheduler
from edge_scheduler.variants import (
    FallbackChain,
    ModelVariant,
    VariantRegistry,
)

app = FastAPI(title="edge-inference-scheduler (read-only overview)")


def _build_scheduler() -> EdgeScheduler:
    reg = VariantRegistry()
    reg.register(ModelVariant("llm", "int8", frozenset({"int8"}),
                              frozenset({"matmul", "rope"}), 400))
    reg.register(ModelVariant("llm", "fp16", frozenset({"fp16"}),
                              frozenset({"matmul", "flash_attention"}), 1500))
    reg.register_chain(FallbackChain("llm", ("int8", "fp16")))
    return EdgeScheduler(
        reg, DevicePool(LocalProbe().detect()), SimulatedRuntime()
    )


SCHEDULER = _build_scheduler()


@app.get("/")
def read_root():
    return {"service": "edge-inference-scheduler", "overview": "/devices"}


@app.get("/devices")
def list_devices():
    out = []
    for st in SCHEDULER.pool.list_devices():
        c = st.capability
        out.append(
            {
                "device_id": c.device_id,
                "name": c.name,
                "state": st.state.value,
                "dtypes": sorted(c.dtypes),
                "ops": sorted(c.ops),
                "memory_mb": c.memory_mb,
                "used_memory_mb": st.used_memory_mb,
                "free_memory_mb": st.free_memory_mb,
                "is_simulated": c.is_simulated,
            }
        )
    return {"devices": out}


@app.get("/variants/{model}/{variant_id}/match")
def match_variant(model: str, variant_id: str):
    variant = SCHEDULER.registry.get(model, variant_id)
    caps = [s.capability for s in SCHEDULER.pool.list_devices()]
    compatible, mismatches = SCHEDULER.registry.find_compatible(variant, caps)
    return {
        "variant": variant.key,
        "required_dtypes": sorted(variant.required_dtypes),
        "required_ops": sorted(variant.required_ops),
        "memory_mb": variant.memory_mb,
        "compatible_devices": [c.device_id for c in compatible],
        "rejections": [
            {
                "device_id": m.device_id,
                "kind": ErrorKind.CAPABILITY_MISSING.value,
                "missing_dtypes": sorted(m.missing_dtypes),
                "missing_ops": sorted(m.missing_ops),
            }
            for m in mismatches
        ],
    }


@app.get("/errors")
def error_kinds():
    return {"error_kinds": [k.value for k in ErrorKind]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=0)
