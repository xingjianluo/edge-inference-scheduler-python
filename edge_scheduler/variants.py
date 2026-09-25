"""模型变体注册与设备能力的硬性匹配规则。

匹配是硬性的：设备必须同时满足变体要求的全部量化精度与算子集，
内存约束在调度准入阶段判定。不满足时给出可区分的缺失项，绝不静默换用。
"""
from __future__ import annotations

from dataclasses import dataclass

from .devices import DeviceCapability
from .errors import ErrorKind, SchedulerError


@dataclass(frozen=True)
class ModelVariant:
    """同一逻辑模型的一个具体变体。

    required_dtypes: 必须全部被设备支持（通常只放一种精度，如 int8）。
    required_ops:    需要的算子集，设备必须是其超集。
    memory_mb:       加载+推理所需的内存/显存。
    """

    model: str
    variant_id: str
    required_dtypes: frozenset[str]
    required_ops: frozenset[str]
    memory_mb: int

    @property
    def key(self) -> str:
        return f"{self.model}@{self.variant_id}"


@dataclass(frozen=True)
class FallbackChain:
    """确定顺序的后备路径：按 variant_ids 顺序依次尝试。"""

    model: str
    variant_ids: tuple[str, ...]


@dataclass(frozen=True)
class MismatchDetail:
    """单个变体与单个设备不匹配的可区分原因。"""

    device_id: str
    missing_dtypes: frozenset[str] = frozenset()
    missing_ops: frozenset[str] = frozenset()

    @property
    def reasons(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.missing_dtypes:
            out.append(f"missing_dtypes={sorted(self.missing_dtypes)}")
        if self.missing_ops:
            out.append(f"missing_ops={sorted(self.missing_ops)}")
        return tuple(out)


def check_compatibility(
    variant: ModelVariant, device: DeviceCapability
) -> MismatchDetail | None:
    """返回 None 表示兼容；否则返回具体缺失项（精度/算子分别列出）。"""
    missing_dtypes = variant.required_dtypes - device.dtypes
    missing_ops = variant.required_ops - device.ops
    if not missing_dtypes and not missing_ops:
        return None
    return MismatchDetail(
        device_id=device.device_id,
        missing_dtypes=frozenset(missing_dtypes),
        missing_ops=frozenset(missing_ops),
    )


class VariantRegistry:
    """模型变体与后备链的注册表。"""

    def __init__(self) -> None:
        self._variants: dict[str, dict[str, ModelVariant]] = {}
        self._chains: dict[str, FallbackChain] = {}

    def register(self, variant: ModelVariant) -> None:
        bucket = self._variants.setdefault(variant.model, {})
        if variant.variant_id in bucket:
            raise SchedulerError(
                ErrorKind.VARIANT_INCOMPATIBLE,
                f"变体重复注册: {variant.key}",
                details={"model": variant.model, "variant_id": variant.variant_id},
            )
        bucket[variant.variant_id] = variant

    def register_chain(self, chain: FallbackChain) -> None:
        for vid in chain.variant_ids:
            # 后备链中引用的所有变体必须存在，避免链路引用空洞
            self.get(chain.model, vid)
        self._chains[chain.model] = chain

    def get(self, model: str, variant_id: str) -> ModelVariant:
        bucket = self._variants.get(model)
        if bucket is None or variant_id not in bucket:
            raise SchedulerError(
                ErrorKind.VARIANT_INCOMPATIBLE,
                f"未注册的模型变体: {model}@{variant_id}",
                details={"model": model, "variant_id": variant_id},
            )
        return bucket[variant_id]

    def chain_for(
        self, model: str, preferred_variant_id: str, *, use_chain: bool = False
    ) -> list[ModelVariant]:
        """返回候选变体（确定顺序）。

        - use_chain=False（默认，严格模式）：只返回指定变体。该变体没有任何
          兼容设备时直接拒绝，绝不静默换用精度/算子要求不符的其它变体。
        - use_chain=True（显式降级模式）：首选变体在前，之后接该模型已注册
          后备链中的其余变体（调用方显式同意降级），去重但保持顺序。
        """
        preferred = self.get(model, preferred_variant_id)
        if not use_chain:
            return [preferred]
        chain = self._chains.get(model)
        ordered_ids: list[str] = [preferred_variant_id]
        if chain is not None:
            for vid in chain.variant_ids:
                if vid not in ordered_ids:
                    ordered_ids.append(vid)
        return [self.get(model, vid) for vid in ordered_ids]

    def find_compatible(
        self, variant: ModelVariant, devices: list[DeviceCapability]
    ) -> tuple[list[DeviceCapability], list[MismatchDetail]]:
        """返回 (兼容设备列表[保持入参顺序], 每个不兼容设备的原因)。"""
        compatible: list[DeviceCapability] = []
        mismatches: list[MismatchDetail] = []
        for dev in devices:
            detail = check_compatibility(variant, dev)
            if detail is None:
                compatible.append(dev)
            else:
                mismatches.append(detail)
        return compatible, mismatches
