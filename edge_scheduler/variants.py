"""模型变体注册与硬性约束描述。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ErrorCode, SchedulerError


@dataclass(frozen=True)
class ModelVariant:
    """一个模型变体：特定量化精度 + 算子集要求 + 内存占用。

    fallback_variant_ids 声明失败时的降级顺序；调度器只会按该顺序
    尝试同一模型的其他已注册变体，绝不静默换用未声明的变体。
    """

    variant_id: str
    model_name: str
    quantization: str  # 如 "fp32" / "fp16" / "int8"
    required_operators: frozenset[str] = field(default_factory=frozenset)
    memory_mb: int = 0
    fallback_variant_ids: tuple[str, ...] = ()


class VariantRegistry:
    """变体注册表：variant_id 唯一，重复注册视为错误。"""

    def __init__(self) -> None:
        self._variants: dict[str, ModelVariant] = {}

    def register(self, variant: ModelVariant) -> None:
        if variant.variant_id in self._variants:
            raise SchedulerError(
                ErrorCode.VARIANT_INCOMPATIBLE,
                f"duplicate variant id: {variant.variant_id}",
            )
        self._variants[variant.variant_id] = variant

    def get(self, variant_id: str) -> ModelVariant:
        try:
            return self._variants[variant_id]
        except KeyError:
            raise SchedulerError(
                ErrorCode.VARIANT_INCOMPATIBLE,
                f"unknown variant id: {variant_id}",
            ) from None

    def variants_for(self, model_name: str) -> list[ModelVariant]:
        """按注册顺序返回某模型的全部变体（确定性）。"""
        return [v for v in self._variants.values() if v.model_name == model_name]
