"""边缘推理调度：本地设备探测 + 模拟运行时 + 确定性调度。"""

from .capabilities import DeviceCapability, probe_local_devices
from .device import Device
from .errors import ErrorCode, SchedulerError
from .runtime import InferenceResult, SimulatedRuntime
from .scheduler import Scheduler
from .variants import ModelVariant, VariantRegistry

__all__ = [
    "Device",
    "DeviceCapability",
    "ErrorCode",
    "InferenceResult",
    "ModelVariant",
    "Scheduler",
    "SchedulerError",
    "SimulatedRuntime",
    "VariantRegistry",
    "probe_local_devices",
]
