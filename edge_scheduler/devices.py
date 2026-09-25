"""本地设备能力探测与设备池状态管理。

仅使用标准库进行本地探测（platform / os / multiprocessing），
不依赖真实加速卡或外部服务。探测结果是确定性的，可在无 GPU 机器复现。
"""
from __future__ import annotations

import multiprocessing
import os
import platform
import threading
from dataclasses import dataclass, field
from enum import Enum

from .errors import ErrorKind, SchedulerError

# 基准算子集：本地 CPU 运行时“支持”的算子（模拟，确定性）
_BASE_CPU_OPS = frozenset({"matmul", "conv2d", "relu", "softmax", "layernorm"})
# 模拟出的 NPU 额外算子
_EXTRA_NPU_OPS = frozenset({"rope", "flash_attention", "quantized_matmul"})


class DeviceState(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True)
class DeviceCapability:
    """设备的硬性能力画像。"""

    device_id: str
    name: str
    dtypes: frozenset[str]                 # 支持的量化精度，如 fp32/int8
    ops: frozenset[str]                    # 支持的算子集
    memory_mb: int                         # 总内存/显存 (MB)
    is_simulated: bool = False


@dataclass
class DeviceRuntimeState:
    """设备在设备池中的运行时状态（已占用内存、设备在线状态）。"""

    capability: DeviceCapability
    state: DeviceState = DeviceState.ONLINE
    used_memory_mb: int = 0
    active_requests: set[str] = field(default_factory=set)

    @property
    def free_memory_mb(self) -> int:
        return self.capability.memory_mb - self.used_memory_mb


class LocalProbe:
    """本地设备探测器。

    - 始终探测出一个真实的本地 CPU 设备（能力来自本机 CPU 数量与 cgroup/物理内存，
      在受限环境下回退到确定的保守默认值）。
    - 另提供两个确定性的模拟设备（低算力 NPU、中算力 GPU），用于复现
      “不同量化精度/算子集只被满足硬约束的设备承载”的场景。
    """

    def __init__(self, *, include_simulated: bool = True) -> None:
        self.include_simulated = include_simulated

    def detect(self) -> list[DeviceCapability]:
        devices = [self._local_cpu_caps()]
        if self.include_simulated:
            devices.append(self._simulated_gpu_caps())
            devices.append(self._simulated_npu_caps())
        return devices

    def _cpu_count(self) -> int:
        try:
            return max(1, multiprocessing.cpu_count())
        except NotImplementedError:
            return 1

    def _host_memory_mb(self) -> int:
        """尽力读取本机内存；任何失败都回退到确定性默认 1024MB。"""
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            mem_mb = int(pages * page_size / (1024 * 1024))
            return max(256, mem_mb)
        except (ValueError, OSError, AttributeError):
            return 1024

    def _local_cpu_caps(self) -> DeviceCapability:
        return DeviceCapability(
            device_id="local-cpu",
            name=f"Local CPU ({platform.processor() or platform.machine()})",
            # 本地 CPU 只保证 fp32；不承诺任何量化精度与专有算子
            dtypes=frozenset({"fp32"}),
            ops=_BASE_CPU_OPS,
            memory_mb=self._host_memory_mb(),
            is_simulated=False,
        )

    def _simulated_gpu_caps(self) -> DeviceCapability:
        return DeviceCapability(
            device_id="sim-gpu",
            name="Simulated mid-tier GPU",
            dtypes=frozenset({"fp32", "fp16"}),
            ops=_BASE_CPU_OPS | {"flash_attention"},
            memory_mb=2048,
            is_simulated=True,
        )

    def _simulated_npu_caps(self) -> DeviceCapability:
        return DeviceCapability(
            device_id="sim-npu",
            name="Simulated low-memory NPU",
            dtypes=frozenset({"fp32", "int8"}),
            ops=_BASE_CPU_OPS | _EXTRA_NPU_OPS,
            memory_mb=512,
            is_simulated=True,
        )


class DevicePool:
    """设备注册表：负责内存准入、占用登记与设备状态一致性。

    所有状态变更都在同一把锁下完成，保证并发下不出现重复占用。
    request_id 是占用归属凭证：同一请求不能重复占用同一设备，
    释放时必须核对凭证与金额，杜绝“已释放资源被再次使用”。
    """

    def __init__(self, capabilities: list[DeviceCapability]):
        self._states: dict[str, DeviceRuntimeState] = {
            c.device_id: DeviceRuntimeState(capability=c) for c in capabilities
        }
        self._lock = threading.RLock()

    def get(self, device_id: str) -> DeviceRuntimeState:
        with self._lock:
            if device_id not in self._states:
                raise SchedulerError(
                    ErrorKind.CAPABILITY_MISSING,
                    f"设备不存在: {device_id}",
                    details={"device_id": device_id},
                )
            return self._states[device_id]

    def list_devices(self) -> list[DeviceRuntimeState]:
        with self._lock:
            return list(self._states.values())

    def try_admit(
        self, device_id: str, request_id: str, need_memory_mb: int
    ) -> None:
        """准入：成功则登记占用；失败抛出归一化 SchedulerError。"""
        with self._lock:
            st = self.get(device_id)
            if st.state is DeviceState.OFFLINE:
                raise SchedulerError(
                    ErrorKind.DEVICE_UNAVAILABLE,
                    f"设备 {device_id} 已离线，拒绝准入",
                    details={"device_id": device_id, "request_id": request_id},
                )
            if request_id in st.active_requests:
                raise SchedulerError(
                    ErrorKind.DUPLICATE_REQUEST,
                    f"请求 {request_id} 已占用设备 {device_id}，禁止重复占用",
                    details={"device_id": device_id, "request_id": request_id},
                )
            if need_memory_mb > st.free_memory_mb:
                raise SchedulerError(
                    ErrorKind.INSUFFICIENT_MEMORY,
                    (
                        f"设备 {device_id} 内存不足: 需要 {need_memory_mb}MB, "
                        f"空闲 {st.free_memory_mb}MB / 总 {st.capability.memory_mb}MB"
                    ),
                    details={
                        "device_id": device_id,
                        "request_id": request_id,
                        "need_mb": need_memory_mb,
                        "free_mb": st.free_memory_mb,
                        "total_mb": st.capability.memory_mb,
                    },
                )
            st.used_memory_mb += need_memory_mb
            st.active_requests.add(request_id)

    def release(
        self, device_id: str, request_id: str, used_memory_mb: int
    ) -> None:
        """释放占用；重复释放必须报错，禁止已释放资源被再次使用。"""
        with self._lock:
            st = self.get(device_id)
            if request_id not in st.active_requests:
                raise SchedulerError(
                    ErrorKind.DUPLICATE_REQUEST,
                    f"请求 {request_id} 未占用设备 {device_id}，禁止重复/越权释放",
                    details={"device_id": device_id, "request_id": request_id},
                )
            st.active_requests.discard(request_id)
            st.used_memory_mb = max(0, st.used_memory_mb - used_memory_mb)

    def mark_offline(self, device_id: str) -> None:
        with self._lock:
            self.get(device_id).state = DeviceState.OFFLINE

    def mark_online(self, device_id: str) -> None:
        with self._lock:
            self.get(device_id).state = DeviceState.ONLINE
