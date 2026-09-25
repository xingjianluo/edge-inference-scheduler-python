"""单测/演示用的判定日志：打印输入、判定依据与归一化结果。"""
from __future__ import annotations

import logging
import sys

_CONFIGURED: set[str] = set()


def build_logger(name: str = "edge_scheduler.tests") -> logging.Logger:
    """返回一个输出到 stderr 的 logger；每个名字只配置一次 handler。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if name not in _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        _CONFIGURED.add(name)
    return logger


def log_case(logger: logging.Logger, case: str, inp: object, basis: object, result: object) -> None:
    """统一打印：输入、判定依据、归一化结果。"""
    logger.info("case=%s input=%s", case, inp)
    logger.info("case=%s basis=%s", case, basis)
    logger.info("case=%s normalized_result=%s", case, result)
