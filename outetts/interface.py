from loguru import logger
import os

from .version.interface import (
    InterfaceHF, 
    InterfaceEXL2, InterfaceEXL2Async,
    InterfaceVLLMBatch, 
)
from .models.config import ModelConfig
from .models.info import Backend, InterfaceVersion

def Interface(config: ModelConfig) -> (
        InterfaceHF | InterfaceEXL2 | InterfaceEXL2Async |
        InterfaceVLLMBatch |):

    if config.backend == Backend.HF:
        return InterfaceHF(config)
    elif config.backend == Backend.EXL2:
        return InterfaceEXL2(config)
    elif config.backend == Backend.EXL2ASYNC:
        return InterfaceEXL2Async(config)
    elif config.backend == Backend.VLLM:
        warning_msg = "VLLM backend is experimental and may cause issues with audio generation."
        logger.warning(warning_msg)
        return InterfaceVLLMBatch(config)
    raise ValueError(f"Invalid backend: {config.backend} - must be one of {list(Backend)}")
