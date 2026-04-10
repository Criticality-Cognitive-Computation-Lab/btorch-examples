import logging
from pathlib import Path

import numpy as np
import torch
import pandas as pd

def is_train_mode(cfg):
    #检查cfg中是否有train
    return ("train" in cfg)

def setup_logging(log_dir: Path) -> logging.Logger:
    # """Setup logging configuration."""
    # log_dir.mkdir(parents=True, exist_ok=True)

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s",
    #     handlers=[
    #         logging.FileHandler(log_dir / "training.log"),
    #         logging.StreamHandler(),
    #     ],
    # )
    # return logging.getLogger(__name__)

    """Setup logging configuration with forced handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    # 获取当前模块的 logger，或者使用 root logger
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)

    #以此清除旧的 handler (防止重复打印)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. 创建 File Handler (写入文件)
    file_handler = logging.FileHandler(log_file, mode='w') # mode='w' 每次运行覆盖，'a' 为追加
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. 创建 Stream Handler (输出到控制台)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s") # 控制台通常只需要消息，不需要太长的时间戳
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logging.getLogger(__name__) # 返回当前模块 logger 供后续使用

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def simple_id_to_root_id_func(neurons: pd.DataFrame, reverse: bool = False):
    return np.vectorize(
        dict(
            neurons[
                ["root_id", "simple_id"] if reverse else ["simple_id", "root_id"]
            ].to_numpy()
        ).get
    )