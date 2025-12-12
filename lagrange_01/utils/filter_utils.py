"""
信号滤波工具函数。

提供低通滤波等信号处理功能，用于平滑控制信号。
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class LowPassFilter:
    """
    一阶低通滤波器。
    
    用于平滑控制信号，减少噪声和突变。
    
    公式：
        y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    
    其中：
        alpha = dt / (dt + tau)
        tau: 时间常数（秒），越小响应越快
        dt: 采样时间间隔（秒）
    """
    
    def __init__(self, tau: float, dt: float, initial_value: Optional[np.ndarray] = None):
        """
        初始化低通滤波器。
        
        Args:
            tau: 时间常数（秒）。越小响应越快，但可能不够平滑。
                典型值：0.1-0.5秒
            dt: 采样时间间隔（秒）
            initial_value: 初始值。如果为None，第一次调用时会使用输入值
        """
        self.tau = tau
        self.dt = dt
        self.alpha = dt / (dt + tau)
        self._last_output = initial_value
        self._initialized = (initial_value is not None)
    
    def update(self, input_value: np.ndarray) -> np.ndarray:
        """
        更新滤波器并返回滤波后的值。
        
        Args:
            input_value: 输入值（可以是标量或数组）
        
        Returns:
            滤波后的输出值
        """
        input_value = np.asarray(input_value, dtype=float)
        
        if not self._initialized:
            # 第一次调用，直接使用输入值
            self._last_output = input_value.copy()
            self._initialized = True
            return self._last_output
        
        # 低通滤波：y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        self._last_output = self.alpha * input_value + (1 - self.alpha) * self._last_output
        
        return self._last_output.copy()
    
    def reset(self, value: Optional[np.ndarray] = None):
        """
        重置滤波器状态。
        
        Args:
            value: 重置后的值。如果为None，下次调用update时会重新初始化
        """
        self._last_output = value
        self._initialized = (value is not None)
    
    def get_last_output(self) -> Optional[np.ndarray]:
        """获取上一次的输出值"""
        return self._last_output.copy() if self._last_output is not None else None


def low_pass_filter(
    input_value: np.ndarray,
    last_output: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    简单的低通滤波函数（无状态版本）。
    
    Args:
        input_value: 当前输入值
        last_output: 上一次的输出值
        alpha: 滤波系数 (0 < alpha <= 1)
               越小越平滑，但响应越慢
               典型值：0.1-0.5
    
    Returns:
        滤波后的输出值
    """
    return alpha * input_value + (1 - alpha) * last_output

