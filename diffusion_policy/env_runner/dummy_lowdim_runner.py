from typing import Dict, Any
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class DummyLowdimRunner(BaseLowdimRunner):
    """
    纯占位用的低维 runner，不做任何 rollout，只返回一个固定分数。
    """

    def __init__(self,
                 output_dir: str,
                 **kwargs):
        # 把 output_dir 传给父类
        super().__init__(output_dir=output_dir)
        # 可选：自己再存一遍
        self.output_dir = output_dir

    def run(self, policy) -> Dict[str, Any]:
        # 占位：返回一个固定的评估结果
        return {
            "test_mean_score": 0.0
        }
