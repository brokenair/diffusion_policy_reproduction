"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
# 注册一个新的解析器，用于在配置中执行任意的python代码，内置解析器不支持
# 算术运算，条件表达式这些，这些功能在config目录下的yaml模板文件里面会用到
OmegaConf.register_new_resolver("eval", eval, replace=True)

# 装饰器，拦截主函数，并传递配置参数
@hydra.main(
    # 关闭hydra的版本检查，也就是不检查环境里面的hydra版本
    version_base=None,
    # __file__是当前文件的绝对路径，这里完整的路径就是train同目录
    # 的diffusion_policy目录下面的config目录，这样理解就很简单了
    # .parent就相当于去掉文件名，得到根目录，然后拼接子目录
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    # 解析配置文件，替换掉所有的占位符，比如${now:}
    OmegaConf.resolve(cfg)
 
    # 根据配置文件中的_target_字段，获取对应的类
    cls = hydra.utils.get_class(cfg._target_)
    # 这里的冒号表示类型，删了没有影响
    # 创建工作空间对象并且把config对象传递给工作空间对象
    workspace: BaseWorkspace = cls(cfg)
    # 运行工作空间的run函数
    workspace.run()

if __name__ == "__main__":
    main()
