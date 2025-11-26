# 安装 conda-pack
pip install -U conda-pack   # 用的是当前 Python，适用于 conda/mamba/micromamba 环境

# 打包名为 umi 的环境为单文件
conda-pack -n umi -o robodiff_env.tar.gz

# 另一台机器解包与修正路径，注意整体的压缩包自动解压了，
# 1) 进入到源代码目录，解到当前目录下的 .umi_env
cd diffusion_policy
PREFIX="$PWD/.robodiff_env"
mkdir -p "$PREFIX" && tar -xzf robodiff_env.tar.gz -C "$PREFIX"

# 2) 激活并修正路径
source "$PREFIX/bin/activate"
conda-unpack            # 或直接执行：$PREFIX/bin/conda-unpack

# wandb api key
67b26c03c3eb54d7e64b23f0cc951d1517bc93b0

# 这个运行会给目录添加任务名称，并且保存在data下面
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# 这个默认保存在根目录的outputs目录，并且不会命名
python train.py --config-dir=. --config-name=bf_lowdim_diffusion_unet.yaml training.seed=42 training.device=cuda:0

