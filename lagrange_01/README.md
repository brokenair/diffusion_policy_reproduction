# Lagrange_01 机械臂控制器 - 开发目录

这是 Lagrange_01 机械臂的纯 Python 控制包开发目录。

## 项目状态

🚧 **开发中** - 核心功能正在实现和测试

## 目录结构

```
lagrange_01/
├── driver/                    # 电机驱动模块
│   ├── robstride.py          # RobStride 电机 SocketCAN 驱动
│   └── performance_test.py   # 性能测试脚本
├── scripts/                   # 脚本和工具
│   ├── circle_demo.py        # 圆形轨迹演示（已完成✓）
│   ├── find_joint_config.py  # IK 求解工具
│   ├── joint_setter.py       # 关节设置工具
│   ├── interactive_joint_tuner.py  # 交互式调试工具
│   └── mjcf_utils.py         # URDF 到 MJCF 转换
├── assets/                    # 资源文件
│   ├── urdf/                 # 机器人 URDF 模型
│   ├── meshes/               # 3D 网格文件
│   └── mjcf/                 # MuJoCo 模型
├── config/                    # 配置文件
│   └── motor_calibration.yaml  # 电机校准参数配置
├── legacy_hardware/           # C++ 参考代码
│   ├── include/              # C++ 头文件
│   └── src/                  # C++ 源文件
└── source_xacro/             # 原始 Xacro 文件（参考）

```

## 待完成功能

### 1. CAN 通信测试 ⏳
- [ ] 单电机连接测试
- [ ] 多电机同时通信
- [ ] 通信延迟测试
- [ ] 错误处理验证

### 2. 电机角度校准 ⏳
- [ ] 零点标定流程
- [ ] 角度映射校准
- [ ] 保存和加载校准数据
- [ ] 校准工具脚本

### 3. 轨迹生成和验证 ⏳
- [x] 圆形轨迹生成（已完成）
- [ ] 直线轨迹
- [ ] 样条曲线轨迹
- [ ] 轨迹平滑和插值
- [ ] 轨迹可视化

### 4. 控制方法 ⏳
- [ ] 关节空间控制
- [ ] 笛卡尔空间控制
- [ ] 阻抗控制
- [ ] 轨迹跟踪控制
- [ ] 控制器参数调优

### 5. 集成测试 ⏳
- [ ] 硬件在环测试
- [ ] 完整控制流程测试
- [ ] 性能基准测试
- [ ] 安全性测试

## 已完成功能 ✓

- ✓ 电机驱动 Python 移植（从 C++ 移植）
- ✓ 运动学模型（Pinocchio）
- ✓ 逆运动学求解（阻尼最小二乘法）
- ✓ 圆形轨迹演示（MuJoCo 可视化）
- ✓ 姿态控制（6DOF）
- ✓ 性能测试（验证 50Hz 控制频率）

## 开发环境

- Python >= 3.9
- numpy < 2.0 (Pinocchio 兼容性)
- mujoco >= 3.0
- pinocchio >= 3.0
- python-can >= 4.0

## 使用说明

### 运行圆形轨迹演示

```bash
cd /home/broken/Desktop/projects/ros2nmoveit/lagrange_01
python -m scripts.circle_demo
```

## 开发计划

1. **第一阶段**：完成 CAN 通信和电机校准
2. **第二阶段**：完善轨迹生成和控制方法
3. **第三阶段**：集成测试和性能优化
4. **第四阶段**：打包发布（功能完善后）

## 注意事项

- 这是**开发目录**，代码仍在测试和完善中
- 不要在生产环境使用未测试的功能
- 硬件测试时注意安全，设置好急停按钮
- 代码更改后建议先在仿真中验证

## 参考

- C++ 原始代码: `legacy_hardware/`
- 配置文件: `config/motor_calibration.yaml`
