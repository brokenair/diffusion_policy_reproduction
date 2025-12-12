# Lagrange-01 开发进度

## 当前状态：CAN 通信测试阶段

### ✅ 已完成
- [x] URDF 模型定义
- [x] MuJoCo 仿真环境（circle_demo.py）
- [x] Pinocchio 正向/逆向运动学
- [x] RobStride 电机底层驱动（driver/robstride.py）
- [x] 电机校准参数配置文件（config/motor_calibration.yaml）

### 🔄 当前任务：测试 CAN 通信
- [ ] 运行 `test_can_communication.py` 验证电机通信
- [ ] 确认所有 6 个电机能正常初始化
- [ ] 验证角度读取功能
- [ ] 测试电机使能/失能功能

### ⏳ 待完成（通信测试后）
- [ ] 完善 `robot_controller.py`（整合 Pinocchio + 电机控制）
- [ ] 编写实际机器人 IK 控制演示
- [ ] 整理文档和使用说明
- [ ] 性能测试和优化

---

## 文件结构

```
lagrange_01/
├── assets/              # 机器人模型资源
│   ├── urdf/           # URDF 模型
│   └── meshes/         # STL 网格文件
├── config/             # 配置文件
│   └── motor_calibration.yaml  # 电机校准参数 ⭐
├── driver/             # 底层驱动（不依赖 Pinocchio）
│   ├── robstride.py    # RobStride 电机 CAN 驱动 ⭐
│   └── __init__.py
└── scripts/            # 应用层脚本（依赖 Pinocchio）
    ├── test_can_communication.py  # CAN 通信测试 ⭐ 当前重点
    ├── circle_demo.py             # 仿真演示（已完成）
    ├── robot_controller.py        # 高层控制器（待完善）
    └── mjcf_utils.py              # MJCF 工具
```

---

## 测试步骤

### 1. 启动 CAN 接口
```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

### 2. 运行通信测试
```bash
python3 -m lagrange_01.scripts.test_can_communication
```

### 3. 预期输出
```
时间(s)      电机1角度(°)  电机2角度(°)  ...
     0.00       xxx.xx        xxx.xx  ...
     0.20       xxx.xx        xxx.xx  ...
```

### 4. 停止测试
按 `Ctrl+C` 安全退出

---

## 注意事项
- ⚠️ 当前阶段只测试通信，不进行实际运动
- ⚠️ 脚本中的运动控制代码已注释，确保安全
- ⚠️ 测试前确认机器人处于安全位置

