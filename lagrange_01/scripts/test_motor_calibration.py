"""
测试电机校准参数的正确性 (Motor-to-Joint Calibration Test)

工作流程：
1. 从真实电机读取原始编码器角度 (Motor space / Hardware space)
2. 应用校准参数转换为标准关节角度 (Joint space / URDF space)
3. 使用MuJoCo实时可视化机器人姿态

术语说明：
- Motor angle: 电机编码器的原始读数（硬件坐标系）
- Joint angle: 经过校准转换的关节角度（标准坐标系/URDF坐标系）
- Motor-to-Joint: 从硬件读数到标准角度的转换（应用校准参数）
- Joint-to-Motor: 从标准角度到硬件指令的转换（逆向转换）

注意：
- 本脚本是纯读取模式（不发送控制命令），需要主动调用 get_parameter() 查询角度
- 在实际控制循环中，发送控制命令后电机会自动返回状态数据包，无需额外查询
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.robstride import RobStrideMotor
from scripts.robot_controller import RobotController
from utils.mjcf_utils import write_mjcf


# 路径配置
ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets" / "urdf" / "Lagrange_01_sim.urdf"
MESH_DIR = (ROOT / "assets" / "meshes").resolve()
MJCF_PATH = ROOT / "assets" / "mjcf" / "lagrange_generated.xml"
CONFIG_PATH = ROOT / "config" / "motor_calibration.yaml"
EE_FRAME = "tool_link"  # 末端执行器frame名称


def main():
    """主函数：读取电机角度，应用校准参数转换为关节角度，MuJoCo可视化"""
    
    print("=" * 60)
    print("电机校准参数测试 (Motor Calibration Test)")
    print("测试 Motor Space → Joint Space 转换的正确性")
    print("=" * 60)
    
    # 1. 初始化MuJoCo模型
    print("\n[1/4] 初始化MuJoCo模型...")
    MJCF_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_mjcf(URDF_PATH, MESH_DIR, MJCF_PATH)
    
    mj_model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    mj_data = mujoco.MjData(mj_model)
    print(f"  ✓ MuJoCo模型加载成功 (DOF: {mj_model.nq})")
    
    # 2. 初始化Pinocchio模型（用于前向运动学）
    print("\n[2/4] 初始化Pinocchio模型...")
    pin_model = pin.buildModelFromUrdf(str(URDF_PATH))
    pin_data = pin_model.createData()
    ee_frame_id = pin_model.getFrameId(EE_FRAME)
    if ee_frame_id >= len(pin_model.frames):
        raise SystemExit(f"Frame '{EE_FRAME}' not found in URDF")
    print(f"  ✓ Pinocchio模型加载成功 (DOF: {pin_model.nq}, Frame: {EE_FRAME})")
    
    # 3. 连接电机
    print("\n[3/4] 连接电机...")
    controller = RobotController.from_config(CONFIG_PATH)
    
    try:
        controller.connect()
        print("  ✓ 所有电机连接成功")
        
        # 4. 启动实时可视化
        print("\n[4/4] 启动实时可视化...")
        print("\n" + "=" * 60)
        print("实时可视化说明：")
        print("  - 黄色数值：Motor angles - 电机编码器原始读数（硬件空间）")
        print("  - 绿色数值：Joint angles - 校准后的关节角度（URDF空间）")
        print("  - 青色数值：Motor temperatures - 电机温度")
        print("  - 蓝色数值：End-effector pose - 末端执行器位姿（位置+姿态）")
        print("  - MuJoCo窗口：根据Joint angles实时显示机器人姿态")
        print("")
        print("验证方法：")
        print("  1. 手动移动机械臂到某个姿态")
        print("  2. 观察MuJoCo中的可视化是否与真实姿态一致")
        print("  3. 如果一致，说明校准参数（direction, offset）配置正确")
        print("")
        print("  按 Ctrl+C 退出")
        print("=" * 60 + "\n")
        
        # 等待电机初始化
        print("等待电机初始化...")
        time.sleep(0.5)
        
        # 首次读取角度，确保数据有效
        _ = controller.get_joint_positions(request_update=True)
        print("初始化完成，开始实时可视化...\n")
        
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            last_update = time.time()
            update_interval = 0.05  # 每50ms更新一次（20Hz）
            
            while viewer.is_running():
                current_time = time.time()
                
                # 定期更新显示
                if current_time - last_update >= update_interval:
                    try:
                        # 读取并转换角度 (Motor space → Joint space)
                        # request_update=True: 主动请求电机发送数据（纯读取模式）
                        joint_angles = controller.get_joint_positions(request_update=True)
                        
                        # 读取原始电机数据用于显示和验证
                        motor_angles = []
                        motor_temps = []
                        for joint_cfg in controller.joint_configs:
                            motor = controller.motors[joint_cfg.name]
                            info = motor.get_info()
                            motor_angles.append(info.angle)  # Raw motor encoder reading
                            motor_temps.append(info.temp)
                        motor_angles = np.array(motor_angles)
                        motor_temps = np.array(motor_temps)
                        
                        # 更新MuJoCo可视化（使用校准后的关节角度）
                        mj_data.qpos[:mj_model.nq] = joint_angles
                        mj_data.qvel[:] = 0.0
                        mujoco.mj_forward(mj_model, mj_data)
                        
                        # 计算末端执行器位姿（使用Pinocchio前向运动学）
                        pin.forwardKinematics(pin_model, pin_data, joint_angles)
                        pin.updateFramePlacements(pin_model, pin_data)
                        ee_pose = pin_data.oMf[ee_frame_id]
                        ee_position = ee_pose.translation
                        ee_rotation = ee_pose.rotation
                        
                        # 将旋转矩阵转换为RPY（ZYX顺序）用于显示
                        # 从旋转矩阵提取欧拉角：R = Rz(yaw) * Ry(pitch) * Rx(roll)
                        sy = np.sqrt(ee_rotation[0, 0]**2 + ee_rotation[1, 0]**2)
                        singular = sy < 1e-6
                        if not singular:
                            roll = np.arctan2(ee_rotation[2, 1], ee_rotation[2, 2])
                            pitch = np.arctan2(-ee_rotation[2, 0], sy)
                            yaw = np.arctan2(ee_rotation[1, 0], ee_rotation[0, 0])
                        else:
                            roll = np.arctan2(-ee_rotation[1, 2], ee_rotation[1, 1])
                            pitch = np.arctan2(-ee_rotation[2, 0], sy)
                            yaw = 0.0
                        rpy = np.array([roll, pitch, yaw])
                        rpy_deg = np.degrees(rpy)
                        
                        # 验证校准参数正确性：反向计算应该恢复原始电机角度
                        # Joint → Motor → Joint 的往返误差应接近0
                        angle_diffs = []
                        for i, (joint_cfg, motor_angle) in enumerate(zip(controller.joint_configs, motor_angles)):
                            # Joint-to-Motor: 从关节角度反算电机角度
                            expected_motor = joint_cfg.joint_to_motor(joint_angles[i])
                            diff = abs(motor_angle - expected_motor)
                            angle_diffs.append(diff)
                        
                        # 格式化输出（固定4行显示）
                        motor_str = "  ".join([f"M{i+1}:{a:6.2f}" for i, a in enumerate(motor_angles)])
                        joint_str = "  ".join([f"J{i+1}:{a:6.2f}" for i, a in enumerate(joint_angles)])
                        temp_str = "  ".join([f"{t:4.1f}°C" for t in motor_temps])
                        pos_str = f"Pos:[{ee_position[0]:6.3f}, {ee_position[1]:6.3f}, {ee_position[2]:6.3f}]"
                        rpy_str = f"RPY:[Roll:{rpy_deg[0]:6.1f}° Pitch:{rpy_deg[1]:6.1f}° Yaw:{rpy_deg[2]:6.1f}°]"
                        
                        # 固定位置更新（使用ANSI转义序列上移4行）
                        if last_update != 0:  # 跳过第一次
                            sys.stdout.write("\033[A\033[A\033[A\033[A")  # 上移四行
                        
                        print(f"\r\033[K\033[33mMotor angles (rad): {motor_str}\033[0m")  # 黄色 - 原始读数
                        print(f"\r\033[K\033[32mJoint angles (rad): {joint_str}\033[0m")  # 绿色 - 校准角度
                        print(f"\r\033[K\033[36mMotor temps:        {temp_str}\033[0m")  # 青色 - 温度
                        print(f"\r\033[K\033[34mEnd-effector pose:  {pos_str}  {rpy_str}\033[0m")  # 蓝色 - 末端位姿
                        sys.stdout.flush()
                        
                        last_update = current_time
                        
                    except Exception as e:
                        print(f"\r\033[K错误: {e}")
                        import traceback
                        traceback.print_exc()
                        break
                
                # 同步可视化
                viewer.sync()
                
                # 控制循环频率
                time.sleep(0.001)
        
        print("\n\n可视化结束")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭电机连接
        print("\n关闭电机连接...")
        controller.disable_all()
        controller.close()
        print("测试完成！")


if __name__ == "__main__":
    main()

