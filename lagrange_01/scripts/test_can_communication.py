"""
CAN 通信测试脚本 - Python 版本
原始 C++ 代码的直接翻译，用于测试 RobStride 电机通信
"""

from __future__ import annotations

import time
import signal
import sys
import math
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.robstride import RobStrideMotor


# 全局变量控制程序运行
running = True


# 信号处理函数
def signal_handler(signum, frame):
    global running
    print("\n收到停止信号，正在退出...")
    running = False


# 正弦波测试参数
PI = 3.1415926535
SINE_AMPLITUDE_DEG = 30.0                                   # 目标幅值
SINE_AMPLITUDE_RAD = (SINE_AMPLITUDE_DEG * PI / 180.0)      # 转换为弧度
SINE_FREQUENCY_HZ = 0.3                                      # 目标频率


# 控制频率
CONTROL_FREQ_HZ = 100
CONTROL_PERIOD = 1.0 / CONTROL_FREQ_HZ


# CAN消息接收回调函数
def can_receive_callback(can_id: int, data: bytes):
    # 这里可以添加额外的消息处理逻辑
    # 主要的消息处理已经在 RobStride_Motor_Analysis 中完成
    pass


def main():
    global running
    
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建电机对象（使用can0接口，标准运动控制模式）
    motor1 = RobStrideMotor(0x01, "can0")
    motor2 = RobStrideMotor(0x02, "can0")
    motor3 = RobStrideMotor(0x03, "can0")
    motor4 = RobStrideMotor(0x04, "can0")
    motor5 = RobStrideMotor(0x05, "can0")
    motor6 = RobStrideMotor(0x06, "can0")
    
    # 初始化电机，这里一定是或，如果是与的话只会初始化一个电机，别弄错了逻辑！！！
    if not motor1.initialize() or not motor2.initialize() or not motor3.initialize() or \
       not motor4.initialize() or not motor5.initialize() or not motor6.initialize():
        print("电机初始化失败！", file=sys.stderr)
        return -1
    
    # 设置接收回调，目前没用
    # motor1.set_receive_callback(can_receive_callback)
    
    # 不要使能电机，直接发送模式即可，要不然容易出问题，换不了模式，最好是失能电机
    motor1.disable()
    motor2.disable()
    motor3.disable()
    motor4.disable()
    motor5.disable()
    motor6.disable()
    # motor6.enable()
    
    # motor6.move_control(0.0, 1, 10, 10, 1)
    # time.sleep(1.0)
    # # motor5.disable()
    # motor6.disable()
    # time.sleep(0.03)
    # motor6.RobStride_Motor_CSP_control(0, 6)
    # time.sleep(1.0)
    #
    # motor6.disable()
    # # motor6.get_parameter(0x7005)
    # time.sleep(1.0)
    
    
    # print("\n开始运动控制模式正弦波轨迹跟踪...")
    # print("时间(s)   目标位置(°)  实际位置(°)  实际速度(°/s)  扭矩(Nm)  温度(°C) ")
    # print("----------------------------------------------------------------")
    
    start_time = time.time()
    next_control_time = start_time
    
    loop_count = 0
    PRINT_INTERVAL = CONTROL_FREQ_HZ // 5  # 每秒打印5次
    
    print("\n开始监控电机状态（每秒更新5次）...")
    print("按 Ctrl+C 停止")
    print("")  # 空行，用于后续覆盖
    
    while running:
        current_time = time.time()
        
        # 等待到下一个控制周期
        if current_time < next_control_time:
            time.sleep(next_control_time - current_time)
            current_time = next_control_time
        
        next_control_time += CONTROL_PERIOD
        
        # 计算时间（秒）
        time_s = current_time - start_time
        
        # 计算当前时刻的目标位置和速度
        # 目标位置 p(t) = A * sin(2*pi*f*t)
        target_pos = SINE_AMPLITUDE_RAD * math.sin(2.0 * PI * SINE_FREQUENCY_HZ * time_s)
        # 目标速度 v(t) = A * 2*pi*f * cos(2*pi*f*t)
        target_vel = SINE_AMPLITUDE_RAD * 2.0 * PI * SINE_FREQUENCY_HZ * \
            math.cos(2.0 * PI * SINE_FREQUENCY_HZ * time_s)
        
        # 发送运动控制指令
        # motor.move_control(0.0, target_pos, target_vel, KP, KD)
        # motor.RobStride_Motor_CSP_control(target_pos, 6)
        
        # motor1.pos_control(target_pos*0.1 + 3.14, 0.03)
        # motor2.pos_control(target_pos*0.1 + 1.93, 0.03)
        # motor3.pos_control(target_pos*0.1 + 2.21, 0.03)
        # motor4.pos_control(target_pos + 3.14, target_vel)
        # motor5.pos_control(target_pos + 3.6, target_vel)
        # motor6.pos_control(angle=target_pos)
        # motor6.csp_control(angle=target_pos, limit_speed=5.0)
        # 用这个模式记得使能电机
        motor6.move_control(torque=0.0, angle=target_pos, speed=target_vel, kp=10.0, kd=0.5)
        #motor6.speed_control(speed=1.0, limit_cur=1.0)
        
        # 定期请求电机状态
        if True:
            motor1.get_parameter(0x7019)  # 机械角度
            motor2.get_parameter(0x7019)  # 机械角度
            motor3.get_parameter(0x7019)  # 机械角度
            motor4.get_parameter(0x7019)  # 机械角度
            motor5.get_parameter(0x7019)  # 机械角度
            # motor5.enable()
            # motor6.get_parameter(0x7019)  # 机械角度
        
        time.sleep(0.001)
        
        # 定期打印状态信息（每秒5次）
        if loop_count % PRINT_INTERVAL == 0:
            info1 = motor1.get_info()
            info2 = motor2.get_info()
            info3 = motor3.get_info()
            info4 = motor4.get_info()
            info5 = motor5.get_info()
            info6 = motor6.get_info()
            
            # 使用 ANSI 转义序列：\033[A 上移一行，\r 回到行首，\033[K 清除到行尾
            output = (f"T:{time_s:5.1f}s | "
                     f"M1:{info1.angle * 180.0 / PI:6.1f} "
                     f"M2:{info2.angle * 180.0 / PI:6.1f} "
                     f"M3:{info3.angle * 180.0 / PI:6.1f} "
                     f"M4:{info4.angle * 180.0 / PI:6.1f} "
                     f"M5:{info5.angle * 180.0 / PI:6.1f} "
                     f"M6:{info6.angle * 180.0 / PI:6.1f} "
                     f"M5_s:{info5.speed:6.1f}"
                     f"M6_s:{info6.speed:6.1f}")
            sys.stdout.write(f"\033[A\r{output}\033[K\n")
            sys.stdout.flush()
        
        loop_count += 1
    
    print("\n\n停止测试，失能电机...")
    
    # 失能电机
    # motor1.disable()
    # motor2.disable()
    # motor3.disable()
    # motor4.disable()
    # motor5.disable()
    motor6.disable()
    time.sleep(0.1)
    
    print("测试结束。")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

