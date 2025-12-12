#include "RobStride_SocketCAN.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <iomanip>
#include <csignal>
#include <atomic>

// 全局变量控制程序运行
std::atomic<bool> running(true);

// 信号处理函数
void signal_handler(int) {
    std::cout << "\n收到停止信号，正在退出..." << std::endl;
    running = false;
}

// 正弦波测试参数
const float PI = 3.1415926535f;
const float SINE_AMPLITUDE_DEG = 30.0f;                                   // 目标幅值
const float SINE_AMPLITUDE_RAD = (SINE_AMPLITUDE_DEG * PI / 180.0f);      // 转换为弧度
const float SINE_FREQUENCY_HZ = 0.3f;                                      // 目标频率

// 控制频率
const int CONTROL_FREQ_HZ = 200;
const auto CONTROL_PERIOD = std::chrono::microseconds(1000000 / CONTROL_FREQ_HZ);

// CAN消息接收回调函数
void can_receive_callback(uint32_t, uint8_t*, uint8_t) {
    // 这里可以添加额外的消息处理逻辑
    // 主要的消息处理已经在 RobStride_Motor_Analysis 中完成
}

int main() {
    // 注册信号处理函数
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 创建电机对象（使用can0接口，标准运动控制模式）
    RobStride_Motor_SocketCAN motor1(0x01, "can0");
    RobStride_Motor_SocketCAN motor2(0x02, "can0");
    RobStride_Motor_SocketCAN motor3(0x03, "can0");
    RobStride_Motor_SocketCAN motor4(0x04, "can0");
    RobStride_Motor_SocketCAN motor5(0x05, "can0");
    RobStride_Motor_SocketCAN motor6(0x06, "can0");

    // 初始化电机，这里一定是或，如果是与的话只会初始化一个电机，别弄错了逻辑！！！
    if (!motor1.initialize() || !motor2.initialize() || !motor3.initialize() ||
        !motor4.initialize() || !motor5.initialize() || !motor6.initialize()) {
        std::cerr << "电机初始化失败！" << std::endl;
        return -1;
        }

    // 设置接收回调，目前没用
    //motor1.set_receive_callback(can_receive_callback);

    // 不要使能电机，直接发送模式即可，要不然容易出问题，换不了模式，最好是失能电机
    motor1.Disenable_Motor();
    motor2.Disenable_Motor();
    motor3.Disenable_Motor();
    motor4.Disenable_Motor();
    motor5.Disenable_Motor();
    motor6.Disenable_Motor();

    // motor6.RobStride_Motor_move_control(0.0f, 1, 10, 10, 1);
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // //motor5.Disenable_Motor();
    // motor6.Disenable_Motor();
    // std::this_thread::sleep_for(std::chrono::milliseconds(30));
    // motor6.RobStride_Motor_CSP_control(0, 6);
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //
    // motor6.Disenable_Motor();
    // // motor6.Get_RobStride_Motor_parameter(0x7005);
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));


    // std::cout << "\n开始运动控制模式正弦波轨迹跟踪..." << std::endl;
    // std::cout << "时间(s)   目标位置(°)  实际位置(°)  实际速度(°/s)  扭矩(Nm)  温度(°C) " << std::endl;
    // std::cout << "----------------------------------------------------------------" << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    auto next_control_time = start_time;

    int loop_count = 0;
    const int PRINT_INTERVAL = CONTROL_FREQ_HZ / 5; // 每秒打印5次

    while (running) {
        auto current_time = std::chrono::steady_clock::now();

        // 等待到下一个控制周期
        if (current_time < next_control_time) {
            std::this_thread::sleep_until(next_control_time);
            current_time = next_control_time;
        }

        next_control_time += CONTROL_PERIOD;

        // 计算时间（秒）
        auto elapsed = current_time - start_time;
        float time_s = std::chrono::duration<float>(elapsed).count();

        // 计算当前时刻的目标位置和速度
        // 目标位置 p(t) = A * sin(2*pi*f*t)
        float target_pos = SINE_AMPLITUDE_RAD * sinf(2.0f * PI * SINE_FREQUENCY_HZ * time_s);
        // 目标速度 v(t) = A * 2*pi*f * cos(2*pi*f*t)
        float target_vel = SINE_AMPLITUDE_RAD * 2.0f * PI * SINE_FREQUENCY_HZ *
            cosf(2.0f * PI * SINE_FREQUENCY_HZ * time_s);

        // 发送运动控制指令
        // motor.RobStride_Motor_move_control(0.0f, target_pos, target_vel, KP, KD);
        // motor.RobStride_Motor_CSP_control(target_pos, 6);

        // motor1.RobStride_Motor_Pos_control(target_pos*0.1 + 3.14, 0.03);
        // motor2.RobStride_Motor_Pos_control(target_pos*0.1 + 1.93, 0.03);
        // motor3.RobStride_Motor_Pos_control(target_pos*0.1 + 2.21, 0.03);
        // motor4.RobStride_Motor_Pos_control(target_pos+ 3.14, target_vel);
        // motor5.RobStride_Motor_Pos_control(target_pos + 3.6, target_vel);
        // motor6.RobStride_Motor_Pos_control(target_pos, target_vel);

        // 定期请求电机状态
        if (true) {
            motor1.Get_RobStride_Motor_parameter(0x7019); // 机械角度
            motor2.Get_RobStride_Motor_parameter(0x7019); // 机械角度
            motor3.Get_RobStride_Motor_parameter(0x7019); // 机械角度
            motor4.Get_RobStride_Motor_parameter(0x7019); // 机械角度
            motor5.Get_RobStride_Motor_parameter(0x7019); // 机械角度
            motor6.Get_RobStride_Motor_parameter(0x7019); // 机械角度
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // 定期打印状态信息
        if (true) {
            Motor_Pos_RobStride_Info info1 = motor1.get_motor_info();
            Motor_Pos_RobStride_Info info2 = motor2.get_motor_info();
            Motor_Pos_RobStride_Info info3 = motor3.get_motor_info();
            Motor_Pos_RobStride_Info info4 = motor4.get_motor_info();
            Motor_Pos_RobStride_Info info5 = motor5.get_motor_info();
            Motor_Pos_RobStride_Info info6 = motor6.get_motor_info();

            std::cout << std::fixed << std::setprecision(2)
                     << std::setw(12) << time_s
                     << std::setw(12) << info1.Angle * 180.0f / PI
                     << std::setw(14) << info2.Angle * 180.0f / PI
                     << std::setw(10) << info3.Angle * 180.0f / PI
                     << std::setw(12) << info4.Angle * 180.0f / PI
                     << std::setw(14) << info5.Angle * 180.0f / PI
                     << std::setw(10) << info6.Angle * 180.0f / PI
                     << std::setw(10) << info6.Pattern
                     << std::endl;
        }

        loop_count++;
    }

    std::cout << "\n停止测试，失能电机..." << std::endl;

    // 失能电机
    // motor1.Disenable_Motor();
    // motor2.Disenable_Motor();
    // motor3.Disenable_Motor();
    // motor4.Disenable_Motor();
    // motor5.Disenable_Motor();
    motor6.Disenable_Motor();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "测试结束。" << std::endl;

    return 0;
}