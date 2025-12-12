#ifndef __ROBSTRIDE_SOCKETCAN_H__
#define __ROBSTRIDE_SOCKETCAN_H__

#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <atomic>
#include <functional>
#include <mutex>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* 1. 模式切换方面
 * 就是电机要切换模式的话，当前模式需要先disenable,然后
 * 再延时一些时间，目前测试30ms是可以的，但是注意，不是所有模式
 * 的切换都要这样，多测试一下，csp到pos好像不需要，其他没是过
 * 但是确实需要注意运控模式，这个模式和其他模式切换容易出现问题。
 *
 * 电机的模式似乎有bug,但是目前只要在切换模式之前disenable一下
 * 应该就没问题
 *
 * 2. 数据读取方面
 * 状态读取和寄存器读取是不一样的，寄存器读取只能得到寄存器，
 * 但是某些情况下可以得到完整的数据包，也就是电机状态信息，
 *
 * todo：
 * 1. 电机状态信息是通过发送什么读取的，只发送位置命令使用监控查看是不是会一直发送enable，会不会得到电机的角度等信息
 * 2. 看看只发送enable或者只发送控制，能否读取信息，能读取什么信息
 *
 * 如果新建一个函数，专门用来模式切换，并且添加模式读取来确认切换成功，这样不就不用在每个模式开始的时候检查模式了吗
 */

// 电机控制模式定义
#define SET_MODE        'j'
#define SET_PARAMETER   'p'

// 各种控制模式
#define MOVE_CONTROL_MODE   0   // 运动控制模式
#define POS_CONTROL_MODE    1   // PP位置模式
#define SPEED_CONTROL_MODE  2   // 速度模式
#define ELECT_CONTROL_MODE  3   // 电流模式
#define SET_ZERO_MODE       4   // 置零模式
#define CSP_CONTROL_MODE    5   // CSP位置模式

// 通讯地址定义
#define Communication_Type_Get_ID 0x00
#define Communication_Type_MotionControl 0x01
#define Communication_Type_MotorRequest 0x02
#define Communication_Type_MotorEnable 0x03
#define Communication_Type_MotorStop 0x04
#define Communication_Type_SetPosZero 0x06
#define Communication_Type_Can_ID 0x07
#define Communication_Type_Control_Mode 0x12
#define Communication_Type_GetSingleParameter 0x11
#define Communication_Type_SetSingleParameter 0x12
#define Communication_Type_ErrorFeedback 0x15
#define Communication_Type_MotorDataSave 0x16
#define Communication_Type_BaudRateChange 0x17
#define Communication_Type_ProactiveEscalationSet 0x18
#define Communication_Type_MotorModeSet 0x19


// 电机参数限制
#define P_MIN -12.5f
#define P_MAX 12.5f
#define V_MIN -33.0f
#define V_MAX 33.0f
#define KP_MIN 0.0f
#define KP_MAX 500.0f
#define KD_MIN 0.0f
#define KD_MAX 5.0f
#define T_MIN -14.0f
#define T_MAX 14.0f

// 数据读写结构
class data_read_write_one {
public:
    uint16_t index;
    float data;
};

static const uint16_t Index_List[] = {
    0X7005, 0X7006, 0X700A, 0X700B, 0X7010, 0X7011, 0X7014,
    0X7016, 0X7017, 0X7018, 0x7019, 0x701A, 0x701B, 0x701C, 0x701D
};

// 电机参数读写类
class data_read_write {
public:
    data_read_write_one run_mode{};        // 运行模式
    data_read_write_one iq_ref{};          // 电流模式Iq指令
    data_read_write_one spd_ref{};         // 转速模式转速指令
    data_read_write_one imit_torque{};     // 转矩限制
    data_read_write_one cur_kp{};          // 电流环 Kp
    data_read_write_one cur_ki{};          // 电流环 Ki
    data_read_write_one cur_filt_gain{};   // 电流滤波系数
    data_read_write_one loc_ref{};         // 位置模式角度指令
    data_read_write_one limit_spd{};       // 位置模式速度限制
    data_read_write_one limit_cur{};       // 速度位置模式电流限制
    // 只读参数
    data_read_write_one mechPos{};         // 输出端机械角度
    data_read_write_one iqf{};             // iq 滤波值
    data_read_write_one mechVel{};         // 输出端转速
    data_read_write_one VBUS{};            // 母线电压
    data_read_write_one rotation{};        // 圈数

    data_read_write(const uint16_t *index_list = Index_List);
};

// 电机位置信息结构
typedef struct {
    float Angle;     // 角度
    float Speed;     // 速度
    float Torque;    // 扭矩
    float Temp;      // 温度
    // 0 = Reset（复位状态）、1 = Cali（标定/校准）、2 = Motor（正常运行/运控）
    int Pattern;     // 运行模式
} Motor_Pos_RobStride_Info;

// 电机设置结构
typedef struct {
    int set_motor_mode;
    float set_current;
    float set_speed;
    float set_acceleration;
    float set_Torque;
    float set_angle;
    float set_limit_cur;
    float set_limit_speed;
    float set_Kp;
    float set_Ki;
    float set_Kd;
} Motor_Set;

// RobStride 电机类（使用 SocketCAN）
class RobStride_Motor_SocketCAN {
private:
    uint8_t CAN_ID;                      // CAN ID
    uint64_t Unique_ID;                  // 64位MCU唯一标识码
    uint16_t Master_CAN_ID;              // 主机ID
    float (*Motor_Offset_MotoFunc)(float Motor_Tar);  // 电机偏移函数指针

    // SocketCAN
    int sockfd;                          // CAN socket文件描述符
    std::string can_interface;           // CAN接口名称

    // 接收线程相关
    std::thread receive_thread;
    std::atomic<bool> receiving;
    std::atomic<bool> connected;

    // 回调函数
    std::function<void(uint32_t id, uint8_t* data, uint8_t length)> receive_callback;

    Motor_Set Motor_Set_All;             // 设定值
    uint8_t error_code;                  // 错误代码

    // 互斥锁保护共享数据
    mutable std::mutex data_mutex;       // 保护电机状态数据的互斥锁

    // 私有方法
    bool init_socketcan();              // 初始化 SocketCAN
    void cleanup_socketcan();           // 清理资源
    void receive_worker();              // 接收线程工作函数
    bool can_tx_ext(uint32_t id, uint8_t* data, uint8_t length);  // 发送CAN消息（扩展ID）
    bool can_tx_std(uint32_t id, uint8_t* data, uint8_t length);  // 发送CAN消息（标准ID）

    // 数据转换函数
    float uint16_to_float(uint16_t x, float x_min, float x_max, int bits);
    int float_to_uint(float x, float x_min, float x_max, int bits);
    float Byte_to_float(uint8_t* data);
    uint8_t mapFaults(uint16_t fault16);


public:
    float output;                        // 输出变量（用于电流控制等）
    Motor_Pos_RobStride_Info Pos_Info;   // 返回值
    data_read_write drw;                 // 电机参数

    // 构造函数和析构函数
    RobStride_Motor_SocketCAN(uint8_t CAN_Id, const std::string& interface = "can0");
    RobStride_Motor_SocketCAN(uint8_t CAN_Id, const std::string& interface, float (*Offset_MotoFunc)(float Motor_Tar));
    ~RobStride_Motor_SocketCAN();

    // 基础通信方法
    bool initialize();                   // 初始化设备
    void set_receive_callback(std::function<void(uint32_t, uint8_t*, uint8_t)> callback);

    // 电机控制方法
    void Enable_Motor();                 // 使能电机
    void Disenable_Motor(uint8_t clear_error = 0);  // 失能电机
    void Set_ZeroPos();                  // 设置零位

    // 参数读写方法
    void Set_RobStride_Motor_parameter(uint16_t Index, float Value, char Value_mode);
    void Get_RobStride_Motor_parameter(uint16_t Index);

    // 各种控制模式
    void RobStride_Motor_move_control(float Torque, float Angle, float Speed, float Kp, float Kd);
    void RobStride_Motor_Pos_control(float Angle, float Speed);
    void RobStride_Motor_Speed_control(float Speed, float limit_cur);
    void RobStride_Motor_current_control(float current);
    void RobStride_Motor_CSP_control(float Angle, float limit_spd);
    void RobStride_Motor_Set_Zero_control();


    // CAN ID 相关方法
    void RobStride_Get_CAN_ID();
    void Set_CAN_ID(uint8_t Set_CAN_ID);

    // 电机配置方法
    void RobStride_Motor_MotorDataSave();
    void RobStride_Motor_BaudRateChange(uint8_t F_CMD);
    void RobStride_Motor_ProactiveEscalationSet(uint8_t F_CMD);
    void RobStride_Motor_MotorModeSet(uint8_t F_CMD);

    // 数据解析方法
    void RobStride_Motor_Analysis(uint8_t* DataFrame, uint32_t ID_ExtId);

    // 获取电机状态
    bool is_connected() const { return connected; }
    Motor_Pos_RobStride_Info get_motor_info() const;
};

#endif // __ROBSTRIDE_SOCKETCAN_H__