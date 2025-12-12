#include "my_robot_hardware/RobStride_SocketCAN.h"
#include <cmath>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>

/*******************************************************************************
* 数据读写结构构造函数 - 基于原始库实现
*******************************************************************************/
data_read_write::data_read_write(const uint16_t *index_list) {
    // 根据原始库实现，使用index_list初始化index值
    run_mode.index = index_list[0];
    iq_ref.index = index_list[1];
    spd_ref.index = index_list[2];
    imit_torque.index = index_list[3];
    cur_kp.index = index_list[4];
    cur_ki.index = index_list[5];
    cur_filt_gain.index = index_list[6];
    loc_ref.index = index_list[7];
    limit_spd.index = index_list[8];
    limit_cur.index = index_list[9];
    mechPos.index = index_list[10];
    iqf.index = index_list[11];
    mechVel.index = index_list[12];
    VBUS.index = index_list[13];
    rotation.index = index_list[14];

    // 初始化所有数据为0
    run_mode.data = 0;
    iq_ref.data = 0;
    spd_ref.data = 0;
    imit_torque.data = 0;
    cur_kp.data = 0;
    cur_ki.data = 0;
    cur_filt_gain.data = 0;
    loc_ref.data = 0;
    limit_spd.data = 0;
    limit_cur.data = 0;
    mechPos.data = 0;
    iqf.data = 0;
    mechVel.data = 0;
    VBUS.data = 0;
    rotation.data = 0;
}

/*******************************************************************************
* 初始化 SocketCAN
*******************************************************************************/
bool RobStride_Motor_SocketCAN::init_socketcan() {
    struct sockaddr_can addr;
    struct ifreq ifr;

    // 创建socket
    sockfd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sockfd < 0) {
        std::cerr << "创建CAN socket失败" << std::endl;
        return false;
    }

    // 设置非阻塞模式
    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

    // 指定CAN接口
    strcpy(ifr.ifr_name, can_interface.c_str());
    if (ioctl(sockfd, SIOCGIFINDEX, &ifr) < 0) {
        std::cerr << "获取接口" << can_interface << "索引失败，请确认接口已启动" << std::endl;
        close(sockfd);
        sockfd = -1;
        return false;
    }

    // 绑定socket到CAN接口
    memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        std::cerr << "绑定CAN接口失败" << std::endl;
        close(sockfd);
        sockfd = -1;
        return false;
    }

    std::cout << "SocketCAN 初始化成功，接口: " << can_interface << std::endl;
    return true;
}

/*******************************************************************************
* 清理资源
*******************************************************************************/
void RobStride_Motor_SocketCAN::cleanup_socketcan() {
    receiving = false;
    connected = false;

    if (receive_thread.joinable()) {
        receive_thread.join();
    }

    if (sockfd >= 0) {
        close(sockfd);
        sockfd = -1;
    }
}

/*******************************************************************************
* 接收线程工作函数
*******************************************************************************/
void RobStride_Motor_SocketCAN::receive_worker() {
    struct can_frame frame;

    while (receiving) {
        ssize_t nbytes = read(sockfd, &frame, sizeof(frame));

        if (nbytes > 0) {
            // 剥离SocketCAN标志位，获取纯CAN ID
            uint32_t clean_id = frame.can_id & CAN_EFF_MASK;

            // 调用消息分析函数
            RobStride_Motor_Analysis(frame.data, clean_id);

            // 调用用户回调（也使用剥离后的ID）
            if (receive_callback) {
                receive_callback(clean_id, frame.data, frame.can_dlc);
            }
        }

        // 短暂休眠避免CPU占用过高
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

/*******************************************************************************
* 发送 CAN 消息
*******************************************************************************/
bool RobStride_Motor_SocketCAN::can_tx_ext(uint32_t id, uint8_t* data, uint8_t length) {
    if (sockfd < 0) {
        return false;
    }

    struct can_frame frame;
    frame.can_id = id | CAN_EFF_FLAG;  // 扩展帧
    // 先截断length，再赋值给can_dlc，避免非法DLC
    if (length > 8) length = 8;
    frame.can_dlc = length;
    memcpy(frame.data, data, length);

    ssize_t nbytes = write(sockfd, &frame, sizeof(frame));
    return (nbytes == sizeof(frame));
}

/*******************************************************************************
* 发送标准ID CAN消息 - 用于MIT模式
*******************************************************************************/
bool RobStride_Motor_SocketCAN::can_tx_std(uint32_t id, uint8_t* data, uint8_t length) {
    if (sockfd < 0) {
        return false;
    }

    struct can_frame frame;
    frame.can_id = id;  // 标准帧（不加CAN_EFF_FLAG）
    // 先截断length，再赋值给can_dlc，避免非法DLC
    if (length > 8) length = 8;
    frame.can_dlc = length;
    memcpy(frame.data, data, length);

    ssize_t nbytes = write(sockfd, &frame, sizeof(frame));
    return (nbytes == sizeof(frame));
}

/*******************************************************************************
* 数据转换函数
*******************************************************************************/
float RobStride_Motor_SocketCAN::uint16_to_float(uint16_t x, float x_min, float x_max, int bits) {
    uint32_t span = (1 << bits) - 1;
    x &= span;
    float offset = x_max - x_min;
    return offset * x / span + x_min;
}

int RobStride_Motor_SocketCAN::float_to_uint(float x, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x_min;
    if (x > x_max) x = x_max;
    else if (x < x_min) x = x_min;
    return (int)((x - offset) * ((float)((1 << bits) - 1)) / span);
}

float RobStride_Motor_SocketCAN::Byte_to_float(uint8_t* data) {
    // 基于原始库实现：组合字节7-4为32位整数再转换为float
    uint32_t uint_data = ((uint32_t)data[7] << 24) | ((uint32_t)data[6] << 16) |
                         ((uint32_t)data[5] << 8) | ((uint32_t)data[4]);
    float result;
    std::memcpy(&result, &uint_data, sizeof(float));
    return result;
}

uint8_t RobStride_Motor_SocketCAN::mapFaults(uint16_t fault16) {
    // 基于原始库实现：将16位故障码映射到8位
    uint8_t fault8 = 0;

    if (fault16 & (1 << 14)) fault8 |= (1 << 4); // 过温故障
    if (fault16 & (1 << 7))  fault8 |= (1 << 5); // 未标定
    if (fault16 & (1 << 3))  fault8 |= (1 << 3); // 编码器错误
    if (fault16 & (1 << 2))  fault8 |= (1 << 0); // 低压故障
    if (fault16 & (1 << 1))  fault8 |= (1 << 1); // 过流故障
    if (fault16 & (1 << 0))  fault8 |= (1 << 2); // 堵转

    return fault8;
}

/*******************************************************************************
* 构造函数
*******************************************************************************/
RobStride_Motor_SocketCAN::RobStride_Motor_SocketCAN(uint8_t CAN_Id, const std::string& interface)
    : CAN_ID(CAN_Id), Unique_ID(0), Master_CAN_ID(0xFD),
      Motor_Offset_MotoFunc(nullptr), sockfd(-1), can_interface(interface),
      receiving(false), connected(false), error_code(0), output(0.0f) {

    Motor_Set_All.set_motor_mode = MOVE_CONTROL_MODE;

    // 初始化参数索引
    drw.run_mode.index = Index_List[0];
    drw.iq_ref.index = Index_List[1];
    drw.spd_ref.index = Index_List[2];
    drw.imit_torque.index = Index_List[3];
    drw.cur_kp.index = Index_List[4];
    drw.cur_ki.index = Index_List[5];
    drw.cur_filt_gain.index = Index_List[6];
    drw.loc_ref.index = Index_List[7];
    drw.limit_spd.index = Index_List[8];
    drw.limit_cur.index = Index_List[9];
    drw.mechPos.index = Index_List[10];
    drw.iqf.index = Index_List[11];
    drw.mechVel.index = Index_List[12];
    drw.VBUS.index = Index_List[13];
    drw.rotation.index = Index_List[14];

    // 初始化电机状态
    memset(&Pos_Info, 0, sizeof(Pos_Info));
}

/*******************************************************************************
* 构造函数（带函数指针）
*******************************************************************************/
RobStride_Motor_SocketCAN::RobStride_Motor_SocketCAN(uint8_t CAN_Id, const std::string& interface,
                                                     float (*Offset_MotoFunc)(float Motor_Tar))
    : CAN_ID(CAN_Id), Unique_ID(0), Master_CAN_ID(0xFD),
      Motor_Offset_MotoFunc(Offset_MotoFunc), sockfd(-1), can_interface(interface),
      receiving(false), connected(false), error_code(0), output(0.0f) {

    Motor_Set_All.set_motor_mode = MOVE_CONTROL_MODE;

    // 初始化参数索引
    drw.run_mode.index = Index_List[0];
    drw.iq_ref.index = Index_List[1];
    drw.spd_ref.index = Index_List[2];
    drw.imit_torque.index = Index_List[3];
    drw.cur_kp.index = Index_List[4];
    drw.cur_ki.index = Index_List[5];
    drw.cur_filt_gain.index = Index_List[6];
    drw.loc_ref.index = Index_List[7];
    drw.limit_spd.index = Index_List[8];
    drw.limit_cur.index = Index_List[9];
    drw.mechPos.index = Index_List[10];
    drw.iqf.index = Index_List[11];
    drw.mechVel.index = Index_List[12];
    drw.VBUS.index = Index_List[13];
    drw.rotation.index = Index_List[14];

    // 初始化电机状态
    memset(&Pos_Info, 0, sizeof(Pos_Info));
}

/*******************************************************************************
* 析构函数
*******************************************************************************/
RobStride_Motor_SocketCAN::~RobStride_Motor_SocketCAN() {
    cleanup_socketcan();
}

/*******************************************************************************
* 初始化设备
*******************************************************************************/
bool RobStride_Motor_SocketCAN::initialize() {
    if (!init_socketcan()) {
        return false;
    }

    // 启动接收线程
    receiving = true;
    connected = true;
    receive_thread = std::thread(&RobStride_Motor_SocketCAN::receive_worker, this);

    return true;
}

/*******************************************************************************
* 设置接收回调
*******************************************************************************/
void RobStride_Motor_SocketCAN::set_receive_callback(std::function<void(uint32_t, uint8_t*, uint8_t)> callback) {
    receive_callback = callback;
}

/*******************************************************************************
* 使能电机 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::Enable_Motor() {
    uint8_t data[8] = {0};
    uint32_t id = (Communication_Type_MotorEnable << 24) | (Master_CAN_ID << 8) | CAN_ID;
    data[0] = 1;  // 使能

    can_tx_ext(id, data, 8);
}

/*******************************************************************************
* 失能电机 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::Disenable_Motor(uint8_t clear_error) {
    uint8_t data[8] = {0};
    uint32_t id = (Communication_Type_MotorStop << 24) | (Master_CAN_ID << 8) | CAN_ID;
    data[0] = clear_error;

    can_tx_ext(id, data, 8);
    Set_RobStride_Motor_parameter(0X7005, MOVE_CONTROL_MODE, SET_MODE);
}

/*******************************************************************************
* 设置零位
*******************************************************************************/
void RobStride_Motor_SocketCAN::Set_ZeroPos() {
    uint8_t data[8] = {0};
    uint32_t id = (Communication_Type_SetPosZero << 24) | (Master_CAN_ID << 8) | CAN_ID;
    data[0] = 1;

    can_tx_ext(id, data, 8);
    Enable_Motor();
}

/*******************************************************************************
* 设置电机参数
*******************************************************************************/
void RobStride_Motor_SocketCAN::Set_RobStride_Motor_parameter(uint16_t Index, float Value, char Value_mode) {
    uint8_t data[8];
    uint32_t id = (Communication_Type_SetSingleParameter << 24) | (Master_CAN_ID << 8) | CAN_ID;

    data[0] = Index & 0xFF;
    data[1] = Index >> 8;
    data[2] = 0;
    data[3] = Value_mode;

    if (Value_mode == SET_PARAMETER) {
        // float 值
        memcpy(&data[4], &Value, sizeof(float));
    } else {
        // 模式值 (uint8)
        data[4] = (uint8_t)Value;
        data[5] = 0;
        data[6] = 0;
        data[7] = 0;
    }

    can_tx_ext(id, data, 8);
}

/*******************************************************************************
* 获取电机参数
*******************************************************************************/
void RobStride_Motor_SocketCAN::Get_RobStride_Motor_parameter(uint16_t Index) {
    uint8_t data[8] = {0};
    uint32_t id = (Communication_Type_GetSingleParameter << 24) | (Master_CAN_ID << 8) | CAN_ID;

    data[0] = Index & 0xFF;
    data[1] = Index >> 8;

    if (Index == 0x7005) {
        std::cout << "查询0x7005已经发送" << std::endl;
    }

    can_tx_ext(id, data, 8);
}

/*******************************************************************************
* 运动控制模式
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_move_control(float Torque, float Angle, float Speed, float Kp, float Kd) {
    Motor_Set_All.set_Torque = Torque;
    Motor_Set_All.set_angle = Angle;
    Motor_Set_All.set_speed = Speed;
    Motor_Set_All.set_Kp = Kp;
    Motor_Set_All.set_Kd = Kd;

    // 设置运动控制模式
    if (drw.run_mode.data != MOVE_CONTROL_MODE) {
        Set_RobStride_Motor_parameter(0X7005, MOVE_CONTROL_MODE, SET_MODE);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Enable_Motor();
        Motor_Set_All.set_motor_mode = MOVE_CONTROL_MODE;
    }

    // 如果电机未运行，使能电机
    if (Pos_Info.Pattern != 2) {
        Enable_Motor();
    }

    uint8_t data[8];
    uint32_t id = (Communication_Type_MotionControl << 24) |
                  (float_to_uint(Motor_Set_All.set_Torque, T_MIN, T_MAX, 16) << 8) | CAN_ID;

    data[0] = float_to_uint(Motor_Set_All.set_angle, P_MIN, P_MAX, 16) >> 8;
    data[1] = float_to_uint(Motor_Set_All.set_angle, P_MIN, P_MAX, 16) & 0xFF;
    data[2] = float_to_uint(Motor_Set_All.set_speed, V_MIN, V_MAX, 16) >> 8;
    data[3] = float_to_uint(Motor_Set_All.set_speed, V_MIN, V_MAX, 16) & 0xFF;
    data[4] = float_to_uint(Motor_Set_All.set_Kp, KP_MIN, KP_MAX, 16) >> 8;
    data[5] = float_to_uint(Motor_Set_All.set_Kp, KP_MIN, KP_MAX, 16) & 0xFF;
    data[6] = float_to_uint(Motor_Set_All.set_Kd, KD_MIN, KD_MAX, 16) >> 8;
    data[7] = float_to_uint(Motor_Set_All.set_Kd, KD_MIN, KD_MAX, 16) & 0xFF;

    can_tx_ext(id, data, 8);
}



/*******************************************************************************
* 速度控制模式 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_Speed_control(float Speed, float limit_cur) {
    Motor_Set_All.set_speed = Speed;
    Motor_Set_All.set_limit_cur = limit_cur;

    if (drw.run_mode.data != SPEED_CONTROL_MODE) {
        Set_RobStride_Motor_parameter(0X7005, SPEED_CONTROL_MODE, SET_MODE);     // 设置电机模式
        Get_RobStride_Motor_parameter(0x7005);
        Enable_Motor();
        Motor_Set_All.set_motor_mode = SPEED_CONTROL_MODE;
        Set_RobStride_Motor_parameter(0X7018, Motor_Set_All.set_limit_cur, SET_PARAMETER);
        Set_RobStride_Motor_parameter(0X7022, 10, SET_PARAMETER);
    }
    Set_RobStride_Motor_parameter(0X700A, Motor_Set_All.set_speed, SET_PARAMETER);
}

/*******************************************************************************
* 电流控制模式 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_current_control(float current) {
    Motor_Set_All.set_current = current;
    output = Motor_Set_All.set_current;  // 基于原始库实现

    if (Motor_Set_All.set_motor_mode != ELECT_CONTROL_MODE) {
        Set_RobStride_Motor_parameter(0X7005, ELECT_CONTROL_MODE, SET_MODE);     // 设置电机模式
        Get_RobStride_Motor_parameter(0x7005);
        Motor_Set_All.set_motor_mode = ELECT_CONTROL_MODE;
        Enable_Motor();
    }
    Set_RobStride_Motor_parameter(0X7006, Motor_Set_All.set_current, SET_PARAMETER);
}

/*******************************************************************************
* CSP位置模式（连续同步位置）- 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_CSP_control(float Angle, float limit_spd) {
    Motor_Set_All.set_angle = Angle;
    Motor_Set_All.set_limit_speed = limit_spd;

    if (drw.run_mode.data != CSP_CONTROL_MODE) {
        Set_RobStride_Motor_parameter(0X7005, CSP_CONTROL_MODE, SET_MODE);
        Get_RobStride_Motor_parameter(0x7005);
        Enable_Motor();
        Set_RobStride_Motor_parameter(0X7017, Motor_Set_All.set_limit_speed, SET_PARAMETER);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Set_RobStride_Motor_parameter(0X7016, Motor_Set_All.set_angle, SET_PARAMETER);
}

/*******************************************************************************
* 设置零位控制模式 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_Set_Zero_control() {
    Set_RobStride_Motor_parameter(0X7005, SET_ZERO_MODE, SET_MODE);     // 设置电机模式
}

/*******************************************************************************
* 获取CAN ID - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Get_CAN_ID() {
    uint8_t data[8] = {0};
    uint32_t id = (Communication_Type_Get_ID << 24) | (Master_CAN_ID << 8) | CAN_ID;
    can_tx_ext(id, data, 8);
}

/*******************************************************************************
* 设置CAN ID - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::Set_CAN_ID(uint8_t Set_CAN_ID) {
    Disenable_Motor(0);
    uint8_t data[8] = {0};
    uint32_t id = (Communication_Type_Can_ID << 24) | (Set_CAN_ID << 16) | (Master_CAN_ID << 8) | CAN_ID;
    can_tx_ext(id, data, 8);
}

/*******************************************************************************
* 电机数据保存 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_MotorDataSave() {
    uint8_t data[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    uint32_t id = (Communication_Type_MotorDataSave << 24) | (Master_CAN_ID << 8) | CAN_ID;
    can_tx_ext(id, data, 8);
}

/*******************************************************************************
* 波特率修改 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_BaudRateChange(uint8_t F_CMD) {
    uint8_t data[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, F_CMD, 0x08};
    uint32_t id = (Communication_Type_BaudRateChange << 24) | (Master_CAN_ID << 8) | CAN_ID;
    can_tx_ext(id, data, 8);
}

/*******************************************************************************
* 主动上报设置 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_ProactiveEscalationSet(uint8_t F_CMD) {
    uint8_t data[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, F_CMD, 0x08};
    uint32_t id = (Communication_Type_ProactiveEscalationSet << 24) | (Master_CAN_ID << 8) | CAN_ID;
    can_tx_ext(id, data, 8);
}

/*******************************************************************************
* 电机模式设置 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_MotorModeSet(uint8_t F_CMD) {
    uint8_t data[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, F_CMD, 0x08};
    uint32_t id = (Communication_Type_MotorModeSet << 24) | (Master_CAN_ID << 8) | CAN_ID;
    can_tx_ext(id, data, 8);
}

/*******************************************************************************
* 位置控制模式 (PP模式位置模式控制) - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_Pos_control(float Angle, float Speed) {
    Motor_Set_All.set_speed = Speed;
    Motor_Set_All.set_angle = Angle;
    // 这里按需修改，目前的数值是瞎写的
    Motor_Set_All.set_limit_speed = 1.0;
    Motor_Set_All.set_acceleration = 5.0;
    std::cout << "[DEBUG][POS] run_mode=" << drw.run_mode.data
                << std::endl;

    // 检查是否需要设置控制模式
    if (drw.run_mode.data != POS_CONTROL_MODE) {
        std::cout << "[DEBUG][POS] run_mode=" << drw.run_mode.data
                  << " 预期=" << POS_CONTROL_MODE
                  << " pattern=" << static_cast<int>(Pos_Info.Pattern)
                  << " set_motor_mode=" << Motor_Set_All.set_motor_mode
                  << std::endl;
        Set_RobStride_Motor_parameter(0X7005, POS_CONTROL_MODE, SET_MODE);     // 设置电机模式
        Get_RobStride_Motor_parameter(0x7005);
        Motor_Set_All.set_motor_mode = POS_CONTROL_MODE;
        std::cout << "[DEBUG][POS] 已发送模式设置命令，等待反馈更新 run_mode" << std::endl;
        Enable_Motor();
        Set_RobStride_Motor_parameter(0X7024, Motor_Set_All.set_limit_speed, SET_PARAMETER);
        Set_RobStride_Motor_parameter(0X7025, Motor_Set_All.set_acceleration, SET_PARAMETER);
    }

    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Set_RobStride_Motor_parameter(0X7016, Motor_Set_All.set_angle, SET_PARAMETER);
}

/*******************************************************************************
* 消息分析函数 - 基于原始库实现
*******************************************************************************/
void RobStride_Motor_SocketCAN::RobStride_Motor_Analysis(uint8_t* DataFrame, uint32_t ID_ExtId) {
    std::lock_guard<std::mutex> lock(data_mutex);  // 保护共享数据

    // 标准模式消息处理
    if (uint8_t((ID_ExtId & 0xFF00) >> 8) == CAN_ID) {
        if (int((ID_ExtId & 0x3F000000) >> 24) == 2) {
            // 电机状态反馈
            Pos_Info.Angle = uint16_to_float(DataFrame[0] << 8 | DataFrame[1], P_MIN, P_MAX, 16);
            Pos_Info.Speed = uint16_to_float(DataFrame[2] << 8 | DataFrame[3], V_MIN, V_MAX, 16);
            Pos_Info.Torque = uint16_to_float(DataFrame[4] << 8 | DataFrame[5], T_MIN, T_MAX, 16);
            Pos_Info.Temp = (DataFrame[6] << 8 | DataFrame[7]) * 0.1f;
            error_code = uint8_t((ID_ExtId & 0x3F0000) >> 16);
            Pos_Info.Pattern = uint8_t((ID_ExtId & 0xC00000) >> 22);
        }
        else if (int((ID_ExtId & 0x3F000000) >> 24) == 17) {
            // 参数读取反馈
            for (int index_num = 0; index_num <= 14; index_num++) {
                if ((DataFrame[1] << 8 | DataFrame[0]) == Index_List[index_num]) {
                    switch (index_num) {
                        case 0: {
                            uint8_t new_mode = uint8_t(DataFrame[4]);
                            if (drw.run_mode.data != new_mode) {
                                std::cout << "模式更新：" << static_cast<int>(drw.run_mode.data)
                                          << " -> " << static_cast<int>(new_mode) << std::endl;
                            }
                            drw.run_mode.data = new_mode;
                            break;
                        }
                        case 1: drw.iq_ref.data = Byte_to_float(DataFrame); break;
                        case 2: drw.spd_ref.data = Byte_to_float(DataFrame); break;
                        case 3: drw.imit_torque.data = Byte_to_float(DataFrame); break;
                        case 4: drw.cur_kp.data = Byte_to_float(DataFrame); break;
                        case 5: drw.cur_ki.data = Byte_to_float(DataFrame); break;
                        case 6: drw.cur_filt_gain.data = Byte_to_float(DataFrame); break;
                        case 7: drw.loc_ref.data = Byte_to_float(DataFrame); break;
                        case 8: drw.limit_spd.data = Byte_to_float(DataFrame); break;
                        case 9: drw.limit_cur.data = Byte_to_float(DataFrame); break;
                        case 10:
                            drw.mechPos.data = Byte_to_float(DataFrame);
                            Pos_Info.Angle = drw.mechPos.data;  // 同步更新Pos_Info.Angle
                            break;
                        case 11: drw.iqf.data = Byte_to_float(DataFrame); break;
                        case 12:
                            drw.mechVel.data = Byte_to_float(DataFrame);
                            Pos_Info.Speed = drw.mechVel.data;  // 同步更新速度
                            break;
                        case 13:
                            drw.VBUS.data = Byte_to_float(DataFrame);
                            break;
                        case 14:
                            drw.rotation.data = Byte_to_float(DataFrame);
                            break;
                    }
                }
            }
        }
        // 添加缺失的电机ID解析逻辑（基于原始库）- 兼容处理
        else if ((uint8_t)((ID_ExtId & 0xFF)) == 0xFE) {
            CAN_ID = uint8_t((ID_ExtId & 0xFF00) >> 8);
            memcpy(&Unique_ID, DataFrame, 8);
        }
    }
}

/*******************************************************************************
* 获取电机状态信息（线程安全）
*******************************************************************************/
Motor_Pos_RobStride_Info RobStride_Motor_SocketCAN::get_motor_info() const {
    std::lock_guard<std::mutex> lock(data_mutex);
    return Pos_Info;
}
