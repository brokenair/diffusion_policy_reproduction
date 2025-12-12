"""
RobStride SocketCAN motor driver in pure Python (no ROS).
This is a direct translation of the existing C++ driver, keeping the same CAN IDs,
parameter indices, and scaling helpers. It supports two backends:
1) python-can (recommended): requires `pip install python-can`
2) raw socketcan (fallback): uses the stdlib `socket` module

Notes:
1. 模式切换方面
就是电机要切换模式的话，当前模式需要先disenable,然后再延时一些时间，
目前测试30ms是可以的，但是注意，不是所有模式的切换都要这样，多测试
一下，csp到pos好像不需要，其他没试过，但是确实需要注意运控模式，这
个模式和其他模式切换容易出现问题。

2. 数据读取方面
状态读取和寄存器读取是不一样的，寄存器读取只能得到寄存器，
发送使能和失能命令都能更新状态包，包括各种信息

todo：
1. 测试数据读取，发送控制命令会不会直接更新状态
2. 测试几个控制模式的参数
3. 最重要的是注意延时，有些地方需要延时，如果出问题需要加延时
"""

from __future__ import annotations

import enum
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import can  # type: ignore
except ImportError:  # pragma: no cover
    can = None


# -----------------------------
# Constants (mirrors C++ header)
# -----------------------------
class ControlMode(int, enum.Enum):
    MOVE = 0
    POS = 1
    SPEED = 2
    CURRENT = 3
    SET_ZERO = 4
    CSP = 5


SET_MODE = ord("j")
SET_PARAMETER = ord("p")

Communication_Type_Get_ID = 0x00
Communication_Type_MotionControl = 0x01
Communication_Type_MotorRequest = 0x02
Communication_Type_MotorEnable = 0x03
Communication_Type_MotorStop = 0x04
Communication_Type_SetPosZero = 0x06
Communication_Type_Can_ID = 0x07
Communication_Type_Control_Mode = 0x12
Communication_Type_GetSingleParameter = 0x11
Communication_Type_SetSingleParameter = 0x12
Communication_Type_ErrorFeedback = 0x15
Communication_Type_MotorDataSave = 0x16
Communication_Type_BaudRateChange = 0x17
Communication_Type_ProactiveEscalationSet = 0x18
Communication_Type_MotorModeSet = 0x19


P_MIN = -12.5
P_MAX = 12.5
V_MIN = -33.0
V_MAX = 33.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 5.0
T_MIN = -14.0
T_MAX = 14.0

Index_List = (
    0x7005,
    0x7006,
    0x700A,
    0x700B,
    0x7010,
    0x7011,
    0x7014,
    0x7016,
    0x7017,
    0x7018,
    0x7019,
    0x701A,
    0x701B,
    0x701C,
    0x701D,
)


# -----------------------------
# Helpers
# -----------------------------
def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    span = x_max - x_min
    x = min(max(x, x_min), x_max)
    return int((x - x_min) * ((1 << bits) - 1) / span)


def uint16_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    span = (1 << bits) - 1
    x &= span
    return (x_max - x_min) * x / span + x_min


def bytes_to_float(data: bytes) -> float:
    """
    将CAN消息字节转换为float，匹配C++实现
    C++代码：data[7]<<24 | data[6]<<16 | data[5]<<8 | data[4]
    在小端系统上，这对应小端序的data[4:8]
    """
    if len(data) >= 8:
        # 从字节4-7读取float（小端序）
        return struct.unpack("<f", data[4:8])[0]
    elif len(data) >= 4:
        # 如果只传入4字节
        return struct.unpack("<f", data[:4])[0]
    else:
        return 0.0


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class MotorPosInfo:
    angle: float = 0.0
    speed: float = 0.0
    torque: float = 0.0
    temp: float = 0.0
    pattern: int = 0  # 0 reset, 1 cali, 2 motor


@dataclass
class MotorSet:
    mode: ControlMode = ControlMode.MOVE
    current: float = 0.0
    speed: float = 0.0
    acceleration: float = 5.0
    torque: float = 0.0
    angle: float = 0.0
    limit_cur: float = 0.0
    limit_speed: float = 1.0
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


# -----------------------------
# Driver implementation
# -----------------------------
class RobStrideMotor:
    def __init__(
        self,
        can_id: int,
        interface: str = "can0",
        master_can_id: int = 0xFD,
        backend: str = "python-can",
        recv_sleep: float = 0.0005,
    ) -> None:
        self.can_id = can_id & 0xFF
        self.master_can_id = master_can_id & 0xFF
        self.interface = interface
        self.backend = backend if backend == "socket" or can else "socket"
        self.recv_sleep = recv_sleep

        self._bus = None
        self._sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None
        self._receiving = threading.Event()
        self._lock = threading.Lock()
        self._pos_info = MotorPosInfo()
        self._motor_set = MotorSet()
        self._run_mode = ControlMode.MOVE
        self._connected = False
        self._receive_callback: Optional[Callable[[int, bytes], None]] = None

    # ---- Public API ----
    def initialize(self) -> bool:
        ok = self._init_backend()
        if not ok:
            return False
        self._receiving.set()
        self._recv_thread = threading.Thread(target=self._receive_worker, daemon=True)
        self._recv_thread.start()
        self._connected = True
        return True

    def close(self) -> None:
        self._receiving.clear()
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=1.0)
        if self._sock:
            self._sock.close()
            self._sock = None
        if self._bus:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None
        self._connected = False

    def set_receive_callback(self, cb: Callable[[int, bytes], None]) -> None:
        self._receive_callback = cb

    def enable(self) -> None:
        data = bytes([1]) + bytes(7)
        msg_id = (Communication_Type_MotorEnable << 24) | (self.master_can_id << 8) | self.can_id
        self._send_ext(msg_id, data)

    def disable(self, clear_error: int = 0) -> None:
        data = bytes([clear_error]) + bytes(7)
        msg_id = (Communication_Type_MotorStop << 24) | (self.master_can_id << 8) | self.can_id
        self._send_ext(msg_id, data)
        self.set_parameter(0x7005, ControlMode.MOVE, mode=True)

    def set_zero(self) -> None:
        data = bytes([1]) + bytes(7)
        msg_id = (Communication_Type_SetPosZero << 24) | (self.master_can_id << 8) | self.can_id
        self._send_ext(msg_id, data)
        self.enable()

    def set_parameter(self, index: int, value: float | int, mode: bool = False) -> None:
        msg_id = (Communication_Type_SetSingleParameter << 24) | (self.master_can_id << 8) | self.can_id
        buf = bytearray(8)
        buf[0] = index & 0xFF
        buf[1] = (index >> 8) & 0xFF
        buf[2] = 0
        buf[3] = SET_MODE if mode else SET_PARAMETER
        if mode:
            buf[4] = int(value) & 0xFF
        else:
            struct.pack_into("<f", buf, 4, float(value))
        self._send_ext(msg_id, bytes(buf))

    def get_parameter(self, index: int) -> None:
        msg_id = (Communication_Type_GetSingleParameter << 24) | (self.master_can_id << 8) | self.can_id
        buf = bytearray(8)
        buf[0] = index & 0xFF
        buf[1] = (index >> 8) & 0xFF
        self._send_ext(msg_id, bytes(buf))

    def pos_control(self, angle: float, limit_speed: float = 1.5, acceleration: float = 5.0) -> None:
        '''
        PP位置模式是点位控制，手册期望的是设好 vel/acc，再发这次的目标位置
        运行过程中速度限制和加速度是无法更改的，用于从一个角度运行到另一个
        角度的精确位置控制，不适用于轨迹控制。

        目前我修改了这个函数的实现，只有三个参数

        note：
            速度和加速度是可以在运行中设置的，也就是下面注释的部分取消注释是可以
            运行的，但是数值的具体意义我并不太理解，说明书说的是vel_max，我测试轨迹实际
            速度的2倍无法运行，12倍会飞车，加速度没有测试，所以目前不知道应该设置成为什
            么，目前就按照上面的默认值
        '''
        self._motor_set.angle = angle
        # 这里按需修改，目前的数值是瞎写的，我记得这两个在运行的时候是无法修改的
        # 这是点位模式，也不需要在电机转动的过程中更改
        self._motor_set.limit_speed = limit_speed
        self._motor_set.acceleration = acceleration

        if self._run_mode != ControlMode.POS:
            self.set_parameter(0x7005, ControlMode.POS, mode=True)
            time.sleep(0.001)
            self.get_parameter(0x7005)
            time.sleep(0.001)
            self.enable()
            time.sleep(0.001)
            self.set_parameter(0x7024, self._motor_set.limit_speed, mode=False)
            time.sleep(0.001)
            self.set_parameter(0x7025, self._motor_set.acceleration, mode=False)
            time.sleep(0.001)

        # self.set_parameter(0x7024, self._motor_set.limit_speed, mode=False)
        # time.sleep(0.001)
        # self.set_parameter(0x7025, self._motor_set.acceleration, mode=False)
        # time.sleep(0.001)
        self.set_parameter(0x7016, self._motor_set.angle, mode=False)

    def speed_control(self, speed: float, limit_cur: float, acceleration: float = 5.0) -> None:
        self._motor_set.speed = speed
        self._motor_set.limit_cur = limit_cur
        self._motor_set.acceleration = acceleration

        if self._run_mode != ControlMode.SPEED:
            self.set_parameter(0x7005, ControlMode.SPEED, mode=True)
            time.sleep(0.001)
            self.get_parameter(0x7005)
            time.sleep(0.001)
            self.enable()
            time.sleep(0.001)
            self.set_parameter(0x7018, self._motor_set.limit_cur, mode=False)
            time.sleep(0.001)
            # 这个设置的是速度模式的加速度
            self.set_parameter(0x7022, self._motor_set.acceleration, mode=False)
            time.sleep(0.001)

        # self.set_parameter(0x7022, self._motor_set.acceleration, mode=False)
        # time.sleep(0.001)
        self.set_parameter(0x700A, self._motor_set.speed, mode=False)

    def current_control(self, current: float) -> None:
        self._motor_set.current = current
        
        if self._run_mode != ControlMode.CURRENT:
            self.set_parameter(0x7005, ControlMode.CURRENT, mode=True)
            time.sleep(0.001)
            self.get_parameter(0x7005)
            time.sleep(0.001)
            self.enable()
            time.sleep(0.001)

        self.set_parameter(0x7006, self._motor_set.current, mode=False)

    def csp_control(self, angle: float, limit_speed: float = 6.0) -> None:
        """
        CSP (Cyclic Synchronous Position) control mode.
        Continuous position control with speed limit.
        
        Args:
            angle: Target angle in radians
            limit_speed: Speed limit (default: 6.0 rad/s)
        
        Note:
            CSP mode is suitable for real-time trajectory tracking.
            Position commands are sent cyclically at high frequency.
            这个和PP模式差不多，设置实际速度的某些倍数也无法运行，
            那就不管了，直接默认定值吧
        """
        self._motor_set.angle = angle
        self._motor_set.limit_speed = limit_speed

        # Switch to CSP mode if not already in it
        if self._run_mode != ControlMode.CSP:
            self.set_parameter(0x7005, ControlMode.CSP, mode=True)
            time.sleep(0.001)
            self.get_parameter(0x7005)
            time.sleep(0.001)
            self.enable()
            time.sleep(0.001)
            self.set_parameter(0x7017, self._motor_set.limit_speed, mode=False)
            time.sleep(0.001)

        # self.set_parameter(0x7017, self._motor_set.limit_speed, mode=False)
        # time.sleep(0.001)
        
        # Send position command (0x7016 = loc_ref register)
        self.set_parameter(0x7016, self._motor_set.angle, mode=False)

    def move_control(self, torque: float, angle: float, speed: float, kp: float, kd: float) -> None:
        self._motor_set.torque = torque
        self._motor_set.angle = angle
        self._motor_set.speed = speed
        self._motor_set.kp = kp
        self._motor_set.kd = kd

        if self._run_mode != ControlMode.MOVE:
            self.set_parameter(0x7005, ControlMode.MOVE, mode=True)
            time.sleep(0.001)
            self.get_parameter(0x7005)
            time.sleep(0.001)
            self.enable()
            time.sleep(0.001)

        data = bytearray(8)
        msg_id = (
            (Communication_Type_MotionControl << 24)
            | (float_to_uint(self._motor_set.torque, T_MIN, T_MAX, 16) << 8)
            | self.can_id
        )
        ang_u = float_to_uint(self._motor_set.angle, P_MIN, P_MAX, 16)
        spd_u = float_to_uint(self._motor_set.speed, V_MIN, V_MAX, 16)
        kp_u = float_to_uint(self._motor_set.kp, KP_MIN, KP_MAX, 16)
        kd_u = float_to_uint(self._motor_set.kd, KD_MIN, KD_MAX, 16)
        data[0] = (ang_u >> 8) & 0xFF
        data[1] = ang_u & 0xFF
        data[2] = (spd_u >> 8) & 0xFF
        data[3] = spd_u & 0xFF
        data[4] = (kp_u >> 8) & 0xFF
        data[5] = kp_u & 0xFF
        data[6] = (kd_u >> 8) & 0xFF
        data[7] = kd_u & 0xFF
        self._send_ext(msg_id, bytes(data))

    def get_info(self) -> MotorPosInfo:
        with self._lock:
            return MotorPosInfo(
                angle=self._pos_info.angle,
                speed=self._pos_info.speed,
                torque=self._pos_info.torque,
                temp=self._pos_info.temp,
                pattern=self._pos_info.pattern,
            )

    # ---- Backend & receive loop ----
    def _init_backend(self) -> bool:
        if self.backend == "python-can":
            if can is None:
                return False
            try:
                self._bus = can.interface.Bus(channel=self.interface, bustype="socketcan", receive_own_messages=False)
                return True
            except Exception as exc:  # pragma: no cover
                print(f"[robstride] python-can init failed: {exc}")
                return False
        # raw socket fallback
        try:
            self._sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
            self._sock.setblocking(False)
            self._sock.bind((self.interface,))
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[robstride] socketcan init failed: {exc}")
            self._sock = None
            return False

    def _send_ext(self, msg_id: int, data: bytes) -> None:
        if len(data) > 8:
            data = data[:8]
        if self.backend == "python-can" and self._bus:
            message = can.Message(  # type: ignore
                arbitration_id=msg_id,
                data=data,
                is_extended_id=True,
            )
            try:
                self._bus.send(message)
            except Exception as exc:  # pragma: no cover
                print(f"[robstride] send failed: {exc}")
        elif self._sock:
            can_id = msg_id | socket.CAN_EFF_FLAG
            frame = struct.pack("=IB3x8s", can_id, len(data), data.ljust(8, b"\x00"))
            try:
                os.write(self._sock.fileno(), frame)
            except Exception as exc:  # pragma: no cover
                print(f"[robstride] raw send failed: {exc}")

    def _receive_worker(self) -> None:
        while self._receiving.is_set():
            if self.backend == "python-can" and self._bus:
                msg = self._bus.recv(timeout=0.001)
                if msg:
                    self._handle_frame(msg.arbitration_id, bytes(msg.data))
            elif self._sock:
                try:
                    frame = self._sock.recv(16)
                    if frame:
                        can_id, dlc, data = struct.unpack("=IB3x8s", frame)
                        if can_id & socket.CAN_EFF_FLAG:
                            can_id &= socket.CAN_EFF_MASK
                        self._handle_frame(can_id, data[:dlc])
                except BlockingIOError:
                    pass
                except Exception as exc:  # pragma: no cover
                    print(f"[robstride] recv error: {exc}")
            time.sleep(self.recv_sleep)

    def _handle_frame(self, can_id: int, data: bytes) -> None:
        if len(data) < 8:
            data = data.ljust(8, b"\x00")
        # Check target motor
        if ((can_id >> 8) & 0xFF) != self.can_id:
            return
        command = (can_id >> 24) & 0x3F

        with self._lock:
            if command == Communication_Type_MotorRequest:
                angle_u = (data[0] << 8) | data[1]
                speed_u = (data[2] << 8) | data[3]
                torque_u = (data[4] << 8) | data[5]
                temp_raw = (data[6] << 8) | data[7]
                self._pos_info.angle = uint16_to_float(angle_u, P_MIN, P_MAX, 16)
                self._pos_info.speed = uint16_to_float(speed_u, V_MIN, V_MAX, 16)
                self._pos_info.torque = uint16_to_float(torque_u, T_MIN, T_MAX, 16)
                self._pos_info.temp = temp_raw * 0.1
                self._pos_info.pattern = (can_id >> 22) & 0x3
            elif command == Communication_Type_GetSingleParameter:
                index = (data[1] << 8) | data[0]
                if index == 0x7005:
                    new_mode = data[4]
                    self._run_mode = ControlMode(new_mode) if new_mode in ControlMode._value2member_map_ else self._run_mode
                elif index == 0x7019:
                    self._pos_info.angle = bytes_to_float(data[4:8])
                elif index == 0x701B:
                    self._pos_info.speed = bytes_to_float(data[4:8])
            elif (can_id & 0xFF) == 0xFE:
                self.can_id = (can_id >> 8) & 0xFF

        if self._receive_callback:
            try:
                self._receive_callback(can_id, data)
            except Exception:
                pass


# Convenience factory
def connect(can_id: int, interface: str = "can0", backend: str = "python-can") -> RobStrideMotor:
    motor = RobStrideMotor(can_id=can_id, interface=interface, backend=backend)
    if not motor.initialize():
        raise RuntimeError("Failed to initialize RobStrideMotor")
    return motor
