"""
MuJoCo 版本的 PushT 环境。

直接使用 mujoco_smoke_test.py 中的场景定义。
"""

import gym
from gym import spaces
import numpy as np
import cv2
import mujoco
import mujoco.viewer
from textwrap import dedent


# 直接使用烟雾测试中的场景定义
# 注意是半尺寸，这里的平面size="0.8 0.8 0.1"，其实边长是1.6
SCENE_XML = dedent(
    """
    <mujoco model="pusht_mujoco">
      <compiler angle="degree" inertiafromgeom="true"/>
      <option timestep="0.01" gravity="0 0 -9.8"/>
      <default>
        <geom
          condim="3"
          friction="0.3 0 0"
          rgba="0.5 0.5 0.5 1"
          solref="0.005 1"
          solimp="0.99 0.999 0.0001"/>
      </default>
      <visual>
        <headlight diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
        <rgba haze="0 0 0 0"/>
        <global azimuth="90" elevation="90" offwidth="1024" offheight="1024"/>
      </visual>
      <worldbody>
        <camera name="top_cam" pos="0 0 2" mode="fixed" xyaxes="1 0 0 0 -1 0"/>
        <light name="key" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3" cutoff="60" exponent="2"/>
        <light name="fill" pos="1.5 1.5 2" dir="-1 -1 -1" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
        <geom name="table" type="plane" size="0.8 0.8 0.1" rgba="0.3 0.2 0.1 1"/>
        <body name="stick" pos="0 0 0.2" quat="0 0 0 1">
            <joint name="slide_x" type="slide" axis="-1 0 0" range="-0.8 0.8"/>
            <joint name="slide_y" type="slide" axis="0 -1 0" range="-0.8 0.8"/>
            <geom type="cylinder" size="0.03 0.2" density="800" rgba="0.8 0.3 0.3 1"/>
        </body>
        <body name="t_block" pos="0 0 0.05">
            <joint name="t_slide_x" type="slide" axis="-1 0 0" range="-0.5 0.5"/>
            <joint name="t_slide_y" type="slide" axis="0 -1 0" range="-0.5 0.5"/>
            <joint name="t_slide_z" type="slide" axis="0 0 1"/>
            <joint name="t_yaw" type="hinge" axis="0 0 1"/>
            <geom type="box" size="0.12 0.03 0.05" density="800"/>
            <geom type="box" size="0.03 0.12 0.05" density="800" pos="0 0.15 0"/>
        </body>
        <!-- 目标区域：透明、无碰撞、固定位置 -->
        <body name="t_goal" pos="0 0 0.025" euler="0 0 45">
            <geom type="box" size="0.12 0.03 0.025" rgba="0.2 0.8 0.2 0.3" contype="0" conaffinity="0"/>
            <geom type="box" size="0.03 0.12 0.025" rgba="0.2 0.8 0.2 0.3" pos="0 0.15 0" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
      <sensor>
        <framepos name="stick_pos" objtype="body" objname="stick"/>
      </sensor>
    </mujoco>
    """
).strip()


class PushTMujocoEnv(gym.Env):
    """MuJoCo 版本的 PushT 环境"""
    
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)
    
    def __init__(
        self,
        render_size=480,
        render_action=True,
        reset_to_state=None,
        # PD 控制参数
        stick_kp=200.0,
        stick_kd=20.0,
    ):
        self._seed = None
        self.seed()
        
        # 场景参数（与烟雾测试一致）
        self.plane_half_extent = np.array([0.8, 0.8])  # 平面半尺寸
        self.stick_range = np.array([0.8, 0.8])  # stick 关节范围
        self.t_block_range = np.array([0.5, 0.5])  # t_block 关节范围
        
        self.render_size = render_size
        self.sim_hz = 100  # 物理仿真频率 (timestep=0.01)
        self.control_hz = self.metadata['video.frames_per_second']  # 控制频率
        
        # PD 控制参数
        self.stick_kp = stick_kp
        self.stick_kd = stick_kd
        
        # 创建 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_string(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        
        # 获取 stick 关节
        self.stick_joint_names = ("slide_x", "slide_y")
        self.stick_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.stick_joint_names
        ]
        self.stick_dof_ids = [self.model.jnt_dofadr[j] for j in self.stick_joint_ids]
        self.stick_qpos_ids = [self.model.jnt_qposadr[j] for j in self.stick_joint_ids]
        
        # 获取 t_block 关节
        self.t_block_joint_names = ("t_slide_x", "t_slide_y", "t_slide_z", "t_yaw")
        self.t_block_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.t_block_joint_names
        ]
        self.t_block_qpos_ids = [self.model.jnt_qposadr[j] for j in self.t_block_joint_ids]
        
        # 获取相机 ID
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam")
        
        # 创建离屏渲染器
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        
        # MuJoCo viewer
        self.viewer = None
        
        # 观测空间：stick_pos(2) + t_block_pos(2) + t_block_angle(1)
        # 使用米坐标系
        self.observation_space = spaces.Box(
            low=np.array([-0.8, -0.8, -0.5, -0.5, -np.pi], dtype=np.float32),
            high=np.array([0.8, 0.8, 0.5, 0.5, np.pi], dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )
        
        # 动作空间：stick 目标位置（米坐标）
        self.action_space = spaces.Box(
            low=np.array([-0.8, -0.8], dtype=np.float32),
            high=np.array([0.8, 0.8], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        # 目标位姿（米坐标）
        self.goal_pose = np.array([0.0, 0.0, np.pi / 4])  # x, y, yaw
        self.success_threshold = 0.95
        
        self.render_action = render_action
        self.reset_to_state = reset_to_state
        self.latest_action = None
        self.n_contact_points = 0
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
    
    def reset(self):
        """重置环境，stick 和 t_block 随机初始化在关节限制内"""
        # 重置 MuJoCo 状态
        mujoco.mj_resetData(self.model, self.data)
        
        if self.reset_to_state is not None:
            self._set_state(self.reset_to_state)
        else:
            # 随机初始化 stick 位置（在关节范围内）
            rs = np.random.RandomState(seed=self._seed)
            stick_x = rs.uniform(-self.stick_range[0] * 0.8, self.stick_range[0] * 0.8)
            stick_y = rs.uniform(-self.stick_range[1] * 0.8, self.stick_range[1] * 0.8)
            
            # 随机初始化 t_block 位置和角度（在关节范围内）
            t_block_x = rs.uniform(-self.t_block_range[0] * 0.8, self.t_block_range[0] * 0.8)
            t_block_y = rs.uniform(-self.t_block_range[1] * 0.8, self.t_block_range[1] * 0.8)
            t_block_yaw = rs.uniform(-np.pi, np.pi)
            
            state = np.array([stick_x, stick_y, t_block_x, t_block_y, t_block_yaw])
            self._set_state(state)
        
        self.latest_action = None
        self.n_contact_points = 0
        
        # 前向计算
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs()
    
    def _set_state(self, state):
        """设置环境状态（米坐标）"""
        stick_x, stick_y = state[0], state[1]
        t_block_x, t_block_y, t_block_yaw = state[2], state[3], state[4]
        
        # 设置 stick 位置
        self.data.qpos[self.stick_qpos_ids[0]] = stick_x
        self.data.qpos[self.stick_qpos_ids[1]] = stick_y
        
        # 设置 t_block 位置和角度
        self.data.qpos[self.t_block_qpos_ids[0]] = t_block_x
        self.data.qpos[self.t_block_qpos_ids[1]] = t_block_y
        self.data.qpos[self.t_block_qpos_ids[2]] = 0  # z 方向
        self.data.qpos[self.t_block_qpos_ids[3]] = t_block_yaw
        
        # 清零速度
        self.data.qvel[:] = 0
        
        # 前向计算
        mujoco.mj_forward(self.model, self.data)
    
    def _get_obs(self):
        """获取观测（米坐标）"""
        stick_pos = self._get_stick_pos()
        t_block_pos = self._get_t_block_pos()
        t_block_yaw = self.data.qpos[self.t_block_qpos_ids[3]]
        
        # 归一化角度到 [-pi, pi]
        t_block_yaw = np.arctan2(np.sin(t_block_yaw), np.cos(t_block_yaw))
        
        obs = np.array([
            stick_pos[0], stick_pos[1],
            t_block_pos[0], t_block_pos[1],
            t_block_yaw
        ], dtype=np.float32)
        return obs
    
    def _get_stick_pos(self):
        """获取 stick 位置（米坐标）"""
        x = self.data.qpos[self.stick_qpos_ids[0]]
        y = self.data.qpos[self.stick_qpos_ids[1]]
        return np.array([x, y])
    
    def _get_t_block_pos(self):
        """获取 t_block 位置（米坐标）"""
        x = self.data.qpos[self.t_block_qpos_ids[0]]
        y = self.data.qpos[self.t_block_qpos_ids[1]]
        return np.array([x, y])
    
    def _get_stick_vel(self):
        """获取 stick 速度"""
        vx = self.data.qvel[self.stick_dof_ids[0]]
        vy = self.data.qvel[self.stick_dof_ids[1]]
        return np.array([vx, vy])
    
    def _apply_stick_pd(self, target_xy):
        """应用 PD 控制到 stick"""
        for axis_idx, dof_id in enumerate(self.stick_dof_ids):
            qpos_idx = self.stick_qpos_ids[axis_idx]
            pos = self.data.qpos[qpos_idx]
            vel = self.data.qvel[dof_id]
            err = target_xy[axis_idx] - pos
            force = self.stick_kp * err - self.stick_kd * vel
            self.data.qfrc_applied[dof_id] = force
    
    def step(self, action):
        """执行一步"""
        n_steps = self.sim_hz // self.control_hz
        self.n_contact_points = 0
        
        if action is not None:
            self.latest_action = np.array(action)
            target_xy = self.latest_action
            
            for _ in range(n_steps):
                # 应用 PD 控制
                self._apply_stick_pd(target_xy)
                
                # 步进仿真
                mujoco.mj_step(self.model, self.data)
                
                # 统计接触点
                self.n_contact_points += self.data.ncon
        else:
            # 无动作时也要步进
            for _ in range(n_steps):
                mujoco.mj_step(self.model, self.data)
        
        # 计算奖励
        coverage = self._compute_coverage()
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, done, info
    
    def _compute_coverage(self):
        """计算 T 块与目标区域的重叠率（简化版本）"""
        t_block_pos = self._get_t_block_pos()
        t_block_yaw = self.data.qpos[self.t_block_qpos_ids[3]]
        
        # 计算与目标的距离和角度差
        pos_error = np.linalg.norm(t_block_pos - self.goal_pose[:2])
        angle_error = abs(t_block_yaw - self.goal_pose[2])
        angle_error = min(angle_error, 2 * np.pi - angle_error)
        
        # 简化的覆盖率计算
        pos_score = max(0, 1 - pos_error / 0.3)
        angle_score = max(0, 1 - angle_error / 0.5)
        coverage = pos_score * angle_score
        
        return coverage
    
    def _get_info(self):
        """获取额外信息"""
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        
        stick_pos = self._get_stick_pos()
        stick_vel = self._get_stick_vel()
        t_block_pos = self._get_t_block_pos()
        t_block_yaw = self.data.qpos[self.t_block_qpos_ids[3]]
        
        info = {
            'pos_agent': stick_pos,
            'vel_agent': stick_vel,
            'block_pose': np.array([t_block_pos[0], t_block_pos[1], t_block_yaw]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step
        }
        return info
    
    def render(self, mode='rgb_array'):
        """渲染画面"""
        if mode == 'human':
            # 使用 MuJoCo 原生 viewer
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # 设置俯视相机
                self.viewer.cam.azimuth = -90
                self.viewer.cam.elevation = -90
                self.viewer.cam.distance = 2.5
                self.viewer.cam.lookat[:] = [0, 0, 0]
            
            self.viewer.sync()
            return None
        
        else:  # rgb_array
            # 使用离屏渲染器
            self.renderer.update_scene(self.data, camera=self.camera_id)
            img = self.renderer.render()
            
            # 绘制动作标记
            if self.render_action and self.latest_action is not None:
                # 将米坐标转换为像素坐标
                action = self.latest_action
                coord_x = int((action[0] + 0.8) / 1.6 * self.render_size)
                coord_y = int((0.8 - action[1]) / 1.6 * self.render_size)
                coord = (coord_x, coord_y)
                marker_size = int(8 / 96 * self.render_size)
                thickness = max(1, int(1 / 96 * self.render_size))
                cv2.drawMarker(
                    img, coord,
                    color=(255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness
                )
            
            return img
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None


class PushTMujocoImageEnv(PushTMujocoEnv):
    """
    图像观测版本的 MuJoCo PushT 环境。
    
    用于训练 Diffusion Policy。
    """
    
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}
    
    def __init__(
        self,
        render_size=480,
    ):
        super().__init__(
            render_size=render_size,
            render_action=False,
        )
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, render_size, render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-0.8,
                high=0.8,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None
    
    def _get_obs(self):
        """获取图像观测"""
        img = super().render(mode='rgb_array')
        agent_pos = self._get_stick_pos()
        
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos.astype(np.float32)
        }
        
        # 绘制动作标记（用于渲染）
        if self.latest_action is not None:
            action = self.latest_action
            coord_x = int((action[0] + 0.8) / 1.6 * self.render_size)
            coord_y = int((0.8 - action[1]) / 1.6 * self.render_size)
            coord = (coord_x, coord_y)
            marker_size = int(8 / 96 * self.render_size)
            thickness = max(1, int(1 / 96 * self.render_size))
            cv2.drawMarker(
                img, coord,
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=marker_size,
                thickness=thickness
            )
        self.render_cache = img
        
        return obs
    
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        if self.render_cache is None:
            self._get_obs()
        return self.render_cache
