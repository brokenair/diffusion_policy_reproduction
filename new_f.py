import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.distributions import Chi2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 1. 超参数
# ========================
T = 200
batch_size = 256
num_train_steps = 10000

# beta schedule（线性）
beta_start = 1e-4
beta_end   = 0.02
betas      = torch.linspace(beta_start, beta_end, T, device=device)
alphas     = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t

# 后验方差 \tilde{beta}_t
alpha_bars_prev = torch.cat(
    [torch.tensor([1.0], device=device), alpha_bars[:-1]], dim=0
)
posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

# ========================
# 2. 噪声网络 epsilon_theta(x_t, t)
# ========================
class Simple1DDenoiser(nn.Module):
    def __init__(self, T, t_embed_dim=16, hidden_dim=64):
        super().__init__()
        self.t_embed = nn.Embedding(T, t_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(1 + t_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t, t):
        """
        x_t: (B, 1)
        t:   (B,) int64, 0..T-1
        """
        t_emb = self.t_embed(t)          # (B, t_embed_dim)
        inp   = torch.cat([x_t, t_emb], dim=-1)
        return self.net(inp)             # (B, 1)

model = Simple1DDenoiser(T).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# ========================
# 3. forward q(x_t | x_0)
# ========================
def q_sample(x0, t, noise=None):
    """
    x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
    x0: (B,1)
    t:  (B,)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bars[t].view(-1, 1)     # (B,1)
    mean = torch.sqrt(alpha_bar_t) * x0
    std  = torch.sqrt(1.0 - alpha_bar_t)
    return mean + std * noise

# ========================
# 4. 真实数据分布：F 分布 -> 截断 [0,10] -> 线性缩放到 [-1,1]
# ========================
# F(d1, d2) = (X1/d1) / (X2/d2)，X1~Chi2(d1), X2~Chi2(d2)
d1, d2 = 5.0, 10.0
chi2_1 = Chi2(d1)
chi2_2 = Chi2(d2)

max_val = 10.0  # 截断上界

def sample_real_x0(batch_size, device):
    """
    先采样 F 分布：F = (Chi2(d1)/d1) / (Chi2(d2)/d2) >= 0
    然后截断到 [0, max_val]，再线性缩放到 [-1,1]
    """
    x1 = chi2_1.sample((batch_size, 1)).to(device) / d1
    x2 = chi2_2.sample((batch_size, 1)).to(device) / d2
    F_val = x1 / x2                           # F >= 0
    F_clamped = torch.clamp(F_val, 0.0, max_val)
    # [0, max_val] -> [-1, 1]
    x0 = (F_clamped / max_val) * 2.0 - 1.0
    return x0

# ========================
# 5. 训练循环：学习 epsilon_theta
# ========================
for step in range(num_train_steps):
    # 真实数据 x0 ~ 由 F 分布构造，缩放到 [-1,1]
    x0 = sample_real_x0(batch_size, device)

    # 随机 t
    t = torch.randint(low=0, high=T, size=(batch_size,), device=device)

    # 前向加噪
    eps = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise=eps)

    # 预测噪声
    eps_hat = model(x_t, t)

    # 噪声回归损失
    loss = torch.mean((eps - eps_hat) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 1000 == 0:
        print(f"step {step+1}/{num_train_steps}, loss = {loss.item():.4f}")

# ========================
# 6. 采样：反向扩散
# ========================
@torch.no_grad()
def p_sample_step(x_t, t_int):
    """
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t)*eps_theta)
              + sigma_t * z
    其中 sigma_t^2 = posterior_var[t_int] = \tilde{beta}_t
    """
    t = torch.full((x_t.shape[0],), t_int, device=device, dtype=torch.long)
    beta_t      = betas[t_int]
    alpha_t     = alphas[t_int]
    alpha_bar_t = alpha_bars[t_int]

    eps_theta = model(x_t, t)  # (B,1)

    coef1 = 1.0 / torch.sqrt(alpha_t)
    coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)

    mean = coef1 * (x_t - coef2 * eps_theta)

    if t_int > 0:
        z = torch.randn_like(x_t)
        sigma_t = torch.sqrt(posterior_var[t_int])
        x_prev = mean + sigma_t * z
    else:
        x_prev = mean

    return x_prev

@torch.no_grad()
def sample_trajectory(num_samples=5000, record_paths=True, num_paths_to_record=10):
    # 初始：x_T ~ N(0,1)
    x_t = torch.randn(num_samples, 1, device=device)
    paths = None
    if record_paths:
        paths = [x_t[:num_paths_to_record, 0].cpu().numpy()]  # T 时刻

    for t_int in reversed(range(T)):
        x_t = p_sample_step(x_t, t_int)
        if record_paths:
            paths.append(x_t[:num_paths_to_record, 0].cpu().numpy())

    x0_samples = x_t.cpu().numpy().reshape(-1)

    if record_paths:
        # 当前 paths 顺序：[x_T, x_{T-1}, ..., x_0]，反转成 0..T
        paths = list(reversed(paths))
        paths = np.stack(paths, axis=0)      # (T+1, num_paths)
        return x0_samples, paths
    else:
        return x0_samples, None

x0_gen, paths = sample_trajectory()

# ========================
# 7. 统计分布 & 打印
# ========================
x0_gen_np = x0_gen  # numpy

left_mask  = (x0_gen_np >= -1.0) & (x0_gen_np < 0.0)
right_mask = (x0_gen_np >=  0.0) & (x0_gen_np <= 1.0)
left_count  = left_mask.sum()
right_count = right_mask.sum()
total = x0_gen_np.shape[0]

print(f"Generated count in [-1, 0): {left_count} ({left_count/total:.3f})")
print(f"Generated count in [0, 1]:  {right_count} ({right_count/total:.3f})")

# 真实分布采样（用很多样本近似真实形状）
x0_real = sample_real_x0(50000, device="cpu")
real_mean = x0_real.mean().item()
real_var  = x0_real.var(unbiased=False).item()
print("Real mean, var:", real_mean, real_var)
print("DDPM mean, var:", x0_gen.mean(), x0_gen.var())

# ========================
# 8. 可视化：直方图 + 轨迹
# ========================
plt.figure(figsize=(10, 4))

# (1) 直方图：真实 vs 生成
plt.subplot(1, 2, 1)
plt.title("Real vs Generated x0 (F-based → [-1,1])")
plt.hist(x0_real.numpy().reshape(-1), bins=80, density=True, alpha=0.6,
         label="Real (F-based)", color=None)
plt.hist(x0_gen, bins=80, density=True, alpha=0.6,
         label="DDPM samples", color=None)
plt.legend()
plt.xlabel("x")

# (2) 轨迹（x_t）
time_axis = list(range(T+1))
plt.subplot(1, 2, 2)
plt.title("A few sample trajectories (x_t)")

# 参考线：y = -1, 0, 1
plt.axhline(y=-1.0, linestyle="--", linewidth=1)
plt.axhline(y= 0.0, linestyle="--", linewidth=1)
plt.axhline(y=-0.5, linestyle="--", linewidth=1)

for i in range(paths.shape[1]):
    plt.plot(time_axis, paths[:, i], marker="o", markersize=2, linewidth=1)

plt.xlabel("t (0 is final data space)")
plt.ylabel("x_t")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()
