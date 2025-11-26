import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 1. 超参数
# ========================
T = 150
batch_size = 128
num_train_steps = 10000

# beta schedule（线性）
beta_start = 1e-5
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
    def __init__(self, T, t_embed_dim=16, hidden_dim=32):
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
# 4. 训练循环：学习 epsilon_theta
# ========================
def sample_real_x0(batch_size, device):
    """
    真实数据分布：
    P(x0) = 0.8 * U(-1, 0) + 0.2 * U(0, 1)
    """
    u = torch.rand(batch_size, 1, device=device)
    left  = -1.0 + torch.rand(batch_size, 1, device=device) * 1.0  # U(-1,0)
    right =  0.0 + torch.rand(batch_size, 1, device=device) * 1.0  # U(0,1)
    x0 = torch.where(u < 0.8, left, right)
    return x0

for step in range(num_train_steps):
    # 真实数据 x0 ~ 0.8*U(-1,0) + 0.2*U(0,1)
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
# 5. 采样：反向扩散
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
def sample_trajectory(num_samples=5000, record_paths=True, num_paths_to_record=5):
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

x0_gen, paths = sample_trajectory(num_paths_to_record=10)

# 统计 [-1,0) 和 [0,1] 的数量
x0_gen_np = x0_gen  # x0_gen 本来就是 numpy，如果你改成 torch 了，就加 .cpu().numpy()
left_mask  = (x0_gen_np >= -1.0) & (x0_gen_np < 0.0)
right_mask = (x0_gen_np >=  0.0) & (x0_gen_np <= 1.0)
left_count  = left_mask.sum()
right_count = right_mask.sum()
total = x0_gen_np.shape[0]

print(f"Generated count in [-1, 0): {left_count} ({left_count/total:.3f})")
print(f"Generated count in [0, 1]:  {right_count} ({right_count/total:.3f})")

# ========================
# 6. 可视化：直方图 + 轨迹
# ========================
# 真实分布采样
x0_real = sample_real_x0(5000, device="cpu")
real_mean = x0_real.mean().item()
real_var  = x0_real.var(unbiased=False).item()
print("Real mean, var:", real_mean, real_var)
print("DDPM mean, var:", x0_gen.mean(), x0_gen.var())

plt.figure(figsize=(10, 4))

# (1) 直方图
plt.subplot(1, 2, 1)
plt.title("Generated x0 vs True 0.8*U(-1,0)+0.2*U(0,1)")
plt.hist(x0_gen, bins=60, density=True, alpha=0.6, label="DDPM samples")

xs = torch.linspace(-1.5, 1.5, 400)
pdf = torch.zeros_like(xs)
pdf[(xs >= -1.0) & (xs <= 0.0)] = 0.8   # 0.8 * 1 / 1
pdf[(xs >  0.0) & (xs <= 1.0)]  = 0.2   # 0.2 * 1 / 1
plt.plot(xs.numpy(), pdf.numpy(), linewidth=2, label="True mixture")

plt.legend()
plt.xlabel("x")

# (2) 轨迹（x_t）
time_axis = list(range(T+1))
plt.subplot(1, 2, 2)
plt.title("A few sample trajectories (x_t)")

# 参考线：y = -1, 0, 1
plt.axhline(y=-1.0, linestyle="--", linewidth=1)
plt.axhline(y= 0.0, linestyle="--", linewidth=1)
plt.axhline(y= 1.0, linestyle="--", linewidth=1)

for i in range(paths.shape[1]):
    plt.plot(time_axis, paths[:, i], marker="o", markersize=2, linewidth=1)

plt.xlabel("t (0 is final data space)")
plt.ylabel("x_t")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

