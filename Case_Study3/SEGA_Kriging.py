import numpy as np
import geatpy as ea
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import os

# 目标函数
def cost(x):
    x1, x2, x3 = x
    return np.pi * (0.51 + x2) * np.sqrt((1.2 * x1 / (x1 + 1))**2 + (x2 - 0.51)**2) + np.pi * (x3 + x2) * np.sqrt((1.2 / (x1 + 1))**2 + (x2 - x3)**2)

# ==============================================================================
#                 模型训练
# ==============================================================================
# 导入数据
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "simulation_results.csv"))
try:
    x = np.genfromtxt(csv_path, delimiter=",")
except OSError:
    raise FileNotFoundError(f"CSV file not found: {csv_path}")
data = np.array(x)
x_train = data[1:, :-1]
y_train = data[1:, -1]
# 定义高斯过程回归的核函数
kernel = C(1.0, (1e-5, 1e8)) * RBF(1.0, (1e-5, 1e8))
# 创建高斯过程回归模型并进行训练
reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)
# 拟合模型
reg.fit(x_train, y_train)

def G(x1, x2, x3):
    # Stack the inputs along a new last axis: shape becomes (M, N, 3)
    x = np.stack([x1, x2, x3], axis=-1)
    # Save the original shape (M, N) for later reshaping
    original_shape = x.shape[:-1]
    # Reshape to (-1, 3), i.e. (M*N, 3), so each row is one 3D sample
    x_flat = x.reshape(-1, 3)
    # Get predictions for each sample
    predictions = reg.predict(x_flat)
    # Reshape predictions back to (M, N)
    return predictions.reshape(original_shape)

def generate_samples(x0, stdx, N):
    if x0.shape[1] != 3:
        raise ValueError("x0 的形状应为 (M, 3)，但得到的形状为 {}".format(x0.shape))
    M = x0.shape[0]
    samples = np.zeros((M, N, 3))
    for i in range(M):
        mean = x0[i]
        samples[i] = np.random.normal(mean, stdx, (N, 3))
    return samples

def reliability_g(x1, x2, x3):
    means = np.array([x1.flatten(), x2.flatten(), x3.flatten()]).T
    variances = np.array([0.006 * 0.006, 0.004 * 0.004, 0.004 * 0.004])
    samples = generate_samples(means, np.sqrt(variances), 10000)
    g_values = G(samples[:, :, 0], samples[:, :, 1], samples[:, :, 2])
    reliability = np.sum(g_values >= 3, axis=1) / 10000
    return 0.99 - reliability

def evalVars(Vars):  # 定义目标函数（含约束）
    f = np.pi * (0.51 + Vars[:, [1]]) * np.sqrt((1.2 * Vars[:, [0]] / (Vars[:, [0]] + 1))**2 + (Vars[:, [1]] - 0.51)**2) + np.pi * (Vars[:, [2]] + Vars[:, [1]]) * np.sqrt((1.2 / (Vars[:, [0]] + 1))**2 + (Vars[:, [1]] - Vars[:, [2]])**2)  # 计算目标函数值
    x1 = Vars[:, [0]]
    x2 = Vars[:, [1]]
    x3 = Vars[:, [2]]
    CV = np.hstack([
        reliability_g(x1, x2, x3).reshape(-1, 1)
    ])  # 将每个约束的结果拼接成二维数组
    return f, CV

problem = ea.Problem(name='soea quick start demo',
                     M=1,
                     maxormins=[1],
                     Dim=3,
                     varTypes=[0, 0, 0],
                     lb=[0.5, 0.2, 0.3],
                     ub=[0.7, 0.3, 0.4],
                     evalVars=evalVars)

algorithm = ea.soea_SEGA_templet(problem,
                                 ea.Population(Encoding='RI', NIND=50),
                                 MAXGEN=100,
                                 logTras=1,
                                 trappedValue=1e-6,
                                 maxTrappedCount=10)

res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, draw=True)
# 获取最优解的决策变量
best_vars = res['Vars']  # 最优个体的决策变量
print("最优解的决策变量:", best_vars)

# 计算最优解的目标函数值和CV值
f_best, cv_best = evalVars(best_vars.reshape(1, -1))  # 将 Vars 转换为二维数组，形状为 (1, Dim)
print("最优解的目标函数值:", f_best)
print("最优解的可靠性值:", 0.99 - cv_best)