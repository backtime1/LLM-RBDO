import numpy as np
import geatpy as ea
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import os
# ==============================================================================
#                 模型训练
# ==============================================================================
# 导入数据
scaler = StandardScaler()
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "car_crash.csv"))
try:
    x = np.genfromtxt(csv_path, delimiter=",")
except OSError:
    raise FileNotFoundError(f"CSV file not found: {csv_path}")
data = np.array(x)
x_train = data[1:, :-10]
scaler_x = StandardScaler()  
x_train_scaled = scaler_x.fit_transform(x_train)
y_train1 = data[1:, -10]
y_train2 = data[1:, -9]
y_train3 = data[1:, -8]
y_train4 = data[1:, -7]
y_train5 = data[1:, -6]
y_train6 = data[1:, -5]
y_train7 = data[1:, -4]
y_train8 = data[1:, -3]
y_train9 = data[1:, -2]
y_train10 = data[1:, -1]

# 定义高斯过程回归的核函数
kernel = C(0.1, (0.001, 1e11)) * RBF(0.1, (1e-5, 1e11))

# 创建高斯过程回归模型并进行训练
reg1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
reg2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
reg3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
reg4 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
reg5 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
reg6 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
reg7 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
reg8 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
reg9 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
reg10 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)

# 拟合模型
reg1.fit(x_train_scaled, y_train1)
reg2.fit(x_train_scaled, y_train2)
reg3.fit(x_train_scaled, y_train3)
reg4.fit(x_train_scaled, y_train4)
reg5.fit(x_train_scaled, y_train5)
reg6.fit(x_train_scaled, y_train6)
reg7.fit(x_train_scaled, y_train7)
reg8.fit(x_train_scaled, y_train8)
reg9.fit(x_train_scaled, y_train9)
reg10.fit(x_train_scaled, y_train10)
# 定义 G1 到 G10 函数
def G1(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg1.predict(x_flat)
    return predictions.reshape(original_shape)

def G2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg2.predict(x_flat)
    return predictions.reshape(original_shape)

def G3(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg3.predict(x_flat)
    return predictions.reshape(original_shape)

def G4(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg4.predict(x_flat)
    return predictions.reshape(original_shape)

def G5(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg5.predict(x_flat)
    return predictions.reshape(original_shape)
def G6(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg6.predict(x_flat)
    return predictions.reshape(original_shape)
def G7(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg7.predict(x_flat)
    return predictions.reshape(original_shape)

def G8(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg8.predict(x_flat)
    return predictions.reshape(original_shape)
def G9(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg9.predict(x_flat)
    return predictions.reshape(original_shape)

def G10(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    x = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
    x_reshaped = x.reshape(-1, x.shape[-1])
    x_scaled_reshaped =scaler_x.transform(x_reshaped)
    x_scaled = x_scaled_reshaped.reshape(x.shape)
    original_shape = x_scaled.shape[:-1]
    x_flat = x_scaled.reshape(-1, 11)
    predictions = reg10.predict(x_flat)
    return predictions.reshape(original_shape)
# 生成样本函数
def generate_samples(means, std_devs, N):
    M = means.shape[0]  # 样本数量
    samples = np.zeros((M, N, 11))  # 形状为 (M, N, 11)
    for i in range(M):
        for j in range(11):  # 遍历每个变量
            if std_devs[j] == 0:
                # 如果标准差为 0，直接使用均值（无随机性）
                samples[i, :, j] = means[i, j]
            else:
                # 否则生成正态分布样本
                samples[i, :, j] = np.random.normal(means[i, j], std_devs[j], N)
    return samples

# 通用可靠性计算函数
def reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func):
    # 将输入变量组合成一个二维数组，形状为 (n, 11)
    means = np.array([x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten(), x5.flatten(),
                      x6.flatten(), x7.flatten(), x8.flatten(), x9.flatten(), x10.flatten(),
                      x11.flatten()]).T
    # print("means shape:", means.shape)  # 打印 means 的形状

    # 给定的标准差
    std_devs = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.006, 0.006, 10, 10])

    # 生成样本
    samples = generate_samples(means, std_devs, 10000)
    # print("samples shape:", samples.shape)  # 打印 samples 的形状

    # 计算 G 函数值
    g_values = G_func(samples[:, :, 0], samples[:, :, 1], samples[:, :, 2], samples[:, :, 3], samples[:, :, 4],
                      samples[:, :, 5], samples[:, :, 6], samples[:, :, 7], samples[:, :, 8], samples[:, :, 9],
                      samples[:, :, 10])
    # print("g_values shape:", g_values.shape)  # 打印 g_values 的形状

    # 计算可靠性
    reliability = np.sum(g_values <= 0, axis=1) / 10000  # 按样本数量求和，得到形状为 (N,) 的数组
    # print("reliability shape:", reliability.shape)  # 打印 reliability 的形状

    # 返回形状为 (N, 1) 的数组
    result = (0.9 - reliability).reshape(-1, 1)
    # print("result shape:", result.shape)  # 打印最终返回值的形状
    return result

# 定义目标函数（含约束）
def evalVars(Vars):
    f = 1.98 + 4.90 * Vars[:, [0]] + 6.67 * Vars[:, [1]] + 6.98 * Vars[:, [2]] + 4.01 * Vars[:, [3]] + 1.78 * Vars[:, [4]] + 2.73 * Vars[:, [6]]
    x1 = Vars[:, [0]]  # 第 1 列
    x2 = Vars[:, [1]]  # 第 2 列
    x3 = Vars[:, [2]]  # 第 3 列
    x4 = Vars[:, [3]]  # 第 4 列
    x5 = Vars[:, [4]]  # 第 5 列
    x6 = Vars[:, [5]]  # 第 6 列
    x7 = Vars[:, [6]]  # 第 7 列
    x8 = Vars[:, [7]]  # 第 8 列
    x9 = Vars[:, [8]]  # 第 9 列

    # 将 x10 和 x11 转换为与 x1 到 x9 形状相同的全零数组
    x10 = np.zeros_like(x1)
    x11 = np.zeros_like(x1)

    # 计算每个约束的可靠性
    CV_list = [
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G1),
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G2),
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G3),
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G4),
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G5),
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G6),
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G7),
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G8),
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G9),
        reliability_g(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, G_func=G10),
    ]

    # 将每个约束的结果拼接成二维数组
    CV = np.hstack(CV_list)

    # 检查 CV 的形状
    if CV.shape[0] != Vars.shape[0]:
        raise ValueError(f"CV 的第一维大小 ({CV.shape[0]}) 必须等于种群大小 ({Vars.shape[0]})")
    if CV.ndim != 2:
        raise ValueError(f"CV 必须是二维数组，当前维度为 {CV.ndim}")

    return f, CV

# 定义优化问题
problem = ea.Problem(name='soea quick start demo',
                     M=1,
                     maxormins=[1],
                     Dim=9,
                     varTypes=[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     lb=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.192, 0.192],
                     ub=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.345, 0.345],
                     evalVars=evalVars)

# 定义算法
algorithm = ea.soea_SEGA_templet(problem,
                                 ea.Population(Encoding='RI', NIND=50),
                                 MAXGEN=100,
                                 logTras=1,
                                 trappedValue=1e-6,
                                 maxTrappedCount=10)

# 运行优化
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, draw=True)

# 获取最优解的决策变量
best_vars = res['Vars']  # 最优个体的决策变量
print("最优解的决策变量:", best_vars)

# 计算最优解的目标函数值和CV值
f_best, cv_best = evalVars(best_vars.reshape(1, -1))  # 将 Vars 转换为二维数组，形状为 (1, Dim)
print("最优解的目标函数值:", f_best)
print("最优解的可靠性值:", 0.9 - cv_best)