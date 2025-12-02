import numpy as np
import geatpy as ea


# 定义 G1 到 G10 函数
def G1(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((1.16 - 0.3717 * x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10) - 1) * 32

def G2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.7 * x7 * x8 + 0.32 * x9 * x10) - 32

def G3(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return (33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22 * x8 * x9) - 32

def G4(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10) - 32) * 10

def G5(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0008757 * x5 * x10 + 0.08045 * x6 * x9 + 0.00139 * x8 * x11 + 0.00001575 * x10 * x11) - 0.32) * 100

def G6(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.0208 * x3 * x8 + 0.121 * x3 * x9 - 0.00364 * x5 * x6 + 0.0007715 * x5 * x10 - 0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10 - 0.018 * x2 ** 2) - 0.32) * 100

def G7(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((0.74 - 0.61 * x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 ** 2) - 0.32) * 100

def G8(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 ** 2) - 4.0) * 10

def G9(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((10.58 - 0.674 * x1 * x2 - 1.95 * x2 * x8 + 0.02054 * x3 * x10 - 0.0198 * x4 * x10 + 0.028 * x6 * x10) - 9.9) * 3

def G10(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return ((16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 ** 2) - 15.7) * 2

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