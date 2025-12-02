import numpy as np
import geatpy as ea

# 目标函数
def cost(x):
    x1, x2 = x
    return  x1 + x2

def G1(x1, x2):
    return x1 ** 2 * x2 / 20 - 1

def G2(x1, x2):
    return (x1 + x2 - 5) ** 2 / 30 + (x1 - x2 - 12) ** 2 / 120 - 1

def G3(x1, x2):
    return 80 / (x1 ** 2 + 8 * x2 - 5) - 1

def generate_samples(x0, stdx, N):
    if x0.shape[1] != 2:
        raise ValueError("x0 的形状应为 (M, 2)，但得到的形状为 {}".format(x0.shape))
    M = x0.shape[0]
    samples = np.zeros((M, N, 2))
    for i in range(M):
        mean = x0[i]
        samples[i] = np.random.normal(mean, stdx, (N, 2))
    return samples

def reliability_g1(x1, x2):
    means = np.array([x1.flatten(), x2.flatten()]).T
    variances = np.array([0.3464 * 0.3464, 0.3464 * 0.3464])
    samples = generate_samples(means, np.sqrt(variances), 10000)
    g_values = G1(samples[:, :, 0], samples[:, :, 1])
    reliability = np.sum(g_values >= 0, axis=1) / 10000
    return 0.98-reliability

def reliability_g2(x1, x2):
    means = np.array([x1.flatten(), x2.flatten()]).T
    variances = np.array([0.3464 * 0.3464, 0.3464 * 0.3464])
    samples = generate_samples(means, np.sqrt(variances), 10000)
    g_values = G2(samples[:, :, 0], samples[:, :, 1])
    reliability = np.sum(g_values >= 0, axis=1) / 10000
    return 0.98-reliability

def reliability_g3(x1, x2):
    means = np.array([x1.flatten(), x2.flatten()]).T
    variances = np.array([0.3464 * 0.3464, 0.3464 * 0.3464])
    samples = generate_samples(means, np.sqrt(variances), 10000)
    g_values = G3(samples[:, :, 0], samples[:, :, 1])
    reliability = np.sum(g_values >= 0, axis=1) / 10000
    return 0.98-reliability

def evalVars(Vars):  # 定义目标函数（含约束）
    f = Vars[:, [0]] + Vars[:, [1]]    # 计算目标函数值
    x1 = Vars[:, [0]]
    x2 = Vars[:, [1]]
    CV = np.hstack([
        reliability_g1(x1, x2).reshape(-1, 1),
        reliability_g2(x1, x2).reshape(-1, 1),
        reliability_g3(x1, x2).reshape(-1, 1)
    ])  # 将每个约束的结果拼接成二维数组
    return f, CV

problem = ea.Problem(name='soea quick start demo',
                     M=1,
                     maxormins=[1],
                     Dim=2,
                     varTypes=[0, 0],
                     lb=[0, 0],
                     ub=[10, 10],
                     evalVars=evalVars)

algorithm = ea.soea_SEGA_templet(problem,
                                 ea.Population(Encoding='RI', NIND=50),
                                 MAXGEN=50,
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
print("最优解的可靠性值:", 0.98-cv_best)