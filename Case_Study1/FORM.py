import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from Scripts.rbdo_utils import reliability_analysis

# 全局计数器，用于统计可靠性评估次数
reliability_eval_count = 0

# 目标函数
def objective_function(x):
    x1, x2 = x
    return x1 + x2  # 需要最小化的目标


# 极限状态函数
def G1(x):
    x1, x2 = x
    return x1 ** 2 * x2 / 20 - 1


def G2(x):
    x1, x2 = x
    return (x1 + x2 - 5) ** 2 / 30 + (x1 - x2 - 12) ** 2 / 120 - 1


def G3(x):
    x1, x2 = x
    return 80 / (x1 ** 2 + 8 * x2 - 5) - 1


# 有限差分梯度计算
def finite_diff_gradient(g_func, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += eps
        x_backward[i] -= eps
        grad[i] = (g_func(x_forward) - g_func(x_backward)) / (2 * eps)
    return grad


# 计算 FORM 可靠性指标 beta
def form_reliability(g_func, x, sigma_x):
    global reliability_eval_count  # 声明使用全局变量
    reliability_eval_count += 1  # 每次调用，计数器加1
    mu_g = g_func(x)
    grad = finite_diff_gradient(g_func, x)
    sigma_g = np.sqrt(np.sum((sigma_x * grad) ** 2))
    beta = mu_g / sigma_g
    pf = norm.cdf(-beta)  # 失效概率
    return beta, 1 - pf  # 返回 beta 和可靠性


# 可靠性优化约束：所有可靠性 >= 0.98
def reliability_constraints(x):
    sigma_x = np.array([0.3464, 0.3464])  # 标准差
    beta_target = norm.ppf(0.98)  # 目标可靠性 0.98 对应 beta
    constraints = [
        form_reliability(G1, x, sigma_x)[0] - beta_target,
        form_reliability(G2, x, sigma_x)[0] - beta_target,
        form_reliability(G3, x, sigma_x)[0] - beta_target,
    ]
    return constraints


def constraint_source_2d(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    ceq1 = x1 ** 2 * x2 / 20 - 1
    ceq2 = (x1 + x2 - 5) ** 2 / 30 + (x1 - x2 - 12) ** 2 / 120 - 1
    ceq3 = 80 / (x1 ** 2 + 8 * x2 - 5) - 1
    return np.column_stack([ceq1, ceq2, ceq3])


# 进行优化求解
x0 = np.array([5.0, 5.0])  # 初始设计点
bounds = [(0, 10), (0, 10)]  # 设计变量范围

#重置计数器
reliability_eval_count = 0

result = minimize(
    objective_function, x0, method='SLSQP',
    bounds=bounds,
    constraints={'type': 'ineq', 'fun': reliability_constraints}
)

if result.success:
    x_opt = result.x
    sigma_x = np.array([0.3464, 0.3464])
    beta_g1, reliability_g1 = form_reliability(G1, x_opt, sigma_x)
    beta_g2, reliability_g2 = form_reliability(G2, x_opt, sigma_x)
    beta_g3, reliability_g3 = form_reliability(G3, x_opt, sigma_x)

    print(f"优化成功！最优设计点：x1={x_opt[0]:.4f}, x2={x_opt[1]:.4f}")
    print(f"最小化目标函数值：{result.fun:.4f}")
    print("\n最优解下的可靠性：")
    print(f"  G1: reliability = {reliability_g1:.4f}, beta = {beta_g1:.4f}")
    print(f"  G2: reliability = {reliability_g2:.4f}, beta = {beta_g2:.4f}")
    print(f"  G3: reliability = {reliability_g3:.4f}, beta = {beta_g3:.4f}")
    print(f"\nFORM 可靠性评估总次数: {reliability_eval_count}")
    rels_mcs, _ = reliability_analysis(x_opt, 10000, sigma_x, 0, constraint_source_2d, objective_function, verbose=False)
    print("\nMCS矫正可靠性：")
    for i, r in enumerate(rels_mcs):
        print(f"  G{i+1}: reliability_mcs = {r:.4f}")
else:
    print("优化失败，请调整参数或优化方法。")
    print(f"\nFORM 可靠性评估总次数: {reliability_eval_count}")
