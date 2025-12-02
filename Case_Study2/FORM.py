import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from Scripts.rbdo_utils import reliability_analysis
from Case_Study2.LLM_Ture_constraint import constraint_source_from_11d

# -------------------------------
# 1. 定义目标函数
# -------------------------------
def objective_function(x):
    # x包含11个设计变量：x1, x2, ..., x11
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    # 目标函数（仅依赖部分变量）
    return 1.98 + 4.90 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 2.73 * x7

# -------------------------------
# 2. 定义10个极限状态（约束）函数
# 安全状态： g_i(x) <= 0
# -------------------------------
def G1(x):
    # pc[0]=1.0
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return ((1.16 - 0.3717 * x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10) - 1.0) * 32

def G2(x):
    # pc[1]=32.0
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.7 * x7 * x8 + 0.32 * x9 * x10) - 32.0

def G3(x):
    # pc[2]=32.0
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return (33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22 * x8 * x9) - 32.0

def G4(x):
    # pc[3]=32.0
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return ((46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10) - 32.0) * 10

def G5(x):
    # pc[4]=0.32
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return ((0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 +
             0.0144 * x3 * x5 + 0.0008757 * x5 * x10 + 0.08045 * x6 * x9 +
             0.00139 * x8 * x11 + 0.00001575 * x10 * x11) - 0.32) * 100

def G6(x):
    # pc[5]=0.32
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return ((0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 +
             0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.0208 * x3 * x8 +
             0.121 * x3 * x9 - 0.00364 * x5 * x6 + 0.0007715 * x5 * x10 -
             0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10 -
             0.018 * x2 ** 2) - 0.32) * 100

def G7(x):
    # pc[6]=0.32
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return ((0.74 - 0.61 * x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 -
             0.166 * x7 * x9 + 0.227 * x2 ** 2) - 0.32) * 100

def G8(x):
    # pc[7]=4.0
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return ((4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 +
             0.009325 * x6 * x10 + 0.000191 * x11 ** 2) - 4.0) * 10

def G9(x):
    # pc[8]=9.9
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return ((10.58 - 0.674 * x1 * x2 - 1.95 * x2 * x8 +
             0.02054 * x3 * x10 - 0.0198 * x4 * x10 + 0.028 * x6 * x10) - 9.9) * 3

def G10(x):
    # pc[9]=15.7
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
    return ((16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 +
             0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 ** 2) - 15.7) * 2

# 将所有约束函数放入列表，便于后续处理
G_functions = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G10]

# -------------------------------
# 3. 有限差分梯度计算函数
# -------------------------------
def finite_diff_gradient(g_func, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += eps
        x_backward[i] -= eps
        grad[i] = (g_func(x_forward) - g_func(x_backward)) / (2 * eps)
    return grad

# -------------------------------
# 4. FORM 可靠性计算函数
# -------------------------------
def form_reliability(g_func, x, sigma_x):
    g_val = g_func(x)
    grad = finite_diff_gradient(g_func, x)
    sigma_g = np.sqrt(np.sum((sigma_x * grad)**2))
    beta = -g_val / sigma_g   # 注意负号
    pf = norm.cdf(-beta)
    reliability = 1 - pf
    return beta, reliability



# -------------------------------
# 5. FORM 优化问题设置
# -------------------------------
# 设计变量标准差（根据原MCS例子）
sigma_x = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.006, 0.006, 10, 10])
# 初始设计点（均值），可根据实际情况调整
x0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)

# 目标可靠性要求，例如：要求所有约束可靠性 ≥ 90%
# 此处设定 beta_target 为 norm.ppf(0.9)（约1.2816）
beta_target = norm.ppf(0.9)

# 定义约束函数：要求每个约束的 FORM 指标满足 beta >= beta_target
def reliability_constraints(x):
    constraints = []
    for g_func in G_functions:
        beta, _ = form_reliability(g_func, x, sigma_x)
        constraints.append(beta - beta_target)
    return np.array(constraints)

# -------------------------------
# 6. 综合FORM设计优化求解
# -------------------------------
# 根据你给定的设计变量边界：
# x1~x7在 [0.5, 1.5]，x8、x9在 [0.192, 0.3455]，x10和x11固定为0
bounds = [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5),
          (0.5, 1.5), (0.5, 1.5), (0.192, 0.3455), (0.192, 0.3455), (0, 0), (0, 0)]

# 定义约束字典（所有约束均为不等式：fun(x) >= 0）
cons = {'type': 'ineq', 'fun': reliability_constraints}

result = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=cons)

# -------------------------------
# 7. 输出结果及FORM可靠性分析
# -------------------------------
if result.success:
    x_opt = result.x
    print("优化成功！最优设计点：")
    print(x_opt)
    print(f"最小目标函数值：{result.fun:.4f}")
    print("\n最优设计点下各约束的FORM可靠性：")
    for i, g_func in enumerate(G_functions):
        beta, reliability = form_reliability(g_func, x_opt, sigma_x)
        print(f"G{i+1}: beta = {beta:.4f}, reliability = {reliability:.4f}")
    rels_mcs, _ = reliability_analysis(x_opt, 10000, sigma_x, 0, constraint_source_from_11d, objective_function, verbose=False)
    print("\nMCS矫正可靠性：")
    for i, r in enumerate(rels_mcs):
        print(f"G{i+1}: reliability_mcs = {r:.4f}")
else:
    print("优化失败，请调整参数或优化方法。")
