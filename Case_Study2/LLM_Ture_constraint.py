import numpy as np
import os
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from Scripts.llm_ops import optimize_with_llm as llm_optimize_with_llm
from Scripts.llm_ops import generate_initial_points_with_numpy as llm_generate_initial_points
from Scripts.api_client import create_client


# 目标函数
def objective_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return 1.98 + 4.90 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 2.73 * x7


# ==============================================================================
#                 实验配置参数
# ==============================================================================
N = 10000 # 生成蒙特卡洛样本的数量
threshold = 0 # 约束阈值
x1_range = [0.5, 1.5]
x2_range = [0.5, 1.5]
x3_range = [0.5, 1.5]
x4_range = [0.5, 1.5]
x5_range = [0.5, 1.5]
x6_range = [0.5, 1.5]
x7_range = [0.5, 1.5]
x8_range = [0.192, 0.345]
x9_range = [0.192, 0.345]
x10_range = [-30, 30] # 虽然 LLM 不直接生成，但这里需要定义范围以便 reliability_analysis 使用
x11_range = [-30, 30] # 同上
reliability_target = 0.9 # 目标可靠性
temperature = 0.2 # 温度参数，控制生成文本的随机性
top_p = 0.9 #  nucleus sampling 参数，控制生成文本的多样性
max_iterations = 100 # 最大迭代次数
stagnation_limit = 10 # 停滞限制次数
penalty_limit = 0.1 # 惩罚限制值
retain_number = 5 # 保留数量
original_ranges = {
    "x1_range": x1_range,
    "x2_range": x2_range,
    "x3_range": x3_range,
    "x4_range": x4_range,
    "x5_range": x5_range,
    "x6_range": x6_range,
    "x7_range": x7_range,
    "x8_range": x8_range,
    "x9_range": x9_range
} # 原始变量范围
target_range = [0, 100] # 目标变量范围
std_dev = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.006, 0.006, 10, 10])  # 目标变量的标准差
penalty_weight = 10000 # 惩罚权重
adition_point_number=10 # 新增点的数量
adition_point_std = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.006, 0.006, 0, 0]) # 新增点的标准差
client = create_client("deepseek") # 创建 deepseek 客户端
model = "deepseek-chat" # 模型名称
template_path = os.path.join(ROOT, "Scripts", "prompt_template_Chinese.md") # 提示模板路径
max_tokens = 512 # 最大 tokens 数

# ==============================================================================
#                 约束源与维度扩展封装
# ==============================================================================
def constraint_source_from_11d(X):
     # 提取参数
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = [X[:, i] for i in range(11)]
    
    # 目标值
    pc = np.array([1.0, 32.0, 32.0, 32.0, 0.32, 0.32, 0.32, 4.0, 9.9, 15.7])
    
    # 初始化约束条件矩阵
    ceq = np.zeros((X.shape[0], 10))

    # 计算各个约束条件
    ceq[:, 0] = ((1.16 - 0.3717 * x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10) - pc[0]) * 32
    ceq[:, 1] = (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.7 * x7 * x8 + 0.32 * x9 * x10) - pc[1]
    ceq[:, 2] = (33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22 * x8 * x9) - pc[2]
    ceq[:, 3] = ((46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10) - pc[3]) * 10
    ceq[:, 4] = ((0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 
                  0.0008757 * x5 * x10 + 0.08045 * x6 * x9 + 0.00139 * x8 * x11 + 0.00001575 * x10 * x11) - pc[4]) * 100
    ceq[:, 5] = ((0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 + 0.03099 * x2 * x6 - 
                  0.018 * x2 * x7 + 0.0208 * x3 * x8 + 0.121 * x3 * x9 - 0.00364 * x5 * x6 + 0.0007715 * x5 * x10 - 
                  0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10 - 0.018 * x2 ** 2) - pc[5]) * 100
    ceq[:, 6] = ((0.74 - 0.61 * x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 ** 2) - pc[6]) * 100
    ceq[:, 7] = ((4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 ** 2) - pc[7]) * 10
    ceq[:, 8] = ((10.58 - 0.674 * x1 * x2 - 1.95 * x2 * x8 + 0.02054 * x3 * x10 - 0.0198 * x4 * x10 + 0.028 * x6 * x10) - pc[8]) * 3
    ceq[:, 9] = ((16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 ** 2) - pc[9]) * 2

    return -ceq

def expand_9d_to_11d(x9):
    return np.concatenate([np.asarray(x9), np.array([0.0, 0.0])])

# ==============================================================================
#                 入口：生成初始点并优化（9D→11D评估）
# ==============================================================================
if __name__ == "__main__":
    ranges = [
        original_ranges["x1_range"],
        original_ranges["x2_range"],
        original_ranges["x3_range"],
        original_ranges["x4_range"],
        original_ranges["x5_range"],
        original_ranges["x6_range"],
        original_ranges["x7_range"],
        original_ranges["x8_range"],
        original_ranges["x9_range"],
    ]
    initial_point = llm_generate_initial_points(ranges, num_points=60)
    best_point, best_cost, best_reliabilities, best_penalty, actual_iterations = llm_optimize_with_llm(
        initial_point=initial_point,
        reliability_target=reliability_target,
        max_iterations=max_iterations,
        stagnation_limit=stagnation_limit,
        penalty_limit=penalty_limit,
        penalty_weight=penalty_weight,
        temperature=temperature,
        top_p=top_p,
        retain_number=retain_number,
        adition_point_number=adition_point_number,
        N=N,
        threshold=threshold,
        original_ranges=original_ranges,
        target_range=target_range,
        client=client,
        max_tokens=max_tokens,
        std=std_dev,
        adition_point_std=adition_point_std,
        constraint_source=constraint_source_from_11d,
        objective_fn=objective_function,
        model=model,
        template_path=template_path,
        verbose=True,
        plot=True,
        print_prompt=True,
        return_details=True,
        expand_point=expand_9d_to_11d,
    )
    print(f"Best point (9D): {best_point}, Best cost: {best_cost}, Best reliabilities: {best_reliabilities}, Best penalty: {best_penalty}, Iterations: {actual_iterations}")

