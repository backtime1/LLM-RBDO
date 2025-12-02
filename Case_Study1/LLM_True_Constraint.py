import numpy as np
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from Scripts.llm_ops import optimize_with_llm as llm_optimize_with_llm
from Scripts.llm_ops import generate_initial_points_with_numpy as llm_generate_initial_points
from Scripts.api_client import create_client

# ==============================================================================
#                 约束定义
# ==============================================================================
def compute_constraints(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    ceq = np.zeros((X.shape[0], 3))
    ceq[:, 0] = x1**2 * x2 / 20 - 1
    ceq[:, 1] = (x1 + x2 - 5)**2 / 30 + (x1 - x2 - 12)**2 / 120 - 1
    ceq[:, 2] = 80 / (x1**2 + 8 * x2 - 5) - 1
    return ceq

def objective_function(x1, x2):
    return x1 + x2

# ==============================================================================
#                 实验配置参数
# ==============================================================================
N = 10000
threshold = 0
x1_range = [0, 10]
x2_range = [0, 10]
reliability_target = 0.98
temperature = 0.2
top_p = 0.9
max_iterations = 50
stagnation_limit = 10
penalty_limit = 0.01
retain_number = 5
original_ranges = {
    "x1_range": x1_range,
    "x2_range": x2_range,
}
target_range = [0, 100]
std_dev = np.array([0.3464, 0.3464])
penalty_weight = 10000
adition_point_std = np.array([0.3464, 0.3464])
adition_point_number=10
model = "deepseek-chat"
template_path = os.path.join(ROOT, "Scripts", "prompt_template_Chinese.md")
max_tokens = 512


if __name__ == "__main__":
    # 步骤 1：在范围内生成初始点集合（整数映射依赖于 llm_ops 内部的映射工具）
    ranges = [x1_range, x2_range]
    initial_point = llm_generate_initial_points(ranges, num_points=20)
    # 步骤 2：调用 LLM 优化流程（传入全部显式参数与真实约束函数、目标函数）
    client = create_client("deepseek")
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
        model=model,
        template_path=template_path,
        constraint_source=compute_constraints,
        objective_fn=objective_function,
        verbose=True,
        return_details=True,
        plot=True,
        print_prompt=True,
    )
    # 步骤 4：输出结果（最优点、最优成本与实际迭代次数）
    print(f"Best point: {best_point}, Best cost: {best_cost}, Best reliabilities: {best_reliabilities}, Best penalty: {best_penalty}, Actual iterations: {actual_iterations}")

