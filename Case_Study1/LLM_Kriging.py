import numpy as np
import os
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from Scripts.llm_ops import optimize_with_llm as llm_optimize_with_llm
from Scripts.llm_ops import generate_initial_points_with_numpy as llm_generate_initial_points
from Scripts.api_client import create_client

# ==============================================================================
#                 模型训练
# ==============================================================================
# 导入数据
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "multy_cons_2D.csv"))
try:
    x = np.genfromtxt(csv_path, delimiter=",")
except OSError:
    raise FileNotFoundError(f"CSV file not found: {csv_path}")
data = np.array(x)
x_train = data[1:, :-3]
y_train1 = data[1:, -3]
y_train2 = data[1:, -2]
y_train3 = data[1:, -1]

# 定义高斯过程回归的核函数
kernel = C(1.0, (1e-5, 1e8)) * RBF(1.0, (1e-5, 1e8))

# 创建高斯过程回归模型并进行训练
reg1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)
reg2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)
reg3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)

# 拟合模型
reg1.fit(x_train, y_train1)
reg2.fit(x_train, y_train2)
reg3.fit(x_train, y_train3)

# 目标函数
def objective_function(x1, x2,):
    return x1+x2

# ==============================================================================
#                 实验配置参数
# ==============================================================================
N = 10000 # 生成蒙特卡洛样本的数量
threshold = 0 # 约束阈值
x1_range = [0, 10] # x1的范围
x2_range = [0, 10] # x2的范围
reliability_target = 0.98 # 目标可靠性
temperature = 0.2 # 温度参数，控制生成文本的随机性
top_p = 0.9 #  nucleus sampling 参数，控制生成文本的多样性
max_iterations = 50 # 最大迭代次数
stagnation_limit = 10 # 停滞限制次数
penalty_limit = 0.01 # 惩罚限制值
retain_number = 5 # 保留数量
original_ranges = {
    "x1_range": x1_range,
    "x2_range": x2_range,
} # 原始变量范围
target_range = [0, 100] # 目标变量范围
std_dev = np.array([0.3464, 0.3464])  # 目标变量的标准差
penalty_weight = 10000 # 惩罚权重
adition_point_number=10 # 新增点的数量
adition_point_std = np.array([0.3464, 0.3464]) # 新增点的标准差
client = create_client("deepseek") # 创建 deepseek 客户端
model = "deepseek-chat" # 模型名称
template_path = os.path.join(ROOT, "Scripts", "prompt_template_Chinese.md") # 提示模板路径
max_tokens = 512 # 最大 tokens 数


if __name__ == "__main__":
    # 步骤 1：在范围内生成初始点集合（整数映射依赖于 llm_ops 内部的映射工具）
    ranges = [x1_range, x2_range]
    initial_point = llm_generate_initial_points(ranges, num_points=20)

    # 步骤 2：调用 LLM 优化流程（传入全部显式参数与代理模型、目标函数）
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
        constraint_source=[reg1, reg2, reg3],
        objective_fn=objective_function,
        verbose=True,
        plot=True,
        print_prompt=True,
        return_details=True,
    )
    # 步骤 4：输出结果（最优点、最优成本与实际迭代次数）
    print(f"Best point: {best_point}, Best cost: {best_cost}, Best reliabilities: {best_reliabilities}, Best penalty: {best_penalty}, Actual iterations: {actual_iterations}")


