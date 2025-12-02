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
    X_scaled = scaler_x.transform(X)
    preds = [
        reg1.predict(X_scaled),
        reg2.predict(X_scaled),
        reg3.predict(X_scaled),
        reg4.predict(X_scaled),
        reg5.predict(X_scaled),
        reg6.predict(X_scaled),
        reg7.predict(X_scaled),
        reg8.predict(X_scaled),
        reg9.predict(X_scaled),
        reg10.predict(X_scaled),
    ]
    ceq = np.column_stack(preds)
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

