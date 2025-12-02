import numpy as np
import os
import sys
from matplotlib import rcParams

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Scripts.llm_ops import optimize_with_llm as llm_optimize_with_llm
from Scripts.llm_ops import generate_initial_points_with_numpy as llm_generate_initial_points
from Scripts.api_client import create_client
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "multy_cons_2D.csv"))
x = np.genfromtxt(csv_path, delimiter=",")
data = np.array(x)
x_train = data[1:, :-3]
y_train1 = data[1:, -3]
y_train2 = data[1:, -2]
y_train3 = data[1:, -1]
kernel = C(1.0, (1e-5, 1e8)) * RBF(1.0, (1e-5, 1e8))
reg1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)
reg2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)
reg3 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, normalize_y=True)
reg1.fit(x_train, y_train1)
reg2.fit(x_train, y_train2)
reg3.fit(x_train, y_train3)

def objective_function(x1, x2):
    return x1 + x2

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
client = create_client("deepseek")
model = "deepseek-chat"
templates = {
    "prompt_template_Chinese_Interrogative_tone": os.path.join(ROOT, "Scripts", "prompt_template_Chinese_Interrogative_tone.md"),
    "prompt_template_Chinese_Short": os.path.join(ROOT, "Scripts", "prompt_template_Chinese_Short.md"),
    "prompt_template_Chinese": os.path.join(ROOT, "Scripts", "prompt_template_Chinese.md"),
}
max_tokens = 512 # 最大 tokens 数
num_runs_per_setting = 2
ranges = [x1_range, x2_range]
initial_point = llm_generate_initial_points(ranges, num_points=20)
results = {}

for prompt_name, template_path in templates.items():
    print(f"\n--- 正在测试 Prompt: {prompt_name} ---")
    best_points = []
    best_costs = []
    best_rels_list = []
    iterations_list = []

    for _ in range(num_runs_per_setting):
        print(f"\n--- 正在运行第 {_+1} 次 --- (Prompt: {prompt_name})")
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
            verbose=False,
            plot=False,
            print_prompt=False,
            return_details=True,
        )
        best_points.append(best_point)
        best_costs.append(best_cost)
        best_rels_list.append(np.array(best_reliabilities))
        iterations_list.append(actual_iterations)
        
        print(f"本次运行结果：最优点: {best_point}, 最优成本: {best_cost}, 最优可靠性为{best_reliabilities}, 迭代次数: {actual_iterations}")
        print("-----------------------------------------------------------------------------------------------------------------------------")

    best_points_arr = np.array(best_points)
    avg_best_point = best_points_arr.mean(axis=0)
    std_best_point = best_points_arr.std(axis=0)
    avg_best_cost = float(np.mean(best_costs))
    std_best_cost = float(np.std(best_costs))
    avg_iterations = float(np.mean(iterations_list))
    std_iterations = float(np.std(iterations_list))
    rels_arr = np.vstack(best_rels_list)
    avg_reliability_G1 = float(np.mean(rels_arr[:, 0]))
    avg_reliability_G2 = float(np.mean(rels_arr[:, 1]))
    avg_reliability_G3 = float(np.mean(rels_arr[:, 2]))

    print(f"\n--- Prompt {prompt_name} 的统计结果 ---")
    print(f"平均最优点: {avg_best_point}")
    print(f"最优点标准差: {std_best_point}")
    print(f"平均最优成本: {avg_best_cost:.4f}")
    print(f"最优成本标准差: {std_best_cost:.4f}")
    print(f"平均迭代次数: {avg_iterations:.2f}")
    print(f"迭代次数标准差: {std_iterations:.2f}")
    print(f"平均可靠性分别为{avg_reliability_G1:.4f},{avg_reliability_G2:.4f},{avg_reliability_G3:.4f}")
    print("=============================================================================================================================")

    results[prompt_name] = {
        "avg_best_point": avg_best_point,
        "std_best_point": std_best_point,
        "avg_best_cost": avg_best_cost,
        "std_best_cost": std_best_cost,
        "avg_iterations": avg_iterations,
        "std_iterations": std_iterations,
        "avg_reliability_G1": avg_reliability_G1,
        "avg_reliability_G2": avg_reliability_G2,
        "avg_reliability_G3": avg_reliability_G3,
    }

print("\n所有 Prompt 对比实验完成！")
print("最终统计结果:")
print("=" * 130)
print(f"{'Prompt':<10} {'平均最优点':<10} {'最优点标准差':<10} {'平均成本':<10} {'成本标准差':<10} {'平均迭代':<10} {'迭代标准差':<10} {'AvgRel G1':<10} {'G2':<10} {'G3':<10}")
print("-" * 130)
for p_name, res in results.items():
    avg_point_str = "[" + ", ".join(f"{v:.3f}" for v in np.array(res['avg_best_point']).ravel()) + "]"
    std_point_str = "[" + ", ".join(f"{v:.3f}" for v in np.array(res['std_best_point']).ravel()) + "]"
    print(f"{p_name:<12} {avg_point_str:<10} {std_point_str:<10} {res['avg_best_cost']:<10.4f} {res['std_best_cost']:<10.4f} {res['avg_iterations']:<10.2f} {res['std_iterations']:<10.2f} {res['avg_reliability_G1']:<10.4f} {res['avg_reliability_G2']:<10.4f} {res['avg_reliability_G3']:<10.4f}")

