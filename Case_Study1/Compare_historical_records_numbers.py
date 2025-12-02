import numpy as np
import os
import sys
import matplotlib.pyplot as plt

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
stagnation_limit=10
penalty_limit = 0.01
std_dev = np.array([0.3464, 0.3464])
penalty_weight = 10000
adition_point_std = np.array([0.3464, 0.3464])
adition_point_number = 10
original_ranges = {
    "x1_range": x1_range,
    "x2_range": x2_range,
}
target_range = [0, 100]
client = create_client("deepseek")
model = "deepseek-chat"
template_path = os.path.join(ROOT, "Scripts", "prompt_template_Chinese.md")
max_tokens = 512

num_runs_per_setting = 2
retain_numbers = [3, 5]

ranges = [x1_range, x2_range]
initial_point = llm_generate_initial_points(ranges, num_points=20)
results = {}

for retain_number in retain_numbers:
    print(f"\n--- 正在测试 retain_number: {retain_number} ---")
    best_points = []
    best_costs = []
    best_rels_list = []
    iterations_list = []

    for i in range(num_runs_per_setting):
        print(f"\n--- 正在运行第 {i+1} 次 --- (retain_number: {retain_number})")
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
            print_prompt=True,
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

    print(f"\n--- retain_number {retain_number} 的统计结果 ---")
    print(f"平均最优点: {avg_best_point}")
    print(f"最优点标准差: {std_best_point}")
    print(f"平均最优成本: {avg_best_cost:.4f}")
    print(f"最优成本标准差: {std_best_cost:.4f}")
    print(f"平均迭代次数: {avg_iterations:.2f}")
    print(f"迭代次数标准差: {std_iterations:.2f}")
    print(f"平均可靠性分别为{avg_reliability_G1:.4f},{avg_reliability_G2:.4f},{avg_reliability_G3:.4f}")
    print("=============================================================================================================================")

    results[str(retain_number)] = {
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

print("\n所有 retain_number 对比实验完成！")
print("最终统计结果:")
print("=" * 130)
print(f"{'保留条数':<10} {'平均最优点':<10} {'最优点标准差':<10} {'平均成本':<10} {'成本标准差':<10} {'平均迭代':<10} {'迭代标准差':<10} {'AvgRel G1':<10} {'G2':<10} {'G3':<10}")
print("-" * 130)

for n in retain_numbers:
    res = results[str(n)]
    avg_point_str = "[" + ", ".join(f"{v:.3f}" for v in np.array(res['avg_best_point']).ravel()) + "]"
    std_point_str = "[" + ", ".join(f"{v:.3f}" for v in np.array(res['std_best_point']).ravel()) + "]"
    print(f"{n:<12} {avg_point_str:<10} {std_point_str:<10} {res['avg_best_cost']:<10.4f} {res['std_best_cost']:<10.4f} {res['avg_iterations']:<10.2f} {res['std_iterations']:<10.2f} {res['avg_reliability_G1']:<10.4f} {res['avg_reliability_G2']:<10.4f} {res['avg_reliability_G3']:<10.4f}")

retains = retain_numbers
optimal_costs = [results[str(n)]["avg_best_cost"] for n in retains]
optimal_costs_std = [results[str(n)]["std_best_cost"] for n in retains]
iteration_counts = [results[str(n)]["avg_iterations"] for n in retains]
iteration_counts_std = [results[str(n)]["std_iterations"] for n in retains]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
fig_dir = os.path.join(os.path.dirname(__file__), "Figures")
os.makedirs(fig_dir, exist_ok=True)

fig1, ax1 = plt.subplots()
x = np.array(retains)
y = np.array(optimal_costs)
yerr = np.array(optimal_costs_std)
ax1.plot(x, y, 'o-', linewidth=3, markersize=8, color='#2E86AB',
         markerfacecolor='#2E86AB', markeredgecolor='white', markeredgewidth=2,
         label='Optimal Cost')
ax1.fill_between(x, y - yerr, y + yerr, alpha=0.3, color='#2E86AB')
ax1.set_xlabel('Number of Historical Records Retained', fontsize=18)
ax1.set_ylabel('Optimal Cost', fontsize=18)
ax1.legend(fontsize=14, loc='upper right')
ax1.set_xticks(retains)
ax1.set_xticklabels([str(num) for num in retains])
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_ylim(6, 7)
plt.tight_layout()
fig1.savefig(os.path.join(fig_dir, "historical_records_optimal_cost_600dpi.png"), dpi=600)
plt.show()

fig2, ax2 = plt.subplots()
y2 = np.array(iteration_counts)
yerr2 = np.array(iteration_counts_std)
ax2.plot(x, y2, 's-', linewidth=3, markersize=8, color='#A23B72',
         markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2,
         label='Iteration Count')
ax2.fill_between(x, y2 - yerr2, y2 + yerr2, alpha=0.3, color='#A23B72')
ax2.set_xlabel('Number of Historical Records Retained', fontsize=18)
ax2.set_ylabel('Iteration Count', fontsize=18)
ax2.legend(fontsize=14, loc='upper right')
ax2.set_xticks(retains)
ax2.set_xticklabels([str(num) for num in retains])
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_ylim(10, 50)
plt.tight_layout()
fig2.savefig(os.path.join(fig_dir, "historical_records_iterations_600dpi.png"), dpi=600)
plt.show()

