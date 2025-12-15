"""LLM-RBDO 优化操作模块

提供以下核心功能：
- 根据历史迭代信息调用 LLM 生成新的候选设计点（连续空间）
- 在约束代理模型下进行带惩罚的迭代优化（支持可行优先）
- 在给定范围内均匀采样生成初始设计点
"""

import numpy as np
import json
import os
import sys
import inspect
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Scripts.rbdo_utils import penalized_cost
from Scripts.mapping_utils import map_float_to_int_array, map_back_to_float_array

def generate_new_point_with_llm(messages, best_point_message, temperature, top_p, original_ranges, target_range, client, max_tokens, model, template_path, print_prompt=True):
    """根据历史消息与当前最优点生成一个新的候选设计点

    参数：
    - messages: 列表，每个元素为包含 'iteration'、'point'、'penalty'、'objective' 的字典
    - best_point_message: 字典，描述最近的最优点（同样包含点、penalty 与 objective）
    - temperature: LLM 采样温度
    - top_p: 核采样阈值
    - original_ranges: 设计空间范围字典，键形如 'x{i}_range'
    - target_range: 整数映射区间 [min, max]
    - client: OpenAI 兼容客户端实例
    - max_tokens: 生成长度上限
    - model: 模型名称
    - template_path: 提示模板路径
    - print_prompt: 是否打印提示信息

    返回：
    - numpy.ndarray，新生成的连续空间设计点（形如 [x1, x2, ...]）
    """
    names = []
    for k in sorted(original_ranges.keys(), key=lambda s: int("".join(filter(str.isdigit, s)) or "0")):
        names.append(k.split("_")[0])
    history_lines = ""
    for m in messages:
        history_lines += f"迭代次数{m['iteration']},生成点: {m['point']}, penalty: {m['penalty']},目标函数值: {m['objective']}\n"
    best_section = f"生成点: {best_point_message['point']}, penalty: {best_point_message['penalty']},目标函数值: {best_point_message['objective']}\n"
    ranges_lines = ""
    for name in names:
        ranges_lines += f"{name}: [{target_range[0]}, {target_range[1]}]\n"
    schema = "[\n    {" + ", ".join([f"\"{name}\": " for name in names]) + "}\n]"
    with open(template_path, "r", encoding="utf-8") as f:
        tpl = f.read()
    tpl = tpl.replace("<<VARIABLE_NAMES>>", ", ".join(names))
    tpl = tpl.replace("<<RANGES>>", ranges_lines.strip())
    tpl = tpl.replace("<<HISTORY>>", history_lines.strip())
    tpl = tpl.replace("<<BEST>>", best_section.strip())
    tpl = tpl.replace("<<OUTPUT_SCHEMA>>", schema)
    full_prompt = tpl
    if print_prompt:
        print("\n---LLM Prompt---")
        print(full_prompt)
        print("---LLM Prompt End---\n")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "你是一个优化算法助手，目的是寻找到一组解让penalty为0的前提下尽可能降低objective。"},
                      {"role": "user", "content": full_prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        new_point_str = response.choices[0].message.content.strip()
        start = new_point_str.rindex('[')
        end = new_point_str.rindex(']') + 1
        new_point_json = new_point_str[start:end].strip()
        new_point = json.loads(new_point_json)
        mapped_point = map_back_to_float_array(new_point[0], original_ranges, target_range)
        return mapped_point
    except Exception:
        mapped_best = map_back_to_float_array(best_point_message["point"], original_ranges, target_range)
        return mapped_best

def optimize_with_llm(initial_point, reliability_target, max_iterations, stagnation_limit,
                      penalty_limit, penalty_weight, temperature, top_p, retain_number, adition_point_number,
                      N, threshold, original_ranges, target_range, client, max_tokens, std, adition_point_std,
                      constraint_source, objective_fn, model, template_path, verbose=False, plot=True, print_prompt=True, return_details=False,
                      expand_point=None):
    """基于 LLM 的迭代优化（带约束惩罚与局部扰动搜索）

    参数：
    - initial_point: numpy.ndarray，形状 (K, D)，K 为初始点数量，D 为变量维度
    - reliability_target: 标量或数组，约束可靠性目标（长度等于约束数）
    - max_iterations: 最大迭代次数
    - stagnation_limit: 允许停滞（无提升）的迭代上限
    - penalty_limit: 接受的新点惩罚阈值（新点的 penalty 需不超过该值）
    - penalty_weight: 标量或数组，对各约束的惩罚权重
    - temperature/top_p: LLM 采样超参数
    - retain_number: 历史消息保留条数（超过后进行截断）
    - adition_point_number: 每次在 LLM 点附近评估的扰动点数量
    - N: Monte Carlo 采样数量
    - threshold: 标量或数组，约束判别阈值
    - original_ranges/target_range: 设计空间范围及整数映射区间
    - client: OpenAI 兼容客户端实例
    - std: 标准差（标量或数组，维度与设计变量一致），用于生成样本与可靠性评估
    - adition_point_std: 扰动点标准差（标量或数组），与 std 解耦，用于局部搜索
    - model/template_path: LLM 模型与提示模板
    - constraint_source: 约束来源（可为模型列表或真实约束函数）
    - objective_fn: 目标函数，接受向量 x 或解包后的 *x
    - verbose: 是否打印所有点的可靠性和目标函数值详细信息（默认 False）
    - plot: 是否绘制优化过程（默认 True）
    - print_prompt: 是否打印 LLM 输入提示（默认 True）
    - return_details: 是否返回结果更详细信息（默认 False）
    - expand_point: 扩展点函数，接受 numpy.ndarray 并返回扩展后的点（默认 None）。该选项适用于优化变量与随机变量不一致的情况下。

    返回：return_details=False时返回(best_point, best_cost, actual_iterations)
    return_details=True时返回(best_point, best_cost, best_reliabilities, actual_iterations)
    """
    messages = []
    best_points_history = []
    current_points = initial_point
    d_design = len(original_ranges)
    expand_fn = expand_point if callable(expand_point) else (lambda x: x)

    # 评估初始点集合：可行优先（penalty == 0），否则选择 penalty 最小者作为当前点
    penalty_objective_list = []
    for point in current_points:
        point_full = expand_fn(point)
        p, c, rels = penalized_cost(
            point_full,
            N=N,
            threshold=threshold,
            reliability_target=reliability_target,
            constraint_source=constraint_source,
            objective_fn=objective_fn,
            std=std,
            penalty_weight=penalty_weight,
            verbose=verbose,
            return_reliabilities=True,
        )
        penalty_objective_list.append({"penalty": p, "cost": c, "reliabilities": rels})

    penalty = [item["penalty"] for item in penalty_objective_list]
    objective = [item["cost"] for item in penalty_objective_list]

    valid_indices = [i for i, p in enumerate(penalty) if p == 0]

    if valid_indices:
        valid_objectives = [objective[i] for i in valid_indices]
        best_index_in_valid = np.argmin(valid_objectives)
        best_index = valid_indices[best_index_in_valid]
    else:
        best_index = np.argmin(penalty)

    best_point = current_points[best_index]
    current_point = best_point
    best_cost = objective[best_index]
    best_penalty = penalty[best_index]

    best_reliabilities = penalty_objective_list[best_index]["reliabilities"]
    best_points_history.append({
        "iteration": 0,
        "best_point": best_point.tolist(),
        "best_cost": best_cost,
        "best_penalty": best_penalty,
        "best_reliabilities": best_reliabilities.tolist() if hasattr(best_reliabilities, "tolist") else best_reliabilities,
    })
    best_costs = []
    new_cost = best_cost
    new_penalty = best_penalty
    stagnation_count = 0

    actual_iterations = 0

    for iteration in range(max_iterations):
        actual_iterations = iteration + 1
        print(f"第 {iteration + 1} 次迭代，最优点：{best_point}，最优成本：{best_cost}，最优点的惩罚值：{best_penalty}, 最优点的可靠性：{best_reliabilities}")
        # 将连续空间点映射到整数编码，便于提示模板对齐
        mapped_current_point = map_float_to_int_array(
            current_point.tolist(),
            original_ranges,
            target_range=target_range
        )
        messages.append({"iteration": iteration + 1, "point": mapped_current_point, "penalty": new_penalty, "objective": new_cost})
        # 最近最优点也进行编码，作为提示中的“最佳参考”
        latest_best_point_int = map_float_to_int_array(
            best_points_history[-1]["best_point"],
            original_ranges,
            target_range=target_range
        )
        best_point_message = {"iteration": iteration + 1, "point": latest_best_point_int, "penalty": best_points_history[-1]["best_penalty"], "objective": best_points_history[-1]["best_cost"]}
        if len(messages) > retain_number:
            messages = messages[-retain_number:]
        best_costs.append(best_cost)

        # 调用 LLM 生成新点，并在其附近按 adition_point_std 进行高斯扰动采样
        new_point_llm = generate_new_point_with_llm(messages, best_point_message, temperature, top_p, original_ranges, target_range, client, max_tokens, model=model, template_path=template_path, print_prompt=print_prompt)
        new_point_full = expand_fn(new_point_llm)
        addition_points = [new_point_full + np.random.normal(0, adition_point_std) for _ in range(adition_point_number)]
        ranges_list = [original_ranges[f"x{i+1}_range"] for i in range(d_design)]
        def in_range(point):
            for i in range(d_design):
                r = ranges_list[i]
                if not (r[0] <= point[i] <= r[1]):
                    return False
            return True
        valid_points = [p for p in addition_points if in_range(p)]
        if len(valid_points) == 0:
            if in_range(new_point_full):
                print("警告: 没有有效的扰动点在指定范围内。将使用LLM直接生成的点进行评估。")
                valid_points = [new_point_full]
            else:
                while True:
                    new_point_llm = generate_new_point_with_llm(messages, best_point_message, temperature, top_p, original_ranges, target_range, client, max_tokens, model=model, template_path=template_path, print_prompt=print_prompt)
                    new_point_full = expand_fn(new_point_llm)
                    addition_points = [new_point_full + np.random.normal(0, adition_point_std) for _ in range(adition_point_number)]
                    valid_points = [p for p in addition_points if in_range(p)]
                    if len(valid_points) == 0 and in_range(new_point_full):
                        print("警告: 没有有效的扰动点在指定范围内。将使用LLM直接生成的点进行评估。")
                        valid_points = [new_point_full]
                    if len(valid_points) > 0:
                        break
        addition_points = valid_points

        group_results = []
        for point in addition_points:
            p, c, rels = penalized_cost(
                point,
                N=N,
                threshold=threshold,
                reliability_target=reliability_target,
                constraint_source=constraint_source,
                objective_fn=objective_fn,
                std=std,
                penalty_weight=penalty_weight,
                verbose=verbose,
                return_reliabilities=True,
            )
            group_results.append({"point": point, "penalty": p, "cost": c, "reliabilities": rels})

        # 选择规则：优先在可行解集内选成本最小者；若无可行解，则选惩罚最小者
        if any(r["penalty"] == 0 for r in group_results):
            best_result_group = min((r for r in group_results if r["penalty"] == 0), key=lambda r: r["cost"])
        else:
            best_result_group = min(group_results, key=lambda r: r["penalty"])

        new_point_full = best_result_group["point"]
        new_penalty = best_result_group["penalty"]
        new_cost = best_result_group["cost"]
        new_reliabilities = best_result_group["reliabilities"]

        # 只有当成本更低且惩罚不超过阈值时，才更新最优记录（避免不可行解“误优”）
        if new_cost < best_cost and new_penalty <= penalty_limit:
            best_point = new_point_full[:d_design]
            best_cost = new_cost
            best_penalty = new_penalty
            stagnation_count = 0
            best_reliabilities = new_reliabilities
            best_points_history.append({
                "iteration": iteration + 1,
                "best_point": best_point.tolist(),
                "best_cost": best_cost,
                "best_penalty": best_penalty,
                "best_reliabilities": best_reliabilities.tolist() if hasattr(best_reliabilities, "tolist") else best_reliabilities,
            })
        else:
            stagnation_count += 1

        # 连续停滞达到上限则停止迭代
        if stagnation_count >= stagnation_limit:
            break

        current_point = new_point_full[:d_design]
        print("=============================================================================================================================")

    if plot:
        rcParams['font.family'] = 'Times New Roman'
        rcParams['axes.unicode_minus'] = False
        plt.plot(range(1, len(best_costs) + 1), best_costs, marker='o', linestyle='-')
        plt.xlabel('Iteration', fontsize=18)
        plt.ylabel('Optimal Cost', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tight_layout()
        try:
            caller_file = inspect.stack()[1].filename
        except Exception:
            caller_file = getattr(sys.modules.get('__main__'), '__file__', None)
        caller_dir = os.path.dirname(caller_file) if caller_file else os.getcwd()
        fig_dir = os.path.join(caller_dir, 'Figures')
        os.makedirs(fig_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(caller_file or 'plot'))[0]
        fig_path = os.path.join(fig_dir, f"{base_name}_iteration_optimal_cost_600dpi.png")
        plt.savefig(fig_path, dpi=600)
        plt.show()

    if return_details:
        return best_point, best_cost, best_reliabilities, best_penalty, actual_iterations
    return best_point, best_cost, actual_iterations

def generate_initial_points_with_numpy(ranges, num_points):
    """在每维范围内均匀采样生成初始设计点

    参数：
    - ranges: 列表形如 [[min, max], ...]，长度为维度 D
    - num_points: 生成点数量

    返回：
    - numpy.ndarray，形状 (num_points, D)
    """
    if not all(len(r) == 2 for r in ranges):
        raise ValueError("每个范围必须包含 [min, max] 两个值")

    points_array = []
    for _ in range(num_points):
        point = [np.random.uniform(r[0], r[1]) for r in ranges]
        points_array.append(point)

    result_points = np.array(points_array)
    formatted_points = [
        {"x" + str(i + 1): p[i] for i in range(len(p))}
        for p in result_points
    ]
    print("生成的初始点集合（JSON 格式）：\n", json.dumps(formatted_points, indent=4))
    return result_points
