import numpy as np
from pydantic.type_adapter import P

def generate_samples(x0, stdx, N):
    """
    生成围绕设计点的正态随机样本。
    参数：
    - x0: 设计点数组（形如 [x1, x2, ...]）
    - stdx: 每个设计变量的标准差数组（与 x0 同维）
    - N: 样本数量
    返回：
    - 形状为 (N, len(x0)) 的样本矩阵
    """
    return np.random.normal(x0, stdx, (N, len(x0)))

def compute_constraints(constraint_source, X):
    """
    使用代理模型或真实约束函数对样本批量计算约束响应。
    - 若 constraint_source 为可调用对象（函数），则调用其：constraint_source(X) -> (N, m)
    - 若 constraint_source 为模型列表，则依次预测并按列拼接。
    """
    if callable(constraint_source):
        return np.asarray(constraint_source(X))
    preds = [m.predict(X) for m in constraint_source]
    return np.column_stack(preds)

def reliability_analysis(x, N, std, threshold, constraint_source, objective_fn, verbose=False):
    """
    基于代理模型进行可靠性分析：在设计点 x 附近生成样本，
    以“约束响应 ≥ 阈值”的比例估计每条约束的可靠性，并计算目标函数值。
    支持任意数量的约束与任意维度的目标函数输入。
    参数：
    - x: 当前设计点（1D数组，长度为设计变量维度）
    - N: 采样数量
    - std: 采样标准差数组（与 x 同维）
    - threshold: 判定阈值（标量或长度为约束数量的数组）
    - models: 约束代理模型列表
    - objective_fn: 目标函数，可接受向量 x 或解包后的 *x
    返回：
    - (reliabilities, objective)，其中 reliabilities 为长度为约束数量的数组
    """
    samples = generate_samples(x, std, N)
    ceq = compute_constraints(constraint_source, samples)
    m = ceq.shape[1]
    threshold_arr = np.asarray(threshold)
    if threshold_arr.ndim == 0:
        threshold_arr = np.full(m, float(threshold_arr))
    else:
        if threshold_arr.shape[0] != m:
            raise ValueError("threshold 的长度必须等于约束数量")
    reliabilities = np.mean(ceq >= threshold_arr, axis=0)
    try:
        obj = objective_fn(x)
    except TypeError:
        obj = objective_fn(*x)
    if verbose:
        print(f"point{x}: reliabilities: {reliabilities}, objective: {obj}")# 打印可靠性和目标函数值
    return reliabilities, obj

def penalized_cost(x, N, threshold, reliability_target, constraint_source, objective_fn, std, penalty_weight, verbose=False, return_reliabilities=False):
    """
    计算带罚的成本：若任一约束可靠性低于目标值，则按二次罚累加；
    同时返回目标函数值，便于后续“可行优先，再优化目标”的选择策略。
    支持约束数量与目标函数维度的通用情形。
    参数：
    - x: 当前设计点（1D数组）
    - N: 采样数量
    - threshold: 判定阈值（标量或长度为约束数量的数组）
    - reliability_target: 可靠性目标（标量或长度为约束数量的数组）
    - models: 约束代理模型列表
    - objective_fn: 目标函数
    - std: 采样标准差数组
    - penalty_weight: 罚权重（标量或长度为约束数量的数组）
    返回：
    - (penalty, objective)
    """
    reliabilities, objective = reliability_analysis(x, N, std, threshold, constraint_source, objective_fn, verbose=verbose)
    m = len(reliabilities)
    target_arr = np.asarray(reliability_target)
    if target_arr.ndim == 0:
        target_arr = np.full(m, float(target_arr))
    else:
        if target_arr.shape[0] != m:
            raise ValueError("reliability_target 的长度必须等于约束数量")
    weight_arr = np.asarray(penalty_weight)
    if weight_arr.ndim == 0:
        weight_arr = np.full(m, float(weight_arr))
    else:
        if weight_arr.shape[0] != m:
            raise ValueError("penalty_weight 的长度必须等于约束数量")
    deficit = np.clip(target_arr - reliabilities, 0.0, None)
    penalty = float(np.sum(weight_arr * deficit ** 2))
    if return_reliabilities:
        return penalty, objective, reliabilities
    return penalty, objective
