import numpy as np

def float_to_int(value, original_range, target_range):
    """
    将连续变量值按线性比例映射到目标整数区间并四舍五入。
    参数：
    - value: 浮点值
    - original_range: 原始范围 [min_original, max_original]
    - target_range: 目标整数范围 [min_target, max_target]
    返回：
    - 映射后的整数值
    """
    min_original, max_original = original_range
    min_target, max_target = target_range
    return round((value - min_original) / (max_original - min_original) * (max_target - min_target) + min_target)

def map_float_to_int_array(point, original_ranges, target_range):
    """
    将多维连续设计点按各自范围线性映射到统一的整数区间。
    参数：
    - point: 设计点序列（如 [x1, x2, ...]）
    - original_ranges: 字典，键为 "x{i}_range"，值为对应变量的 [min, max]
    - target_range: 统一整数区间 [min_target, max_target]
    返回：
    - 整数数组（长度与设计维度一致）
    """
    int_point = []
    for i, value in enumerate(point):
        key = f"x{i + 1}_range"
        float_min, float_max = original_ranges[key]
        int_min, int_max = target_range
        mapped_value = int_min + (value - float_min) * (int_max - int_min) / (float_max - float_min)
        int_point.append(round(mapped_value))
    return int_point

def map_back_to_float_array(point, original_ranges, target_range):
    """
    将按整数区间编码的点映射回原始连续设计空间。
    参数：
    - point: 字典形式的点，如 {"x1": 12, "x2": 34, ...}
    - original_ranges: 字典，键为 "x{i}_range"，值为对应变量的 [min, max]
    - target_range: 整数区间 [min_target, max_target]
    返回：
    - 对应的连续设计点（numpy 数组）
    """
    mapped_point = []
    for key, value in point.items():
        param_range = original_ranges[f"{key}_range"]
        min_original, max_original = param_range
        min_target, max_target = target_range
        float_value = (value - min_target) / (max_target - min_target) * (max_original - min_original) + min_original
        mapped_point.append(float_value)
    return np.array(mapped_point)
