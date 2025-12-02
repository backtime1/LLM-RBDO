"""
拉丁超立方抽样（Latin Hypercube Sampling）实现
用于二维多约束问题的参数采样和约束计算
"""

import numpy as np
from pyDOE import lhs
import pandas as pd
import os
from datetime import datetime

def compute_constraints(x):
    """
    计算给定参数样本的约束条件
    
    Args:
        x: numpy.ndarray, 形状为 (n_samples, 2) 的参数矩阵
        
    Returns:
        numpy.ndarray: 形状为 (n_samples, 3) 的约束条件矩阵
    """
    # 提取参数
    x1, x2 = x[:, 0], x[:, 1]
    
    # 初始化约束条件矩阵
    ceq = np.zeros((x.shape[0], 3))

    # 计算各个约束条件
    ceq[:, 0] = x1**2 * x2 / 20 - 1
    ceq[:, 1] = (x1 + x2 - 5)**2 / 30 + (x1 - x2 - 12)**2 / 120 - 1
    ceq[:, 2] = 80 / (x1**2 + 8 * x2 - 5) - 1
    
    return ceq


def main():
    """主函数：执行拉丁超立方抽样并保存结果"""
    # 定义参数分布特征
    means = np.array([5, 5])
    std = np.array([2.5, 2.5])

    # 计算参数边界
    lower_bound = means - 2 * std
    upper_bound = means + 2 * std

    # 执行拉丁超立方抽样
    num_samples = 100
    latin_hypercube_samples = lhs(2, samples=num_samples, criterion='maximin')

    # 缩放抽样结果
    scaled_samples = lower_bound + latin_hypercube_samples * (upper_bound - lower_bound)

    # 计算约束条件
    constraints = compute_constraints(scaled_samples)

    # 合并结果
    combined_results = np.hstack((scaled_samples, constraints))

    # 创建DataFrame并保存
    columns = ['x1', 'x2', 'ceq1', 'ceq2', 'ceq3']
    df = pd.DataFrame(combined_results, columns=columns)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(os.path.dirname(__file__), f'multy_cons_2D_{ts}.csv')
    df.to_csv(out_path, index=False)

    print("拉丁超立方抽样结果及对应的约束条件:\n", combined_results)
    print(f"结果已保存到 {out_path} 文件中。")

if __name__ == "__main__":
    main()
