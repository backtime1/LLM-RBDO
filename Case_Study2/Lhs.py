"""
拉丁超立方抽样（Latin Hypercube Sampling）实现
用于汽车碰撞问题的参数采样和约束计算
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
        x: numpy.ndarray, 形状为 (n_samples, 11) 的参数矩阵
        
    Returns:
        numpy.ndarray: 形状为 (n_samples, 10) 的约束条件矩阵
    """
    # 提取参数
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = [x[:, i] for i in range(11)]
    
    # 目标值
    pc = np.array([1.0, 32.0, 32.0, 32.0, 0.32, 0.32, 0.32, 4.0, 9.9, 15.7])
    
    # 初始化约束条件矩阵
    ceq = np.zeros((x.shape[0], 10))

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

    return ceq


def main():
    """主函数：执行拉丁超立方抽样并保存结果"""
    # 定义参数边界
    lower_bound = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.192, 0.192, -30, -30])
    upper_bound = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.345, 0.345, 30, 30])

    # 执行拉丁超立方抽样
    num_samples = 200
    latin_hypercube_samples = lhs(11, samples=num_samples, criterion='maximin')

    # 缩放抽样结果
    scaled_samples = lower_bound + latin_hypercube_samples * (upper_bound - lower_bound)

    # 计算约束条件
    constraints = compute_constraints(scaled_samples)

    # 合并结果
    combined_results = np.hstack((scaled_samples, constraints))

    # 创建DataFrame并保存
    columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
               'ceq1', 'ceq2', 'ceq3', 'ceq4', 'ceq5', 'ceq6', 'ceq7', 'ceq8', 'ceq9', 'ceq10']
    df = pd.DataFrame(combined_results, columns=columns)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(os.path.dirname(__file__), f'car_crash_{ts}.csv')
    df.to_csv(out_path, index=False)

    print("拉丁超立方抽样结果及对应的约束条件:\n", combined_results)
    print(f"结果已保存到 {out_path} 文件中。")

if __name__ == "__main__":
    main()
