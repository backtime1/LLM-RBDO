import numpy as np
from pyDOE import lhs
from nozzle import simulate_nozzle_and_get_max_mach
import pandas as pd
import os
from datetime import datetime
# 设置均值和方差
# 计算下限和上限
means = np.array([2/3, 0.45, 0.468])
std = np.array([1/150, 0.0045, 0.00468])

# 计算下限和上限
lower_bound = means - 3 * std
upper_bound = means + 3 * std
# 执行拉丁超立方抽样
num_samples = 50  # 设置抽样数量
latin_hypercube_samples = lhs(3, samples=num_samples, criterion='maximin')

# 缩放抽样结果到指定范围
scaled_samples = lower_bound + latin_hypercube_samples * (upper_bound - lower_bound)

# 输出结果
rounded_samples = scaled_samples
print("拉丁超立方抽样结果:\n", rounded_samples)

# 调用仿真函数
max_mach_results = []
for i, sample in enumerate(rounded_samples):
    x1, x2, x3 = sample  # 提取每个样本点的参数
    print(f"\n正在处理样本点 {i + 1}: x1={x1}, x2={x2}, x3={x3}")

    try:
        # 调用仿真函数
        max_mach = simulate_nozzle_and_get_max_mach(x1, x2, x3, cleanup_tmp=True,use_project_temp=False)# 不使用项目临时目录
        max_mach_results.append(max_mach)
        print(f"马赫数最大值为: {max_mach}")
    except Exception as e:
        print(f"仿真失败: {e}")
        max_mach_results.append(np.nan)  # 如果失败，记录为 NaN

# 输出所有样本点的仿真结果
print("\n所有样本点的仿真结果:")
for i, (sample, max_mach) in enumerate(zip(rounded_samples, max_mach_results)):
    print(f"样本点 {i + 1}: x1={sample[0]}, x2={sample[1]}, x3={sample[2]}, 最大马赫数={max_mach}")

# 保存结果到 CSV 文件
results_df = pd.DataFrame({
    "x1": rounded_samples[:, 0],
    "x2": rounded_samples[:, 1],
    "x3": rounded_samples[:, 2],
    "Max_Mach": max_mach_results
})
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = os.path.join(os.path.dirname(__file__), f'simulation_results_{ts}.csv')
results_df.to_csv(out_path, index=False)
print(f"仿真结果已保存到 {out_path}")
