import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
# 提供的 f_min 数据
iterations = np.arange(1, 22)  # 对应的代数 (0, 1, ..., 20)
f_min_values = [
    2.53415E+00, 2.52689E+00, 2.48778E+00, 2.48778E+00, 2.48778E+00,
    2.45678E+00, 2.45678E+00, 2.45626E+00, 2.45224E+00, 2.45224E+00,
    2.45089E+00, 2.45016E+00, 2.45016E+00, 2.45016E+00, 2.44977E+00,
    2.44977E+00, 2.44966E+00, 2.44929E+00, 2.44929E+00, 2.44929E+00,
    2.44929E+00
]

# 绘制 f_min 的迭代曲线图
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(10, 6))
plt.plot(iterations, f_min_values, marker='o',  linestyle='-')
# plt.title('Relationship Between Iteration and Optimal Cost', fontsize=20)
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Optimal Cost', fontsize=18)
plt.grid(False)
# 设置刻度大小
plt.tick_params(axis='both', which='major', labelsize=15)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.show()