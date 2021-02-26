import numpy as np

x_d_1 = np.array([200, 95, 85, 75, 65, 55, 45, 1000])
x_d_2 = np.array([50, 73, 72, 71, 69, 68, 67, 1000])

# 平均值 mean
x_d_1_m = np.mean(x_d_1)
x_d_2_m = np.mean(x_d_2)
print(x_d_1_m)
print(x_d_2_m)

# 方差 variance
x_d_1_v = np.square(x_d_1 - x_d_1_m)
x_d_2_v = np.square(x_d_2 - x_d_2_m)

# 总体方差
x_d_1_s_v = np.mean(x_d_1_v)
x_d_2_s_v = np.mean(x_d_2_v)

# 总体标准差 standard deviation
x_d_1_s_d = np.sqrt(x_d_1_s_v)
x_d_2_s_d = np.sqrt(x_d_2_s_v)
print(x_d_1_s_d)
print(x_d_2_s_d)

# 标准化 Standardization
x_d_1_s = (x_d_1 - x_d_1_m) / x_d_1_s_d
x_d_2_s = (x_d_2 - x_d_2_m) / x_d_2_s_d
print(x_d_1_s)
print(x_d_2_s)

# 标准化后的数据的均值和方差
print(np.mean(x_d_1_s))
print(np.mean(x_d_2_s))
print(np.mean(np.square(x_d_1_s - np.mean(x_d_1_s))))
print(np.mean(np.square(x_d_2_s - np.mean(x_d_2_s))))
