"""
模拟过拟合
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# 进化前CP数据（10组）
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
# 进化后CP数据（10组）
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

# 改变1
# y_data = b + w_1 * x_data + w_2 * (x_data ** 2) + w_3 * (x_data ** 3) (假设模型)

# 画图
# 改变2：超过两个参数就不容易可视化了
fig = plt.figure(figsize=(7, 7))
plt.ion()

# linear regression
b = -120
w_1 = -4
w_2 = -4
w_3 = -4
# learning rate (步长)
lr = 1
# 迭代数
iteration = 120000

# 分别给b, w设置不同的learning rate
lr_b = 0
lr_w_1 = 0
lr_w_2 = 0
lr_w_3 = 0

# 存储b, w, loss的历史值，用于画图
b_history = [b]
w_1_history = [w_1]
w_2_history = [w_2]
w_3_history = [w_3]
loss_history = []
lr_b_history = []
lr_w_1_history = []
lr_w_2_history = []
lr_w_3_history = []
start = time.time()
# 迭代
for i in range(iteration):
    # 对一个独立样本计算梯度，然后在所有样本上进行平均化，获得总梯度
    grad_b = 0.0
    grad_w_1 = 0.0
    grad_w_2 = 0.0
    grad_w_3 = 0.0
    loss = 0.0
    for n in range(len(x_data)):
        grad_b += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3))) * 1.0
        grad_w_1 += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3))) \
            * x_data[n]
        grad_w_2 += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3))) \
            * (x_data[n] ** 2)
        grad_w_3 += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3))) \
            * (x_data[n] ** 3)
        loss += (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3))) ** 2

    grad_b /= len(x_data)
    grad_w_1 /= len(x_data)
    grad_w_2 /= len(x_data)
    grad_w_3 /= len(x_data)
    loss /= len(x_data)
    lr_b += grad_b ** 2
    lr_w_1 += grad_w_1 ** 2
    lr_w_2 += grad_w_2 ** 2
    lr_w_3 += grad_w_3 ** 2

    # update param
    b -= lr / np.sqrt(lr_b) * grad_b
    w_1 -= lr / np.sqrt(lr_w_1) * grad_w_1
    w_2 -= lr / np.sqrt(lr_w_2) * grad_w_2
    w_3 -= lr / np.sqrt(lr_w_3) * grad_w_3

    b_history.append(b)
    w_1_history.append(w_1)
    w_2_history.append(w_2)
    w_3_history.append(w_3)
    loss_history.append(loss)
    lr_b_history.append(lr / np.sqrt(lr_b))
    lr_w_1_history.append(lr / np.sqrt(lr_w_1))
    lr_w_2_history.append(lr / np.sqrt(lr_w_2))
    lr_w_3_history.append(lr / np.sqrt(lr_w_3))

    if i % 10000 == 0:
        print("Step %i, w_1: %0.4f, w_2: %0.4f, w_3: %0.4f, b: %0.4f, Loss: %0.4f" %
              (i, w_1_history[i], w_2_history[i], w_3_history[i], b_history[i], loss_history[i]))

        # 更新图
        fig.clf()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x_data, y_data)
        x_line = np.arange(0, 1500, 1)
        y_line = b_history[i] + w_1_history[i] * x_line + w_2_history[i] * (x_line ** 2) + \
            w_3_history[i] * (x_line ** 3)
        ax1.plot(x_line, y_line)
        plt.pause(0.1)

end = time.time()
print("大约需要时间：", end - start)

plt.ioff()
plt.show()
