import numpy as np
import matplotlib.pyplot as plt
import time

# 进化前CP数据（10组）
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
# 进化后CP数据（10组）
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

# y_data = b + w * x_data (假设模型)

# 画图
fig = plt.figure(figsize=(7, 7))
plt.ion()

# bias (b)
x = np.arange(-200, -100, 1)
# weight (w)
y = np.arange(-5, 5, 0.1)
# 存放loss值
Z = np.zeros((len(x), len(y)))
# 生成坐标矩阵
X, Y = np.meshgrid(x, y)
# 遍历x，y，计算出每组数据的loss
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0  # meshgrid输出结果：y为行，x为列
        for n in range(len(x_data)):
            Z[j][i] += (y_data[n] - b - w * x_data[n]) ** 2
        # 平均loss
        Z[j][i] /= len(x_data)

# linear regression
# 随机初始化w, b
b = -120
w = -4
# learning rate (步长)
lr = 0.0000001
# 迭代数 1400000
iteration = 1400000

# 存储b, w, loss的历史值，用于画图
b_history = [b]
w_history = [w]
loss_history = []
start = time.time()
# 迭代
for i in range(iteration):
    # 对一个独立样本计算梯度，然后在所有样本上进行平均化，获得总梯度
    grad_b = 0.0
    grad_w = 0.0
    loss = 0.0
    for n in range(len(x_data)):
        grad_b += -2.0 * (y_data[n] - (b + w * x_data[n])) * 1.0
        grad_w += -2.0 * (y_data[n] - (b + w * x_data[n])) * x_data[n]
        loss += (y_data[n] - (b + w * x_data[n])) ** 2

    # update param
    b -= lr * grad_b / len(x_data)
    w -= lr * grad_w / len(x_data)
    loss /= len(x_data)

    b_history.append(b)
    w_history.append(w)
    loss_history.append(loss)

    if i % 10000 == 0:
        print("Step %i, w: %0.4f, b: %0.4f, Loss: %0.4f" % (i, w_history[i], b_history[i], loss_history[i]))

        # 更新图
        fig.clf()
        ax1 = fig.add_subplot(321)
        ax1.scatter(x_data, y_data)
        x_line = np.arange(0, 1000, 1)
        y_line = b_history[i] + w_history[i] * x_line
        ax1.plot(x_line, y_line)

        ax2 = fig.add_subplot(323, projection='3d')
        ax2.plot_surface(X, Y, Z)
        ax2.scatter(b_history[:-1], w_history[:-1], loss_history, 'o-', s=15, c="red")

        ax3 = fig.add_subplot(325)
        ax3.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
        ax3.plot([-188.4], [2.67], 'x', ms=12, mew=3, color="orange")
        ax3.plot(b_history[:-1], w_history[:-1], 'o-', ms=3, lw=1.5, color="black")

        ax4 = fig.add_subplot(322)
        ax4.plot(loss_history)

        plt.pause(0.1)

end = time.time()
print("大约需要时间：", end - start)

plt.ioff()
plt.show()
