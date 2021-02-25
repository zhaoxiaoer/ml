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
x_d = np.array(x_data)
y_d = np.array(y_data)

# 改变1
# y_data = b + w_1 * x_data + w_2 * (x_data ** 2) + w_3 * (x_data ** 3) + w_4 * (x_data ** 4) + \
# w_5 * (x_data ** 5) (假设模型)

# 画图
# 改变2：超过两个参数就不容易可视化了
fig = plt.figure(figsize=(7, 7))
plt.ion()

# linear regression
b = 99
w_1 = 0.000006
w_2 = 0.000001
w_3 = 0.0000005
w_4 = 0.00000006
w_5 = 0.0000000001
# learning rate (步长)
lr = 1
# 迭代数
iteration = 12000000

# 分别给b, w设置不同的learning rate
lr_b = 0
lr_w_1 = 0
lr_w_2 = 0
lr_w_3 = 0
lr_w_4 = 0
lr_w_5 = 0

# 存储b, w, loss的历史值，用于画图
b_history = [b]
w_1_history = [w_1]
w_2_history = [w_2]
w_3_history = [w_3]
w_4_history = [w_4]
w_5_history = [w_5]
loss_history = []
lr_b_history = []
lr_w_1_history = []
lr_w_2_history = []
lr_w_3_history = []
lr_w_4_history = []
lr_w_5_history = []
start = time.time()
# 迭代
for i in range(iteration):
    # 对一个独立样本计算梯度，然后在所有样本上进行平均化，获得总梯度
    grad_b = 0.0
    grad_w_1 = 0.0
    grad_w_2 = 0.0
    grad_w_3 = 0.0
    grad_w_4 = 0.0
    grad_w_5 = 0.0
    loss = 0.0
    for n in range(len(x_data)):
        loss += (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3) +
                              w_4 * (x_data[n] ** 4) + w_5 * (x_data[n] ** 5))) ** 2
        grad_b += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3) +
                                       + w_4 * (x_data[n] ** 4) + w_5 * (x_data[n] ** 5))) * 1.0
        grad_w_1 += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3) +
                                         w_4 * (x_data[n] ** 4) + w_5 * (x_data[n] ** 5))) * x_data[n]
        grad_w_2 += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3) +
                                         w_4 * (x_data[n] ** 4) + w_5 * (x_data[n] ** 5))) * (x_data[n] ** 2)
        grad_w_3 += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3) +
                                         w_4 * (x_data[n] ** 4) + w_5 * (x_data[n] ** 5))) * (x_data[n] ** 3)
        grad_w_4 += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3) +
                                         w_4 * (x_data[n] ** 4) + w_5 * (x_data[n] ** 5))) * (x_data[n] ** 4)
        grad_w_5 += -2.0 * (y_data[n] - (b + w_1 * x_data[n] + w_2 * (x_data[n] ** 2) + w_3 * (x_data[n] ** 3) +
                                         w_4 * (x_data[n] ** 4) + w_5 * (x_data[n] ** 5))) * (x_data[n] ** 5)

    grad_b /= len(x_data)
    grad_w_1 /= len(x_data)
    grad_w_2 /= len(x_data)
    grad_w_3 /= len(x_data)
    grad_w_4 /= len(x_data)
    grad_w_5 /= len(x_data)
    loss /= len(x_data)

    # # 用矩阵方式，效率却更低
    # loss = np.mean(np.square(y_d - (b + w_1 * x_d + w_2 * np.power(x_d, 2) + w_3 * np.power(x_d, 3)
    #                                 + w_4 * np.power(x_d, 4) + w_5 * np.power(x_d, 5))))
    # grad_b = np.mean(-2.0 * (y_d - (b + w_1 * x_d + w_2 * np.power(x_d, 2) + w_3 * np.power(x_d, 3)
    #                                 + w_4 * np.power(x_d, 4) + w_5 * np.power(x_d, 5))))
    # grad_w_1 = np.mean(-2.0 * (y_d - (b + w_1 * x_d + w_2 * np.power(x_d, 2) + w_3 * np.power(x_d, 3) +
    #                                   w_4 * np.power(x_d, 4) + w_5 * np.power(x_d, 5))) * x_d)
    # grad_w_2 = np.mean(-2.0 * (y_d - (b + w_1 * x_d + w_2 * np.power(x_d, 2) + w_3 * np.power(x_d, 3) +
    #                                   w_4 * np.power(x_d, 4) + w_5 * np.power(x_d, 5))) * np.power(x_d, 2))
    # grad_w_3 = np.mean(-2.0 * (y_d - (b + w_1 * x_d + w_2 * np.power(x_d, 2) + w_3 * np.power(x_d, 3) +
    #                                   w_4 * np.power(x_d, 4) + w_5 * np.power(x_d, 5))) * np.power(x_d, 3))
    # grad_w_4 = np.mean(-2.0 * (y_d - (b + w_1 * x_d + w_2 * np.power(x_d, 2) + w_3 * np.power(x_d, 3) +
    #                                   w_4 * np.power(x_d, 4) + w_5 * np.power(x_d, 5))) * np.power(x_d, 4))
    # grad_w_5 = np.mean(-2.0 * (y_d - (b + w_1 * x_d + w_2 * np.power(x_d, 2) + w_3 * np.power(x_d, 3) +
    #                                   w_4 * np.power(x_d, 4) + w_5 * np.power(x_d, 5))) * np.power(x_d, 5))

    lr_b += grad_b ** 2
    lr_w_1 += grad_w_1 ** 2
    lr_w_2 += grad_w_2 ** 2
    lr_w_3 += grad_w_3 ** 2
    lr_w_4 += grad_w_4 ** 2
    lr_w_5 += grad_w_5 ** 2

    # update param
    b -= lr / np.sqrt(lr_b) * grad_b
    w_1 -= lr / np.sqrt(lr_w_1) * grad_w_1
    w_2 -= lr / np.sqrt(lr_w_2) * grad_w_2
    w_3 -= lr / np.sqrt(lr_w_3) * grad_w_3
    w_4 -= lr / np.sqrt(lr_w_4) * grad_w_4
    w_5 -= lr / np.sqrt(lr_w_5) * grad_w_5

    # b_history.append(b)
    # w_1_history.append(w_1)
    # w_2_history.append(w_2)
    # w_3_history.append(w_3)
    # w_4_history.append(w_4)
    # loss_history.append(loss)
    # lr_b_history.append(lr / np.sqrt(lr_b))
    # lr_w_1_history.append(lr / np.sqrt(lr_w_1))
    # lr_w_2_history.append(lr / np.sqrt(lr_w_2))
    # lr_w_3_history.append(lr / np.sqrt(lr_w_3))
    # lr_w_4_history.append(lr / np.sqrt(lr_w_4))

    if i % 10000 == 0:
        b_history.append(b)
        w_1_history.append(w_1)
        w_2_history.append(w_2)
        w_3_history.append(w_3)
        w_4_history.append(w_4)
        w_5_history.append(w_5)
        loss_history.append(loss)

        i //= 10000
        print("Step %i, w_1: %0.18f, w_2: %0.18f, w_3: %0.18f, w_4: %0.18f, w_5: %0.18f, b: %0.18f, Loss: %0.4f" %
              (i, w_1_history[i], w_2_history[i], w_3_history[i], w_4_history[i], w_5_history[i], b_history[i], loss_history[i]))

        # 更新图
        fig.clf()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x_data, y_data)
        x_line = np.arange(0, 700, 1)
        y_line = b_history[i] + w_1_history[i] * x_line + w_2_history[i] * (x_line ** 2) + \
            w_3_history[i] * (x_line ** 3) + w_4_history[i] * (x_line ** 4) + w_5_history[i] * (x_line ** 5)
        ax1.plot(x_line, y_line)
        ax1.set_xlim([0, 700])
        ax1.set_ylim([0, 1700])
        plt.pause(0.1)

end = time.time()
print("大约需要时间：", end - start)

plt.ioff()
plt.show()
