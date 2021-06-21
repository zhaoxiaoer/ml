# python 读取并显示图片的三种方法
# 1. opencv
# 2. matplotlib
# 3. PIL
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片

# import os
# print('11111')
# print(os.getcwd())
# print('2222')

# 1. 显示图片
im = Image.open('../data/hw3/food-11/testing/0001.jpg')
im.show()
# img = mpimg.imread('../data/hw3/food-11/testing/0001.jpg')  # img 是一个 np.array
# plt.imshow(img)
# plt.axis('off')
# plt.show()


# 2. 显示某个通道
# 将 PIL Image 图片转换为 numpy 数组
# im_array = np.array(im)  # 深复制
# # np.asarray(im)  # 浅复制
# print(im_array.shape)
# im_0 = Image.fromarray(im_array[:, :, 0])
# im_0.show()
# im_1 = Image.fromarray(im_array[:, :, 1])
# im_1.show()
# im_2 = Image.fromarray(im_array[:, :, 2])
# im_2.show()
# img_0 = img[:, :, 0]
# plt.imshow(img_0, cmap=plt.get_cmap('gray'))  # 默认是 'hot'，热量图
# plt.axis('off')
# plt.show()

# 3. 显示 numpy 数组
# 使用 matplotlib.image 读入图片数组
# 如果图片数组是 float32 型的 0-1 之间的数据，
# 需要转换为 PIL.Image 要求的 uint8 型的 0-255 的数据
# import matplotlib.image as mpimg
# test = mpimg.imread('../data/hw3/food-11/testing/0001.jpg')
# im = Image.fromarray(test)
# # im = Image.fromarray(np.uint8(test*255))
# im.show()

# 4. 保存 PIL 图片
# im_0.save('0001_1.jpg')
# plt.imshow(img_0, cmap=plt.get_cmap('gray'))  # 默认是 'hot'，热量图
# plt.axis('off')
# plt.savefig('0001_11.jpg')  # 保存 matplotlib 画出的图像
# # 直接保存数组
# np.save('0001_11', img_0)
# img_new = np.load('0001_11.npy')
# plt.imshow(img_new, cmap=plt.get_cmap('gray'))  # 默认是 'hot'，热量图
# plt.axis('off')
# plt.show()


# 5. RGB 转换为灰度图
# im_L = im.convert('L')
# im_L.show()

# 6. 对图像进行缩放
im_new = im.resize(size=(200, 200))
im_new.show()
img_new_array = np.array(im_new)
print(img_new_array.shape)



