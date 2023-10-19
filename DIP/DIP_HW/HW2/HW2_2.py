import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)


hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.subplot(2, 2, 1)
plt.title("原始图像")
plt.imshow(image, cmap='gray')
plt.subplot(2, 2, 2)
plt.title("原始直方图")
plt.plot(hist_original)
plt.xlim([0, 256])


equ = cv2.equalizeHist(image)


plt.subplot(2, 2, 3)
plt.title("均衡化后的图像")
plt.imshow(equ, cmap='gray')


hist_equalized = cv2.calcHist([equ], [0], None, [256], [0, 256])
plt.subplot(2, 2, 4)
plt.title("均衡化后的直方图")
plt.plot(hist_equalized)
plt.xlim([0, 256])


plt.tight_layout()
# plt.show()
plt.savefig('equalize.png')
