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

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 可以调整clipLimit和tileGridSize参数

equalized = clahe.apply(image)

plt.subplot(2, 2, 3)
plt.title("CLAHE均衡化后的图像")
plt.imshow(equalized, cmap='gray')

hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
plt.subplot(2, 2, 4)
plt.title("CLAHE均衡化后的直方图")
plt.plot(hist_equalized)
plt.xlim([0, 256])

plt.tight_layout()
# plt.show()

plt.savefig('CLAHE.png')