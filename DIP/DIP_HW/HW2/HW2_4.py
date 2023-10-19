import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
equalized = cv2.equalizeHist(image)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title("均衡化后的图像")
plt.imshow(equalized, cmap='gray')

hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
plt.subplot(132)
plt.title("均衡化后的直方图")
plt.plot(hist_equalized)
plt.xlim([0, 256])

kernel = np.ones((3, 3), np.float32) / 9
filtered_image = cv2.filter2D(equalized, -1, kernel)

filtered_image = np.vstack((np.zeros_like(filtered_image[:1, :]), filtered_image, np.zeros_like(filtered_image[-1:, :])))

filtered_image = np.hstack((np.zeros_like(filtered_image[:, :1]), filtered_image, np.zeros_like(filtered_image[:, -1:])))

plt.subplot(133)
plt.title("盒状滤波响应图")
plt.imshow(filtered_image, cmap='gray')

plt.tight_layout()
# plt.show()
plt.savefig('filter.png')