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

# 3. 使用Roberts算子、Sobel算子和Prewitt算子进行边缘检测
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

roberts_edges_x = cv2.filter2D(equalized, -1, roberts_x)
roberts_edges_y = cv2.filter2D(equalized, -1, roberts_y)

sobel_edges_x = cv2.filter2D(equalized, -1, sobel_x)
sobel_edges_y = cv2.filter2D(equalized, -1, sobel_y)

prewitt_edges_x = cv2.filter2D(equalized, -1, prewitt_x)
prewitt_edges_y = cv2.filter2D(equalized, -1, prewitt_y)


plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.title("Roberts X方向")
plt.imshow(np.abs(roberts_edges_x), cmap='gray')

plt.subplot(132)
plt.title("Sobel X方向")
plt.imshow(np.abs(sobel_edges_x), cmap='gray')

plt.subplot(133)
plt.title("Prewitt X方向")
plt.imshow(np.abs(prewitt_edges_x), cmap='gray')

plt.tight_layout()
# plt.show()
plt.savefig('X.png')

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.title("Roberts Y方向")
plt.imshow(np.abs(roberts_edges_y), cmap='gray')

plt.subplot(132)
plt.title("Sobel Y方向")
plt.imshow(np.abs(sobel_edges_y), cmap='gray')

plt.subplot(133)
plt.title("Prewitt Y方向")
plt.imshow(np.abs(prewitt_edges_y), cmap='gray')

plt.tight_layout()
# plt.show()
plt.savefig('Y.png')
