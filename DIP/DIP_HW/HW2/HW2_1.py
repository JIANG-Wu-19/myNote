from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open('test.png')

pixel_values = list(image.getdata())

hist, bins = np.histogram(pixel_values, bins=256, range=(0, 256))

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

plt.bar(bins[:-1], hist, width=1, color='gray', alpha=0.7)
plt.xlabel('灰度值')
plt.ylabel('出现次数')
plt.title('灰度直方图')
plt.grid(True)

# plt.show()

plt.savefig('hist1.png')
