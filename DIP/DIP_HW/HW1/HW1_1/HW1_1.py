from PIL import Image, ImageOps, ImageFilter

# 打开图像文件
input_image_path = 'lenna-RGB.tif'
output_image_path = 'output_image.tif'
image = Image.open(input_image_path)

# 灰度化
gray_image = ImageOps.grayscale(image)
gray_image.save('gray_image.tif')

# 二值化
threshold = 80  # 可以根据需要调整阈值
binary_image = gray_image.point(lambda p: p > threshold and 255)
binary_image.save('binary_image_80.tif')


for i in range(10, 180, 17):
    binary_image = gray_image.point(lambda p: p > i and 255)
    binary_image.save('binary_images/binary_image_' + str(i) + '.tif')

binary_image.close()

# 4*4均值化
blurred_image = gray_image.filter(ImageFilter.BLUR)
equalized_image = ImageOps.equalize(blurred_image)
equalized_image.save('equalized_image.tif')

# 关闭图像文件
image.close()
gray_image.close()
equalized_image.close()
