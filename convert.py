import csv
import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

csv_path = 'test_data.csv'
y = []
img = []
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    header_row = next(reader)
    # print(header_row)
    for row in reader:
        y.append(int(row[0]))
        a = list(map(float, row[1:]))
        img.append(list(map(int, a)))

j = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k = 0
for i in range(len(y)):
    k = k + 1
    print(k)
    j[y[i]] += 1
    x = np.array(img[i])
    # x = ~x
    x = x * 255
    x = x.reshape(28, 28)
    # plt.imshow(x, cmap='gray')
    # plt.axis('off')
    # plt.savefig('test/' + str(y[i] + 1) + '/' + str(j[y[i]]) + '.png')
    image = Image.fromarray(np.uint8(x))
    image = image.convert("L")
    image.save('test/' + str(y[i] + 1) + '/' + str(j[y[i]]) + '.bmp')

# x = np.array(img[0])
# # x = ~x
# x = x * 255
# x = x.reshape(28, 28)
# # plt.imshow(x, cmap='gray')
# # plt.axis('off')
# # plt.savefig('test/' + str(y[0] + 1) + '/' + str(j[y[0]]) + '.png')
# image = Image.fromarray(np.uint8(x))
# image.save('test/' + str(y[0] + 1) + '/' + str(j[y[0]]) + '.png')
