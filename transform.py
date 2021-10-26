import csv
import os
import cv2

IMG_DIR = r"C:\Users\LYZ\Documents\Code\CNNNetwork\train"  # 在此处修改为测试图片的地址


def convert_img_to_csv(img_dir):
    with open(r'train.csv', 'w',
              newline='') as f:
        column_name = ['label']
        column_name.extend('pixel%d' % i for i in range(64 * 64))

        writer = csv.writer(f)
        writer.writerow(column_name)

        for i in range(1, 13):
            img_file_path = os.path.join(img_dir, str(i))
            img_list = os.listdir(img_file_path)

            for img_name in img_list:
                img_path = os.path.join(img_file_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = ~img
                image_data = [i - 1]
                image_data.extend(img.flatten())
                # print(image_data)
                writer.writerow(image_data)


if __name__ == "__main__":
    convert_img_to_csv(IMG_DIR)
