import os
import cv2
import numpy as np

images_root = '/mnt/data/ziyiliu/test_images/CAMELYON16/gt'
save_root = '/mnt/data/ziyiliu/test_images/CAMELYON16/Resized_LowResolution_8'
if os.path.exists(save_root) is False:
    os.makedirs(save_root)

image_names = os.listdir(images_root)

for name in image_names:
    if name == 'prompt.csv':
        continue
    print(name)
    image = cv2.imread(os.path.join(images_root, name))
    resized_image = cv2.resize(image, (256//4, 256//4))
    # output = (resized_image * 255.0).round().astype(np.uint8)  # float32 to uint8
    cv2.imwrite(f'{save_root}/{name}', resized_image)