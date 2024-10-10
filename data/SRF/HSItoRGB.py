############单张图像##############
# import cv2
# import torch
# import scipy.io as sio
# import matplotlib.pyplot as plt
# import numpy as np
# # 加载高光谱数据和光谱响应函数
# high_spectral_data = sio.loadmat('/share/wangxinying/dataset/Harvard/val_mat/imgf7.mat')['ref']
# spectral_response = sio.loadmat('/share/wangxinying/code/LPNet_4090_4/data/SRF/D700_zhuanzhi.mat')['filters']
#
# # 将 numpy 数组转换为 PyTorch 张量
# high_spectral_data = torch.tensor(high_spectral_data)
# spectral_response = torch.tensor(spectral_response)
#
# # 标准化高光谱数据
# high_spectral_data = high_spectral_data / high_spectral_data.max() * 255
#
# # 使用光谱响应函数生成 RGB 图像
# RGB_image = torch.matmul(high_spectral_data, spectral_response)
#
# # 将 RGB 图像标准化到 [0, 1] 范围内
# RGB_image = (RGB_image - RGB_image.min()) / (RGB_image.max() - RGB_image.min())
#
# # 将 PyTorch 张量转换为 NumPy 数组，并转换为 OpenCV 格式
# RGB_image_numpy = (RGB_image.numpy() * 255).astype(np.uint8)
# RGB_image_cv2 = cv2.cvtColor(RGB_image_numpy, cv2.COLOR_RGB2BGR)
#
# # 保存 RGB 图像
# cv2.imwrite('RGB_image_cv2.png', RGB_image_cv2)


import os
import cv2
import torch
import scipy.io as sio
import numpy as np

# 设置文件夹路径
folder_path = '/share/wangxinying/dataset/Harvard/val_mat/'
save_path = '/share/wangxinying/dataset/Harvard/val_rgb/'

# 获取文件夹中所有文件的列表
file_list = os.listdir(folder_path)

# 加载光谱响应函数
spectral_response = sio.loadmat('/share/wangxinying/code/LPNet_4090_4/data/SRF/D700_zhuanzhi.mat')['filters']
spectral_response = torch.tensor(spectral_response)

# 循环处理每个图像文件
for file_name in file_list:
    if file_name.endswith('.mat'):  # 如果文件是.mat格式的高光谱图像
        # 加载高光谱图像
        high_spectral_data = sio.loadmat(os.path.join(folder_path, file_name))['ref']
        high_spectral_data = torch.tensor(high_spectral_data)
        high_spectral_data = high_spectral_data / high_spectral_data.max() * 255

        # 使用光谱响应函数生成 RGB 图像
        RGB_image = torch.matmul(high_spectral_data, spectral_response)
        RGB_image = (RGB_image - RGB_image.min()) / (RGB_image.max() - RGB_image.min())

        # 将 PyTorch 张量转换为 NumPy 数组，并转换为 OpenCV 格式
        RGB_image_numpy = (RGB_image.numpy() * 255).astype(np.uint8)
        RGB_image_cv2 = cv2.cvtColor(RGB_image_numpy, cv2.COLOR_RGB2BGR)

        # 保存 RGB 图像
        output_file_name = os.path.splitext(file_name)[0] + '.png'
        cv2.imwrite(os.path.join(save_path, output_file_name), RGB_image_cv2)

