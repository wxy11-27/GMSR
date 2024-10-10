# import os
# import numpy as np
# import h5py
# def load_mat_files(folder_path):
#     data_list = []
#
#     # 遍历文件夹中的每个文件
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".mat"):
#             file_path = os.path.join(folder_path, file_name)
#
#             # 使用 h5py.File 加载 .mat 文件
#             with h5py.File(file_path, 'r') as mat_file:
#                 # 提取 'rad' 和 'rgb' 数据
#                 if 'rad' in mat_file and 'rgb' in mat_file:
#                     rad_data = np.array(mat_file['rad'], dtype=np.float32)
#                     rgb_data = np.array(mat_file['rgb'], dtype=np.float32)
#
#                     # 将数据添加到列表
#                     data_list.append({'rad': rad_data, 'rgb': rgb_data})
#                 else:
#                     print(f"Warning: Missing 'rad' or 'rgb' key in file {file_name}")
#
#     return data_list
#
# def save_numpy_data(data_list, output_folder):
#     # 创建输出文件夹
#     os.makedirs(output_folder, exist_ok=True)
#
#     # 遍历数据列表并保存每个文件
#     for i, data in enumerate(data_list):
#         file_path = os.path.join(output_folder, f"data_{i + 1}.npz")
#         np.savez(file_path, rad=data['rad'], rgb=data['rgb'])
#         print(f"Saved data to {file_path}")
#
# # 替换为实际的文件夹路径
# input_folder = "/share/wangxinying/code/MFormer/data/2020CLEAN/2020Trainmat/"
# output_folder = "/share/wangxinying/code/MFormer/data/2020CLEAN/2020Train/"
#
# # 调用函数加载数据
# loaded_data = load_mat_files(input_folder)
#
# # 调用函数保存 NumPy 数据到新文件夹
# save_numpy_data(loaded_data, output_folder)

import h5py
import numpy as np
import os

# 替换为实际的文件夹路径
mat_folder_path = "/media/titan3/File_JiaWenZ2/wxy/code/MFormer/data/2020CLEAN/2020Trainmat/"

# 创建保存 .npz 文件的文件夹，如果不存在的话
output_folder_path = "/media/titan3/File_JiaWenZ2/wxy/code/MFormer/data/2020CLEAN/2020Train/"
os.makedirs(output_folder_path, exist_ok=True)

# 获取文件夹中所有 .mat 文件
mat_files = [f for f in os.listdir(mat_folder_path) if f.endswith('.mat')]

# 遍历每个 .mat 文件
for mat_file in mat_files:
    file_path = os.path.join(mat_folder_path, mat_file)

    # 读取 .mat 文件
    with h5py.File(file_path, 'r') as mat_data:
        # 获取 'rad' 和 'rgb' 数据
        rad_data = np.array(mat_data['rad'])
        rgb_data = np.array(mat_data['rgb'])

        # 构建保存 .npz 文件的路径
        npz_file_path = os.path.join(output_folder_path, f"{mat_file[:-4]}.npz")

        # 将数据保存到对应的 .npz 文件
        np.savez(npz_file_path, rad=rad_data, rgb=rgb_data)
        print(mat_file)
print("Data has been successfully converted and saved as individual .npz files.")