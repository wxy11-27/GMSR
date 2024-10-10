import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
import cv2


class HyperDatasetValid(udata.Dataset):
    def __init__(self, mode='valid'):
        if mode != 'valid':
            raise Exception("Invalid mode!", mode)
        # data_path = '/share/wangxinying/dataset/NTIRE2020_Clean/val_npz/'
        # data_names = glob.glob(os.path.join(data_path, '*.npz'))
        # data_path = '/share/wangxinying/dataset/Harvard/val_npz/'
        # data_names = glob.glob(os.path.join(data_path, '*.npz'))
        data_path = '/share/wangxinying/dataset/CAVE/val_npz/'
        data_names = glob.glob(os.path.join(data_path, '*.npz'))
        self.keys = data_names
        self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # mat = h5py.File(self.keys[index], 'r')
        # hyper = np.float32(np.array(mat['rad']))
        # hyper = np.transpose(hyper, [2, 1, 0])
        # hyper = torch.Tensor(hyper)
        # rgb = np.float32(np.array(mat['rgb']))
        #
        # rgb = np.transpose(rgb, [2, 1, 0])
        # rgb = torch.Tensor(rgb)
        # mat.close()
        # return rgb, hyper
        loaded_data = np.load(self.keys[index])
        try:
            hyper, rgb = loaded_data['rad'], loaded_data['rgb']
        except:
            import random
            index = random.randint(0, len(self.keys))
            loaded_data = np.load(self.keys[index])
            hyper, rgb = loaded_data['rad'], loaded_data['rgb']
        # hyper = loaded_data['rad']
        # hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = torch.Tensor(rgb)
        return rgb, hyper


class HyperDatasetTrain1(udata.Dataset):
    def __init__(self, mode='train'):
        if mode != 'train':
            raise Exception("Invalid mode!", mode)
        # data_path = '/share/wangxinying/dataset/NTIRE2020_Clean/train_npz/'
        # data_names = glob.glob(os.path.join(data_path, '*.npz'))
        # data_path = '/share/wangxinying/dataset/Harvard/train_npz/'
        # data_names = glob.glob(os.path.join(data_path, '*.npz'))
        data_path = '/share/wangxinying/dataset/CAVE/train_npz/'
        data_names = glob.glob(os.path.join(data_path, '*.npz'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        loaded_data = np.load(self.keys[index])
        try:
            hyper, rgb = loaded_data['rad'], loaded_data['rgb']
        except:
            import random
            index = random.randint(0, len(self.keys))
            loaded_data = np.load(self.keys[index])
            hyper, rgb = loaded_data['rad'], loaded_data['rgb']
        # hyper = loaded_data['rad']
        #hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)

        # rgb = loaded_data['rgb']
        #rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        return rgb, hyper

        # #mat = h5py.File(self.keys[index], 'r')
        # hyper = np.float32(np.array(mat['rad']))
        # hyper = np.transpose(hyper, [2, 1, 0])
        # hyper = torch.Tensor(hyper)
        # rgb = np.float32(np.array(mat['rgb']))
        # rgb = np.transpose(rgb, [2, 1, 0])
        # rgb = torch.Tensor(rgb)
        # mat.close()
        # return rgb, hyper


